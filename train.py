from __future__ import (absolute_import, division, print_function, unicode_literals)

import argparse
import json
import os
import sys
import gc
import numpy as np
import pandas as pd
import datetime
import functools

import tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import reporter as reporter_module
from chainer.dataset import DatasetMixin

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors

from util.preprocess import process_data

SEED = 12365172

def disk_cache(cache_path):
    def decorator(fn):
        def ret_fn(*args, **kwargs):
            try:
                with open(cache_path, 'rb') as f:
                    ret = np.load(f)
            except FileNotFoundError:
                ret = fn(*args, **kwargs)
                with open(cache_path, 'wb') as f:
                    np.save(f, ret)
            return ret
        return ret_fn
    return decorator


@disk_cache('.train.npy')
def vectorize(data, maxlen=40):
    tk = Tokenizer(nb_words=200000)
    tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))

    x1 = tk.texts_to_sequences(data.question1.values.astype(str))
    x1 = pad_sequences(x1, maxlen=maxlen)

    x2 = tk.texts_to_sequences(data.question2.values.astype(str))
    x2 = pad_sequences(x2, maxlen=maxlen)


    return x1, x2, tk

@disk_cache('.test.npy')
def vectorize_with(tokenizer, data, maxlen=40):
    tk = tokenizer
    x1 = tk.texts_to_sequences(data.question1.values.astype(str))
    x1 = pad_sequences(x1, maxlen=maxlen)

    x2 = tk.texts_to_sequences(data.question2.values.astype(str))
    x2 = pad_sequences(x2, maxlen=maxlen)

    return x1, x2, tokenizer


@disk_cache('.embeddings.npy')
def embedding_matrix(path, vocabulary):
    N = len(vocabulary) + 1
    DIMS = 300
    # Only word2vec embeddings are in binary file
    binary = True if 'Google' in path else False
    word2vec = KeyedVectors.load_word2vec_format(path, binary=binary)

    embedding_matrix = np.zeros([N, DIMS], dtype='float32')
    # embedding_matrix = np.random.randn(N, DIMS).astype('float32')
    with open('missing_words.txt', 'w') as ofile:
        for word, i in tqdm.tqdm(vocabulary.items()):
            if word in word2vec.vocab:
                embedding_matrix[i] = word2vec.word_vec(word)
            else:
                ofile.write(word)
                ofile.write('\n')

    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    return embedding_matrix


class SimpleModel(chainer.Chain):
    INPUT_DIM = 300
    def __init__(self, vocab_size, lstm_units, dense_units, lstm_dropout,
                 dense_dropout):
        super().__init__(
            q_embed=L.LSTM(self.INPUT_DIM, lstm_units),
            fc1=L.Linear(3 * lstm_units, dense_units),
            fc2=L.Linear(dense_units, dense_units),
            fc3=L.Linear(dense_units, dense_units),
            fc4=L.Linear(dense_units, 2),
            bn1=L.BatchNormalization(3 * lstm_units),
            bn2=L.BatchNormalization(dense_units),
            bn3=L.BatchNormalization(dense_units),
            bn4=L.BatchNormalization(dense_units),
        )
        self.embed = L.EmbedID(vocab_size, self.INPUT_DIM)
        self.lstm_dropout = lstm_dropout
        self.dense_dropout = dense_dropout
        self.train = True

    def __call__(self, x1, x2):
        x1 = self.embed(x1)
        x2 = self.embed(x2)

        self.q_embed.reset_state()
        seq_length = x1.shape[1]
        q1_f = q1_b = q2_f = q2_b = None
        for step in range(seq_length):
            q1_f = F.dropout(self.q_embed(x1[:, step, :]), self.lstm_dropout, self.train)

        self.q_embed.reset_state()
        for step in range(seq_length):
            q2_f = F.dropout(self.q_embed(x2[:, step, :]), self.lstm_dropout, self.train)

        x = F.concat([
            F.absolute(q1_f - q2_f),
            F.normalize(q1_f - q2_f),
            q1_f * q2_f], axis=1)
        x = self.fc1(F.dropout(F.relu(self.bn1(x, not self.train)), self.dense_dropout, self.train))
        x = self.fc2(F.dropout(F.relu(self.bn2(x, not self.train)), self.dense_dropout, self.train))
        x = self.fc3(F.dropout(F.relu(self.bn3(x, not self.train)), self.dense_dropout, self.train))
        x = self.fc4(F.dropout(F.relu(self.bn4(x, not self.train)), self.dense_dropout, self.train))

        return x

class PandasWrapper(DatasetMixin):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def get_example(self, i):
        return self.df.iloc[i].values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='data/test.csv')
    parser.add_argument('--train', type=str, default='data/train.csv')
    parser.add_argument('--glove_embeddings_path', type=str,
                        default='data/glove.840B.300d_gensim.txt')
    parser.add_argument('--fasttext_embeddings_path', type=str,
                        default='data/wiki.en.vec')
    parser.add_argument('--lexvec_embeddings_path', type=str,
                        default='data/lexvec.commoncrawl.300d.W.pos.vectors')
    parser.add_argument('--word2vec_embeddings_path', type=str,
                        default='data/GoogleNews-vectors-negative300.bin')
    parser.add_argument('--embeddings', type=str, default='glove')
    parser.add_argument('--out', type=str, default='result/')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batch', '-b', type=int, default=128)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--lstm',  type=int, default=150)
    parser.add_argument('--dense',  type=int, default=100)
    parser.add_argument('--lstm_dropout',  type=float, default=0.15)
    parser.add_argument('--dense_dropout',  type=float, default=0.15)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--desc',  type=str)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--submission', type=str, default='submission.gz')
    parser.add_argument('--model', type=str, default='')

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(SEED)
    out_dir = os.path.join(args.out, datetime.datetime.now().strftime('%m-%d-%H-%M'))

    data = process_data(pd.read_csv(args.train, encoding='utf-8'))
    q1, q2, tokenizer = vectorize(data)
    labels = data.is_duplicate.values.astype('int32')

    fn = {
        'glove': functools.partial(embedding_matrix, args.glove_embeddings_path),
        'word2vec': functools.partial(embedding_matrix, args.word2vec_embeddings_path),
        'lexvec': functools.partial(embedding_matrix, args.lexvec_embeddings_path),
        'fasttext': functools.partial(embedding_matrix, args.fasttext_embeddings_path),
    }[args.embeddings]
    embeddings = fn(tokenizer.word_index)

    model = L.Classifier(SimpleModel(
        len(embeddings), args.lstm, args.dense, args.lstm_dropout, args.dense_dropout))
    model.predictor.embed.W.data = embeddings

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        model.predictor.embed.to_gpu()

    if args.submit:
        test = process_data(pd.read_csv(args.test, encoding='utf-8'))
        q1, q2, _ = vectorize_with(tokenizer, test)
        chainer.serializers.load_npz(args.model, model.predictor)
        model.predictor.train = False

        predicted = np.zeros(len(q1))
        batch_size = 1024
        for pos in tqdm.tqdm(range(0, len(q1), batch_size)):
            bq1 = chainer.cuda.to_gpu(q1[pos:pos+batch_size], device=args.gpu)
            bq2 = chainer.cuda.to_gpu(q2[pos:pos+batch_size], device=args.gpu)

            preds = F.softmax(model.predictor(bq1, bq2))[:, 1].data.get()
            a, b = 0.165 / 0.37, (1 - 0.165) / (1 - 0.37)
            predicted[pos:pos+batch_size] = preds if not args.reweight else  a*preds / (a*preds + b*(1-preds))

        output = pd.DataFrame(predicted, columns=['is_duplicate'])
        output['test_id'] = np.arange(len(predicted))
        output.to_csv(args.submission, index=False, header=True, columns=['test_id', 'is_duplicate'], compression='gzip')
        return 0

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay))

    dataset = chainer.datasets.TupleDataset(q1, q2, labels)
    train, test = chainer.datasets.split_dataset_random(dataset, int(len(labels) * 0.9), seed=SEED)

    train_iter = chainer.iterators.SerialIterator(train, args.batch)
    test_iter = chainer.iterators.SerialIterator(test, 1024,
                                                 repeat=False, shuffle=False)

    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=out_dir)

    eval_model = model.copy()  # Model with shared params and distinct states
    eval_rnn = eval_model.predictor
    eval_rnn.train = False
    trainer.extend(extensions.Evaluator(test_iter, eval_model, device=args.gpu))

    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(
        extensions.snapshot_object(model.predictor, 'model_{.updater.iteration}'),
        trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar(update_interval=10), invoke_before_training=True)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    with open(os.path.join(out_dir, 'parameters.json'), 'w') as pfile:
        json.dump(vars(args), pfile)

    trainer.run()


    return 0


if __name__ == '__main__':
    sys.exit(main())
