from __future__ import (absolute_import, division, print_function, unicode_literals)

import argparse
import os
import sys
import ipdb
import gc
import numpy as np
import pandas as pd

import tqdm

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import reporter as reporter_module

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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
def vectorize(data):
    tk = Tokenizer(nb_words=200000)

    tk.fit_on_texts(list(data.question1.values) + list(data.question2.values.astype(str)))
    x1 = [np.asarray(x, dtype=np.int32) if len(x) else np.array([0], dtype=np.int32) for x in tk.texts_to_sequences(data.question1.values.astype(str))]
    x2 = [np.asarray(x, dtype=np.int32) if len(x) else np.array([0], dtype=np.int32) for x in tk.texts_to_sequences(data.question2.values.astype(str))]

    return np.asarray(x1), np.asarray(x2), tk

@disk_cache('.test.npy')
def vectorize_with(tokenizer, data):
    tk = tokenizer
    x1 = [np.asarray(x, dtype=np.int32) if len(x) else np.array([0], dtype=np.int32) for x in tk.texts_to_sequences(data.question1.values.astype(str))]
    x2 = [np.asarray(x, dtype=np.int32) if len(x) else np.array([0], dtype=np.int32) for x in tk.texts_to_sequences(data.question2.values.astype(str))]

    return x1, x2, tokenizer


@disk_cache('.embeddings.npy')
def embedding_matrix(path, vocabulary):
    N = len(vocabulary) + 1
    DIMS = 300
    GLOVE_LINES_COUNT = 2196017
    embeddings_index = {}
    with open(path) as f:
        for line in tqdm.tqdm(f, total=GLOVE_LINES_COUNT):
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except ValueError:
                print('Problem with {}'.format(line))
                continue
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.random.randn(N, DIMS).astype('float32')
    for word, i in tqdm.tqdm(vocabulary.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


class CustomUpdater(chainer.training.StandardUpdater):
    def update_core(self):
        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)

        optimizer = self._optimizers['main']
        loss_func = self.loss_func or optimizer.target

        optimizer.update(loss_func, *in_arrays)


class CustomEvaluator(extensions.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        target = self._targets['main']
        eval_func = self.eval_func or target

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                eval_func(*in_arrays)
            summary.add(observation)

        return summary.compute_mean()


def converter(batch, device):
    def to_device(device, x):
        if device is None:
            return x
        elif device < 0:
            return chainer.cuda.to_cpu(x)
        else:
            return chainer.cuda.to_gpu(x, device, chainer.cuda.Stream.null)

    q1 = [to_device(device, np.asarray(b[0])) for b in batch]
    q2 = [to_device(device, np.asarray(b[1])) for b in batch]
    t = to_device(device, np.asarray([b[2] for b in batch]))

    return q1, q2, t


class SimpleModel(chainer.Chain):
    def __init__(self, layer_num, vocab_size, in_dim, hidden_dim, dropout=0.0):
        super().__init__(
            f_embedding=L.NStepLSTM(layer_num, in_dim, hidden_dim, dropout),
            b_embedding=L.NStepLSTM(layer_num, in_dim, hidden_dim, dropout),
            fc1=L.Linear(4 * hidden_dim, hidden_dim),
            fc2=L.Linear(hidden_dim, hidden_dim),
            fc3=L.Linear(hidden_dim, hidden_dim),
            fc4=L.Linear(hidden_dim, 2),
        )
        self.embed = L.EmbedID(vocab_size, in_dim)
        self.dropout = dropout
        self.train = True

    def __call__(self, x1, x2):
        sections = np.cumsum(np.array([len(x) for x in x1[:-1]], dtype=np.int32))
        x1 = F.split_axis(self.embed(F.concat(x1, axis=0)), sections, axis=0)

        _, _, q1_f = self.f_embedding(None, None, x1, self.train)
        _, _, q1_b = self.b_embedding(None, None, x1[::-1], self.train)

        q1_f = F.concat([x[-1, None] for x in q1_f], axis=0)
        q1_b = F.concat([x[-1, None] for x in q1_b], axis=0)

        sections = np.cumsum(np.array([len(x) for x in x2[:-1]], dtype=np.int32))
        x2 = F.split_axis(self.embed(F.concat(x2, axis=0)), sections, axis=0)

        _, _, q2_f = self.f_embedding(None, None, x2, self.train)
        _, _, q2_b = self.b_embedding(None, None, x2[::-1], self.train)

        q2_f = F.concat([x[-1, None] for x in q2_f], axis=0)
        q2_b = F.concat([x[-1, None] for x in q2_b], axis=0)

        x = F.concat([q1_f, q2_f, q1_b, q2_b], axis=1)
        # x = F.concat([q1_f, q2_f], axis=1)
        x = F.relu(self.fc1(F.dropout(x, self.dropout, self.train)))
        x = F.relu(self.fc2(F.dropout(x, self.dropout, self.train)))
        x = F.relu(self.fc3(F.dropout(x, self.dropout, self.train)))
        x = self.fc4(F.dropout(x, self.dropout, self.train))

        return x


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='data/test.csv')
    parser.add_argument('--train', type=str, default='data/train.csv')
    parser.add_argument('--embeddings', type=str, default='data/glove.840B.300d.txt')
    parser.add_argument('--out', type=str, default='result/')
    parser.add_argument('--resume', type=str)
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batch', '-b', type=int, default=128)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--submission', type=str, default='submission.gz')
    parser.add_argument('--model', type=str, default='')

    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(SEED)


    data = pd.read_csv(args.train)
    q1, q2, tokenizer = vectorize(data)
    labels = data.is_duplicate.values.astype('int32')

    embeddings = embedding_matrix(args.embeddings, tokenizer.word_index)

    model = L.Classifier(SimpleModel(4, len(embeddings), 300, 100, 0.4))
    model.predictor.embed.W.data = embeddings

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()
        model.predictor.embed.to_gpu()

    if args.submit:
        test = pd.read_csv(args.test)
        q1, q2, _ = vectorize_with(tokenizer, test)
        chainer.serializers.load_npz(args.model, model.predictor)

        predicted = np.zeros(len(q1))
        for pos in tqdm.tqdm(range(0, len(q1), args.batch)):
            bq1 = [chainer.cuda.to_gpu(x, device=args.gpu) for x in q1[pos:pos+args.batch]]
            bq2 = [chainer.cuda.to_gpu(x, device=args.gpu) for x in q2[pos:pos+args.batch]]

            predicted[pos:pos+args.batch] = F.softmax(model.predictor(bq1, bq2))[:, 1].data.get()

        output = pd.DataFrame(predicted, columns=['is_duplicate'])
        output['test_id'] = np.arange(len(predicted))
        output.to_csv(args.submission, index=False, header=True, columns=['test_id', 'is_duplicate'], compression='gzip')
        return 0

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    dataset = chainer.datasets.TupleDataset(q1, q2, labels)
    train, test = chainer.datasets.split_dataset_random(dataset, int(len(labels) * 0.9), seed=SEED)

    train_iter = chainer.iterators.SerialIterator(train, args.batch)
    test_iter = chainer.iterators.SerialIterator(test, args.batch,
                                                 repeat=False, shuffle=False)

    updater = CustomUpdater(train_iter, optimizer, device=args.gpu, converter=converter)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    eval_model = model.copy()  # Model with shared params and distinct states
    eval_rnn = eval_model.predictor
    eval_rnn.train = False
    trainer.extend(CustomEvaluator(test_iter, eval_model, device=args.gpu, converter=converter))

    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(
        extensions.snapshot_object(model.predictor, 'model_{.updater.iteration}'),
        trigger=chainer.training.triggers.MinValueTrigger('validation/main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar(update_interval=1), invoke_before_training=True)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


    return 0


if __name__ == '__main__':
    sys.exit(main())
