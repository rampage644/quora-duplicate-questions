'''Training and submission'''

from __future__ import (absolute_import, unicode_literals, print_function, division)

import pickle
import time
import argparse
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import models.simple
import util.data
import util.preprocess

# TODO: refactor
try:
    with open('transformer.pckl', 'rb') as ifile:
        transformer = pickle.load(ifile)
except FileNotFoundError:
    print('No transformer pickle, will train from scratch')
    transformer = None

def data_from(path, limit=None):
    # Data loading
    t0 = time.time()

    data = pd.read_csv(path, nrows=limit)
    print('Data loading took {:.3f} seconds'.format(time.time() - t0))

    # Data preprocessing
    global transformer
    t0 = time.time()
    data, transformer = util.preprocess.preprocess_lsi(data, transformer)
    if transformer:
        with open('transformer.pckl', 'wb') as ofile:
            pickle.dump(transformer, ofile)
    print('Data preprocessing took {:.3f} seconds'.format(time.time() - t0))

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--train_path', type=str, default='data/train.csv')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--test_path', type=str, default='data/test.csv')
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--search', action='store_true')
    args = parser.parse_args()

    train, val = train_test_split(data_from(args.train_path, args.limit), train_size=0.8)
    # X, y = train[train.columns.drop(['is_duplicate', 'id', 'qid1', 'qid2'])], train.is_duplicate
    # Xval, yval = val[val.columns.drop(['is_duplicate', 'id', 'qid1', 'qid2'])], val.is_duplicate
    X, y = train[train.columns.drop('is_duplicate')], train.is_duplicate
    Xval, yval = val[val.columns.drop('is_duplicate')], val.is_duplicate

    def results(model, X, y, prefix):
        predicted = model.predict_proba(X)
        score = model.score(X, y)
        loss = log_loss(y, predicted)
        try:
            accuracy = accuracy_score(y, predicted)
        except ValueError:
            accuracy = 0.0
        print('{} score: {:.2f}, loss: {:.4f}, accuracy: {:.2f}'.format(
            prefix, score, loss, accuracy
        ))

    model = SGDClassifier(loss='log', n_iter=1, n_jobs=-1, alpha=0.001)

    t0 = time.time()
    if args.search:
        parameters = {
            # 'penalty': ['l1', 'l2'],
            # 'C': [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3]
            # 'n_estimators': [50, 100, 200, 300, 500],
            'loss': ['log', 'modified_huber'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': np.logspace(-6, 6, 13),
            'n_iter': [1, 3, 5, 10]
        }

        grid_search = GridSearchCV(model, parameters, n_jobs=-1, verbose=1)

        print('Performing grid search...')
        grid_search.fit(X, y)
        print('done in {:.3f}s'.format(time.time() - t0))
        print()

        print('Best score: {:.3f}'.format(grid_search.best_score_))
        print('Best parameters set:')
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print('\t%s: %r' % (param_name, best_parameters[param_name]))
    else:
        model.fit(X, y)
        print('Model training took {:.3f} seconds'.format(time.time() - t0))
        results(model, X, y, 'Training')
        results(model, Xval, yval, 'Validation')



    if args.submit:
        print('Submitting..')
        X = data_from(args.test_path, args.limit)
        X['is_duplicate'] = model.predict_proba(X)[:, 1]
        X['test_id'] = np.arange(len(X))
        X.to_csv(args.output, index=False, header=True, columns=['test_id', 'is_duplicate'])


if __name__ == '__main__':
    main()


#%%
# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.stem.porter import PorterStemmer
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# import numpy as np
# import pandas as pd
# import nltk
# import itertools
# import util.data
# import string

# data = util.data.load_data('data/train.csv')
# data.question1 = data.question1.apply(str)
# data.question2 = data.question2.apply(str)

# stops = set(stopwords.words('english'))
# s = PorterStemmer()
# l = WordNetLemmatizer()
# def tokenize(text):
#     text = text.lower()
#     return [l.lemmatize(t) for t in nltk.word_tokenize(text) if t not in stops and t not in string.punctuation]

# vectorizer = TfidfVectorizer(min_df=1, tokenizer=tokenize)
# g = itertools.chain(data.question1, data.question2)

# vectorizer.fit(g)

# vectorizer.vocabulary_

# nltk.word_tokenize('askjdb asdbasiuy asjihdjhask asdbas')
# s.stem('oed')
