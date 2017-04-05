'''Training and submission'''

from __future__ import (absolute_import, unicode_literals, print_function, division)

import time
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import models.simple
import util.data
import util.preprocess


def model_from(args):
    return models.simple.PreprocessSimpleFeatires


def data_from(path):
    # Data loading
    t0 = time.time()
    data = util.data.load_data(path)
    print(f'Data loading took {time.time() - t0:.3f} seconds')

    # Data preprocessing
    t0 = time.time()
    data = util.preprocess.preprocess(data)
    print(f'Data preprocessing took {time.time() - t0:.3f} seconds')

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--train_path', type=str, default='data/train.csv')
    parser.add_argument('--test_path', type=str, default='data/test.csv')
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--model', type=str, default='simple')
    args = parser.parse_args()

    train, val = train_test_split(data_from(args.train_path), train_size=0.8)
    X, y = train[train.columns.drop(['is_duplicate', 'id', 'qid1', 'qid2'])], train.is_duplicate
    Xval, yval = val[val.columns.drop(['is_duplicate', 'id', 'qid1', 'qid2'])], val.is_duplicate

    model = RandomForestClassifier(n_jobs=4)
    model.fit(X, y)

    def results(X, y, prefix):
        predicted = model.predict_proba(X)
        score = model.score(X, y)
        loss = log_loss(y, predicted)
        try:
            accuracy = accuracy_score(y, predicted)
        except ValueError:
            accuracy = 0.0
        print(f'{prefix} score: {score:.2f}, loss: {loss:.4f}, accuracy: {accuracy:.2f}')

    results(X, y, 'Training')
    results(Xval, yval, 'Validation')

    if args.submit:
        print('Submitting..')
        test = data_from(args.test_path)
        X = test[test.columns.drop(['test_id'])]
        test['is_duplicate'] = model.predict_proba(X)[:, 1]
        test.to_csv(args.output, index=False, header=True, columns=['test_id', 'is_duplicate'])


if __name__ == '__main__':
    main()
