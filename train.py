'''Training and submission'''

from __future__ import (absolute_import, unicode_literals, print_function, division)

import time
import argparse
import pandas as pd

from sklearn.model_selection import train_test_split

import models.simple
import util.data


def model_from(args):
    return models.simple.NoPreprocessSimpleFeatires


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--train_path', type=str, default='data/train.csv')
    parser.add_argument('--test_path', type=str, default='data/test.csv')
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--model', type=str, default='simple')
    args = parser.parse_args()

    t0 = time.time()
    train = util.data.load_data(args.train_path)
    print(f'Data loading took {time.time() - t0:.3f} seconds')
    model = model_from(args.model)()

    train, val = train_test_split(train, train_size=0.8)

    model.fit(train[['question1', 'question2']], train.is_duplicate)
    score = model.score(val[['question1', 'question2']], val.is_duplicate)
    print(f'Score is {score:.2f}')

    if args.submit:
        print('Submitting..')
        test = util.data.load_test_data(args.test_path)
        test['is_duplicate'] = model.predict(test[test.columns.drop('test_id')])
        test.to_csv(args.output, index=False, header=True, columns=['test_id', 'is_duplicate'])


if __name__ == '__main__':
    main()
