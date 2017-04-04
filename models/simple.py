'''Simple NLP models'''

from __future__ import (absolute_import, unicode_literals, print_function, division)

from sklearn.ensemble import RandomForestClassifier

from . import BaseModel


class DummyModel(BaseModel):
    def score(self):
        return 0.0


class NoPreprocessSimpleFeatires(BaseModel):
    '''Simple features inlcude:
     - number of words in each question
     - length of each question
     - common words count
    '''

    def common_words(_, x):
        q1, q2 = x
        return len(set(str(q1).lower().split()) & set(str(q2).lower().split()))

    words_count = lambda _, x: len(str(x).split())
    length = lambda _, x: len(str(x))

    def preprocess(self, X, y):
        X['q1_words_num'] = X['question1'].map(self.words_count)
        X['q2_words_num'] = X['question2'].map(self.words_count)
        X['q1_length'] = X['question1'].map(self.length)
        X['q2_length'] = X['question2'].map(self.length)
        X['common_words'] = X[['question1', 'question2']].apply(self.common_words, axis=1)

        del X['question1']
        del X['question2']

        return X, y

    def fit(self, X, y):
        X, y = self.preprocess(X, y)

        self.classifier = RandomForestClassifier(n_jobs=-1)
        self.classifier.fit(X, y)

    def score(self, X, y):
        X, y = self.preprocess(X, y)
        return self.classifier.score(X, y)

    def predict(self, X):
        X, _ = self.preprocess(X, None)
        return self.classifier.predict(X)


#%%
