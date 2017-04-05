'''Training and submission'''
from __future__ import (absolute_import, unicode_literals, print_function, division)

import string
import spacy


def common_words(x):
    q1, q2 = x
    return len(set(str(q1).lower().split()) & set(str(q2).lower().split()))


def words_count(question):
    return len(str(question).split())


def length(question):
    return len(str(question))


def preprocess(X):
    nlp = spacy.load('en')
    def clean(x):
        x = str(x)
        # losing info about personal pronouns here (due to lemmatization)
        return ' '.join([token.lemma_ for token in nlp(x) if token.is_alpha and not token.is_stop])

    translator = str.maketrans('', '', string.punctuation)
    def clean_simple(x):
        x = str(x).lower()
        return x.translate(translator)

    X.question1 = X.question1.apply(clean)
    X.question2 = X.question2.apply(clean)

    X['q1_words_num'] = X['question1'].map(words_count)
    X['q2_words_num'] = X['question2'].map(words_count)

    X['q1_length'] = X['question1'].map(length)
    X['q2_length'] = X['question2'].map(length)

    X['common_words'] = X[['question1', 'question2']].apply(common_words, axis=1)

    del X['question1']
    del X['question2']

    return X
