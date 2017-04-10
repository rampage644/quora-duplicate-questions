'''Training and submission'''
from __future__ import (absolute_import, unicode_literals, print_function, division)

import string
import spacy
import string
import nltk

import numpy as np
import pandas as pd
import itertools

from numpy.linalg import norm

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from Stemmer import Stemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity


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


    l = WordNetLemmatizer()
    def clean_nltk(x):
        x = str(x).lower()
        return ' '.join([l.lemmatize(t) for t in nltk.word_tokenize(x) if t not in stops and t not in string.punctuation])

    X.question1 = X.question1.apply(clean_simple)
    X.question2 = X.question2.apply(clean_simple)

    X['q1_words_num'] = X['question1'].map(words_count)
    X['q2_words_num'] = X['question2'].map(words_count)

    X['q1_length'] = X['question1'].map(length)
    X['q2_length'] = X['question2'].map(length)

    X['common_words'] = X[['question1', 'question2']].apply(common_words, axis=1)

    del X['question1']
    del X['question2']

    return X


tr = str.maketrans('', '', string.punctuation)
l = WordNetLemmatizer()
s = Stemmer('english')
def tokenize(text):
    return s.stemWords(nltk.word_tokenize(text.translate(tr)))


def preprocess_lsi(X, transformer):
    X.question1 = X.question1.apply(str)
    X.question2 = X.question2.apply(str)

    vectorizer = TfidfVectorizer(min_df=1, stop_words='english', tokenizer=tokenize)
    svd_model = TruncatedSVD(n_components=100, algorithm='randomized', n_iter=5)
    svd_transformer = Pipeline([('tfidf', vectorizer),
                                ('svd', svd_model)])

    g = itertools.chain(X.question1, X.question2)
    if not transformer:
        svd_transformer.fit(g)
        transformer = svd_transformer

    x1, x2 = transformer.transform(X.question1), transformer.transform(X.question2)
    # Memory inefficient
    # X['cosine_distance'] = np.diag(cosine_similarity(x1, x2))
    # cosine = lambda x, y: x.T.dot(y) / norm(x) / norm(y)
    # X['cosine_distance'] = np.array([cosine(x, y) if x.sum() and y.sum() else 0.0 for x, y in zip(x1, x2)])
    # X['l2_distance'] = np.linalg.norm(x1 - x2, axis=1)
    # X['l1_distance'] = np.abs(x1 - x2).sum(axis=1)

    del X['question1']
    del X['question2']
    ret = pd.DataFrame(np.concatenate([np.abs(x1 - x2), np.sqrt((x1-x2) ** 2),  x1 * x2], axis=1))
    try:
        ret['is_duplicate'] = X.is_duplicate
    except AttributeError:
        pass

    return ret, transformer
