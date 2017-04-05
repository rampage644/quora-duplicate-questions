'''Data utilities'''

from __future__ import (absolute_import, unicode_literals, print_function, division)


import pandas as pd
import spacy


def load_data(path):
    return pd.read_csv(path)
