# -*- coding: utf-8 -*-

import pandas as pd
from datetime import datetime, date
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

class RemoveVarianceless:
    def __init__(self, minvar=1.e-8):
        self.minvar = minvar
        self.varianceless_rows_ = []

    def get_params(self, deep=True):
        return {'minvar': self.minvar}
    
    def set_params(self, **parameters):
        self.minvar = parameters['minvar']
        return self

    def fit(self, X, y=None):
        self.varianceless_rows_ = [i for i,j in enumerate(X.var(axis=0)) if j < self.minvar]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X, y=None):
        return np.delete(X, self.varianceless_rows_, 1)

def records_groupby(dct, key):
    """
    Similar to pandas groupby method but for dicts of lists.
    """
    curval = dct[key][0]
    jstart = 0
    for j in xrange(len(dct[key])):
        if dct[key][j] != curval:
            yield {k: [v[i] for i in range(jstart, j)] for k,v in dct.iteritems()}
            jstart = j
            curval = dct[key][j]
    yield {k: [v[i] for i in range(jstart, j)] for k,v in dct.iteritems()}

seasons = [('winter', (date(1,  1,  1),  date(1,  3, 20))),
           ('spring', (date(1,  3, 21),  date(1,  6, 20))),
           ('summer', (date(1,  6, 21),  date(1,  9, 22))),
           ('autumn', (date(1,  9, 23),  date(1, 12, 20))),
           ('winter', (date(1, 12, 21),  date(1, 12, 31)))]

def get_season(now):
    if isinstance(now, datetime):
        now = now.date()

#    print(now)
    try:
        now = now.replace(year=1)
    except ValueError:
        # Happens in Feb sometimes :/
        now = now.replace(day=28,year=1)
    for season, (start, end) in seasons:
        if start <= now <= end:
            return season
    assert 0, 'never happens'

def scatter_figure(df, outfile='scatter.png'):
    pd.scatter_matrix(df, figsize=(df.shape[1],df.shape[1]))
    plt.savefig(outfile)