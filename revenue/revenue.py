#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import pandas as pd
import numpy as np
from kaggle import util

from time import mktime
from datetime import datetime as dt
from scipy.stats import randint as sp_randint
from subprocess import check_call

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingRegressor, BaggingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import RandomizedSearchCV

from os.path import join, dirname, realpath

class RevenueCompetition:
    __name__ = 'restaurant-revenue-prediction'
    __train__ = join(dirname(realpath(__file__)), 'data', 'train.csv')
    __test__ = join(dirname(realpath(__file__)), 'data', 'test.csv')
    
    @classmethod
    def load_data(cls, train = True):
        return pd.read_csv(cls.__train__ if train else cls.__test__, 
                           index_col = 'Id')
                           
    @classmethod
    def save_data(cls, y_pred, outfile, gzip=True):
        df = pd.DataFrame(y_pred, columns=['Prediction'],
                          index=np.arange(100000))
        df.to_csv(outfile, header = True, index_label='Id')
        if gzip:
            check_call(['gzip', outfile])

class RevenueTransform:
    def __init__(self, rescale = False):
        self.dictVectorizer_ = None
        self.rescale = rescale

    def get_params(self, deep=True):
        return { 'rescale': self.rescale }
    
    def set_params(self, **parameters):
        if 'rescale' in parameters:
            self.rescale = parameters['rescale']
        return self

    def fit(self, X, y=None):
        self.dictVectorizer_ = None
        self.transform(X)
        return self

    def fit_transform(self, df, y=None):
        self.dictVectorizer_ = None
        return self.transform(df)

    def transform(self, df, y=None):
        X = df.copy()

        # Unix timestamp of date (subtracted from Jan 1, 2015)
        X['Timestamp'] = X['Open Date'].apply(lambda x: 1420099200.0 - mktime(dt.strptime(x, "%m/%d/%Y").timetuple()))

        # The season
#        X['Season'] = X['Open Date'].apply(lambda x: util.get_season(dt.strptime(x, "%m/%d/%Y")))

        # Rescale parameters
        if self.rescale:
            X[[c for c in X.columns if c[0] == 'P']] = np.log(1.+X[[c for c in X.columns if c[0] == 'P']])

        del X['Open Date']

        # Vectorize these columns
        vectorize = ['Type','City','City Group']

        if self.dictVectorizer_ is None:
            self.dictVectorizer_ = DictVectorizer(sparse=False)
            self.dictVectorizer_.fit(X[vectorize].to_dict('records'))
        res = self.dictVectorizer_.transform(X[vectorize].to_dict('records'))
        for col in vectorize:
            del X[col]
        X = X.join(pd.DataFrame(res, columns=self.dictVectorizer_.get_feature_names()))

        return X