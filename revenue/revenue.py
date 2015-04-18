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
from sklearn.preprocessing import MinMaxScaler
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
    def __init__(self):
        self.dictVectorizer_ = None
        self.normalizer_ = None

    def transform(self, df):
        X = df.copy()

        # Unix timestamp of date
        X['Timestamp'] = X['Open Date'].apply(lambda x: mktime(dt.strptime(x, "%m/%d/%Y").timetuple()))

        # The season
        X['Season'] = X['Open Date'].apply(lambda x: util.get_season(dt.strptime(x, "%m/%d/%Y")))

        # Rescale parameters
        X[[c for c in X.columns if c[0] == 'P']] = np.log(1.+X[[c for c in X.columns if c[0] == 'P']])

        del X['Open Date']

        # Vectorize these columns
        vectorize = ['Type','Season','City','City Group']

        if self.dictVectorizer_ is None:
            self.dictVectorizer_ = DictVectorizer(sparse=False)
            self.dictVectorizer_.fit(X[vectorize].to_dict('records'))
        res = self.dictVectorizer_.transform(X[vectorize].to_dict('records'))
        for col in vectorize:
            del X[col]
        X = X.join(pd.DataFrame(res, columns=self.dictVectorizer_.get_feature_names()))

        # Normalize these columns
        normalize = [c for c in X.columns if c[0] == 'P'] + ['Timestamp']
        if self.normalizer_ is None:
            self.normalizer_ = MinMaxScaler()
            self.normalizer_.fit(X[normalize].values)
        res = self.normalizer_.transform(X[normalize].values)
        for col in normalize:
            del X[col]
        X = X.join(pd.DataFrame(res, columns=normalize))

        return X

# Grid search parameters
params = dict(
    nn__learning_rate=[0.0001,0.001,0.01,0.1],
    nn__n_components=sp_randint(5,15),
    nn__n_iter=sp_randint(10,1000),
    nn__random_state=[0],
    gb__n_estimators=sp_randint(10,1000),
    gb__max_depth=sp_randint(5,15),
    gb__min_samples_split=sp_randint(1,5),
    gb__learning_rate=[0.0001,0.001,0.01,0.1],
    gb__loss=['ls', 'lad', 'huber'],
    gb__random_state=[0],
)

if __name__ == '__main__':
    tr = RevenueTransform()
    train_df = tr.transform(RevenueCompetition.load_data())
    cv = RandomizedSearchCV(Pipeline([('nn',BernoulliRBM()),
                                      ('gb',GradientBoostingRegressor())]),
                            params, n_iter=20, verbose=True, cv=10, n_jobs=-1)

    reg = BaggingRegressor(cv, n_estimators=200, oob_score=True,
                           verbose=True, random_state=0)
    y = train_df['revenue'].values
    del train_df['revenue']
    X = train_df.values
    reg.fit(X,y)
    
    test_df = tr.transform(RevenueCompetition.load_data(train=False))
    y_pred = np.exp(reg.predict(test_df.values))
    RevenueCompetition.save_data(y_pred, 'data/submit_20150417_1.csv')
