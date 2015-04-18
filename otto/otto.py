#!/usr/bin/env python

from os.path import join, dirname, realpath
from math import log
import random
import pandas as pd
import numpy as np
from subprocess import check_call

from kaggle import util
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint

random.seed(0)

class OttoCompetition:
    __name__ = 'otto-group-product-classification-challenge'
    __train__ = join(dirname(realpath(__file__)), 'data', 'train.csv')
    __test__ = join(dirname(realpath(__file__)), 'data', 'test.csv')
    
    @classmethod
    def load_data(cls, train = True):
        return pd.read_csv(cls.__train__ if train else cls.__test__, 
                           index_col = 'id')

    @classmethod
    def score(cls, y_true, y_pred, classes):
        """ Log Loss for Kaggle. """
        idx = [list(classes).index(y) for y in y_true]
        logloss = sum([-log(max(min(y[i],1. - 1.e-15),1.e-15)) for y,i in zip(y_pred,idx)])
        return logloss / len(y_true)
        
    @classmethod
    def save_data(cls, y_pred, outfile, gzip=True):
        df = pd.DataFrame(y_pred, columns=['Class_{}'.format(i) for i in range(1,10)],
                          index=np.arange(1,144369))
        df.to_csv(outfile, header = True, index_label='id')
        if gzip:
            check_call(['gzip', outfile])

class OttoTransform:
    def __init__(self):
        self.normalizer_ = None

    def transform(self, df):
        X = df.copy()

        normalize = [c for c in X.columns if c[0] == 'f']

        # Rescale parameters
        X[normalize] = np.log(1.+X[normalize])

        # Normalize these columns
        if self.normalizer_ is None:
            self.normalizer_ = MinMaxScaler()
            self.normalizer_.fit(X[normalize].values)
        X[normalize] = self.normalizer_.transform(X[normalize].values)

        return X

params = dict(
    nn__learning_rate=[0.0001,0.001,0.01,0.1],
    nn__n_components=sp_randint(5,50),
    nn__n_iter=sp_randint(10,1000),
    nn__random_state=[0],
    gb__n_estimators=sp_randint(10,1000),
    gb__max_depth=sp_randint(5,15),
    gb__min_samples_split=sp_randint(1,5),
    gb__learning_rate=[0.0001,0.001,0.01,0.1],
    gb__random_state=[0],
)

if __name__ == '__main__':
    tr = OttoTransform()
    train_df = tr.transform(OttoCompetition.load_data())
    
    cv = RandomizedSearchCV(Pipeline([('nn',BernoulliRBM()),
                                      ('gb',GradientBoostingClassifier())]),
                            params, n_iter=20, verbose=True, cv=5, n_jobs=4)

    reg = BaggingClassifier(cv, n_estimators=20, oob_score=True,
                           verbose=True, random_state=0)
    y = train_df['target'].values
    del train_df['target']
    X = train_df.values
    reg.fit(X,y)
    
    test_df = tr.transform(OttoCompetition.load_data(train=False))
    y_pred = reg.predict_proba(test_df.values)
    OttoCompetition.save_data(y_pred,'data/submit_20150417.csv')
