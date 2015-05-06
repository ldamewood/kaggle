#!/usr/bin/env python

from kaggle import KaggleCompetition

from os.path import join, dirname, realpath
from math import log
import random
import pandas as pd
import numpy as np
from subprocess import check_call
import datetime
from scipy.optimize import minimize

from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit

random.seed(0)

class OttoCompetition(KaggleCompetition):
    __full_name__ = 'otto-group-product-classification-challenge'
    __short_name__ = 'otto'
    __data__ = {
        'train': join(dirname(realpath(__file__)), 'data', 'train.csv'),
        'test': join(dirname(realpath(__file__)), 'data', 'test.csv'),
        'tsne': join(dirname(realpath(__file__)), 'data', 'tsne.csv'),
        'holdout': join(dirname(realpath(__file__)), 'data', 'holdout.csv'),
    }
    
    @classmethod
    def load_data(cls, train=True):
        infile = cls.__data__['train' if train else 'test']
        df = pd.read_csv(infile, index_col = 'id')
        
#        if tsne:
#            cols = ['V1','V2','V3']
#            tsne = pd.read_csv(cls.__data__['tsne'], index_col=0, header=False, names=cols)
#            if train:
#                for col in cols:
#                    df[col] = tsne.loc[df.index][col].values
#            else:
#                for col in cols:
#                    df[col] = tsne.loc[df.index+61878][col].values
        
        if 'target' in df.columns:
            X = df.drop('target', axis=1).values
            y = df['target'].values
        else:
            X = df.values
            y = None
        
        
        return X, y

    @classmethod
    def score(cls, y_true, y_pred, classes):
        """ Log Loss for Kaggle. """
        idx = [list(classes).index(y) for y in y_true]
        logloss = sum([-log(max(min(y[i],1. - 1.e-15),1.e-15)) for y,i in zip(y_pred,idx)])
        return logloss / len(y_true)
        
    @classmethod
    def save_data(cls, y_pred, gzip=True):
        df = pd.DataFrame(y_pred, columns=['Class_{}'.format(i) for i in range(1,10)],
                          index=np.arange(1,144369))
        outfile = '{}_{}.csv'.format(cls.__short_name__, 
                                        datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        df.to_csv(join(dirname(realpath(__file__)), 'data', outfile), header = True, index_label='id')
        if gzip:
            check_call(['gzip', outfile])
            print('Written to {}.gz'.format(outfile))
        else:
            print('Written to {}'.format(outfile))

    @classmethod
    def holdout(cls):
        return pd.read_csv(cls.__data__['holdout']).values.ravel()
    
    @classmethod
    def save_holdout(cls):
        infile = cls.__data__['train']
        df = pd.read_csv(infile, index_col = 'id')
        y = df['target']
        idx1, idx2 = next(iter(StratifiedShuffleSplit(y, test_size=0.05, random_state=0)))
        holdout = pd.DataFrame({'id':idx2})
        holdout.to_csv(cls.__data__['holdout'], index=False)

    @classmethod
    def prediction_weights(cls, y_preds, y_real):

        def log_loss_func(weights):
            weights /= np.sum(weights)
            yp = sum([w*y for w,y in zip(weights, y_preds)])
            return log_loss(y_real, yp)
        
        bounds = [(0,1)]*len(y_preds)
        starting = [0.1]*len(y_preds)
        res = minimize(log_loss_func, starting, method='L-BFGS-B', bounds=bounds)
        return res

class OttoScaler:
    def __init__(self, rescale = False):
        self.scaler_ = StandardScaler()
        X1, _ = OttoCompetition.load_data(train=True)
        X2, _ = OttoCompetition.load_data(train=False)
        X = np.log(1.+np.vstack([X1,X2]))
        self.scaler_.fit(X)

    def get_params(self, deep=True):
        return {}
    
    def set_params(self, **parameters):
        return self

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return self.scaler_.transform(X)

