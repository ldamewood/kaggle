#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition

import pandas as pd
import numpy as np
import csv
from subprocess import check_call

from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import StratifiedKFold, train_test_split

class FFMClassifier:
    def __init__(self):
        pass

class FFMFormat:
    def __init__(self):
        self.field_index_ = None
        self.feature_index_ = None

    def get_params(self):
        pass
    
    def set_params(self, **parameters):
        pass

    def fit(self, df, y=None):
        self.field_index_ = {col: i for i,col in enumerate(df.columns)}
        self.feature_index_ = dict()
        last_idx = 0
        for col in df.columns:
            vals = np.unique(df[col])
            for val in vals:
                if np.isnan(val): continue
                name = '{}_{}'.format(col, val)
                if name not in self.feature_index_:
                    self.feature_index_[name] = last_idx
                    last_idx += 1
            self.feature_index_[col] = last_idx
            last_idx += 1
        return self

    def fit_transform(self, df, y=None):
        self.fit(df, y)
        return self.transform(df)

    def transform_row_(self, row):
        ffm = []
        for col,val in row.loc[row!=0].to_dict().iteritems():
            name = '{}_{}'.format(col, val)
            ffm.append('{}:{}:1'.format(self.field_index_[col], self.feature_index_[name]))
            ffm.append('{}:{}:{}'.format(self.field_index_[col], self.feature_index_[col], val))
        return ' '.join(ffm)

    def transform(self, df):
        return pd.Series({idx: self.transform_row_(row) for idx,row in df.iterrows()})

def load_fold_predictions(fold=0):
    np.vstack([np.loadtxt('data/ffm_predict_fold_{}_Class_{}.csv'.format(fold, i)) for i in xrange(1,10)]).T

if __name__ == '__main__':
    train_df = pd.read_csv(OttoCompetition.__train__, index_col='id')

    y = train_df['target']
    del train_df['target']

    tr = FFMFormat()
    train_ffm = tr.fit_transform(train_df)
    
    ffm_params = [
        '--norm',
        '-l', '0.005', # regularization
        '-r', '0.05', # learning rate
        '-k', '10', # factors
        '-t', '200', # num iterations
        '-s', '8', # num threads
    ]
    
    Xt, X_test, yt, y_test = train_test_split(train_ffm.values, y.values, test_size = 0.2)
    
    ll = []
    for i, (train_index, valid_index) in enumerate(StratifiedKFold(yt, n_folds = 10, random_state=0)):
        print('Fold {}'.format(i))
        X_train, X_valid = Xt[train_index], Xt[valid_index]
        y_train, y_valid = yt[train_index], yt[valid_index]
        valid_set = []
        lb = LabelBinarizer()
        ybin = lb.fit_transform(yt)
        for ylabel in lb.classes_:
            print(ylabel)
            tdf = pd.DataFrame(np.vstack([(y_train == ylabel).T,X_train.T]).T)
            vdf = pd.DataFrame(np.vstack([(y_valid == ylabel).T,X_valid.T]).T)
            train_file = './ffm/ffm_train_fold_{}_{}.csv'.format(i, ylabel)
            valid_file = './ffm/ffm_valid_fold_{}_{}.csv'.format(i, ylabel)
            model_file = './ffm/ffm_model_fold_{}_{}.csv'.format(i, ylabel)
            predt_file = './ffm/ffm_predt_fold_{}_{}.csv'.format(i, ylabel)
            tdf.to_csv(train_file, sep=" ", header=False, index=False,
                       quote=csv.QUOTE_NONE, quotechar=" ")
            vdf.to_csv(valid_file, sep=" ", header=False, index=False,
                       quote=csv.QUOTE_NONE, quotechar=" ")
            check_call(['ffm-train'] + ffm_params + ['-p', valid_file, train_file, model_file])
            check_call(['ffm-predict', valid_file, model_file, predt_file])
            valid_set.append(np.loadtxt(predt_file))
            yp = np.array(valid_set).T
            yp = (yp / yp.sum(axis=1)[:, np.newaxis])
        ll.append(OttoCompetition.score(y_valid, yp, lb.classes_.tolist()))