#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script uses the xgboost library to train and predict the rain data once
it is separated into time steps and missing values filled. It will automatically
do cross-validation and then create a submit file.

Try to be memory efficient by deleting variables that are no longer needed: it
requires around 32GB of ram depending on the number of extra features.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from rain import RainCompetition
from subprocess import check_call
from sklearn.cross_validation import StratifiedKFold

def score(y_pred, dm):
    y_real = dm.get_float_info('label')
    ids = dm.get_uint_info('extra_index')
    yp = RainCompetition.collapse(y_pred, ids)
    yr = RainCompetition.collapse(y_real, ids)
    return RainCompetition.score(yp, yr)

def train_xgboost(train_X, train_y, train_ids, valid_X, valid_y, valid_ids, rounds = 100):
    xg_train = xgb.DMatrix(train_X, label=train_y, missing=np.nan)
    xg_train.set_uint_info('extra_index', train_ids)
    xg_valid = xgb.DMatrix(valid_X, label=valid_y, missing=np.nan)
    xg_valid.set_uint_info('extra_index', valid_ids)
    ## setup parameters for xgboost
    evals = dict()
    params = {
            'eta': 0.2,
            'target': 'target',
            'max_depth': 11,
            'min_child_weight': 4,
            'subsample': 1,
            'min_loss_reduction': 1,
            'column_subsample': 1,
            'validation_set': xg_valid,
            'num_class' : 71,
            'objective': 'multi:softprob',
            'eval:metric': 'mlogloss',
            }
    
    watchlist = [ (xg_train, 'train'), (xg_valid, 'valid') ]
    bst = xgb.train(params, xg_train, rounds, watchlist, feval=score,
                    early_stopping_rounds=100, evals_result=evals)
    print(evals)
    return bst

#def predict_xgboost(bst, X):
#    xg = xgb.DMatrix( X, missing = np.nan )
#    prob_y = bst.predict( xg )
#    return prob_y

if __name__ == '__main__':
    print('Loading data...')
    train_df = pd.read_csv(RainCompetition.__train_split__, compression='gzip')
        
    print('Removing features without variance:')
    remove = []
    for col in train_df.columns:
        if train_df[col].std() < 1.e-5:
            remove.append(col)
            print('Removing column {}'.format(col))
            del train_df[col]
            
    X = train_df[[col for col in train_df.columns if col not in ['Id', 'Group', 'Expected']]].values
    y = np.array(train_df['Expected'], dtype = int).clip(0,70)
    ids = train_df['Id'].astype('int')
    
    del train_df
    
    xgbs, crp_score = [], []
    print('Fitting...')
    for i, (train_index, valid_index) in enumerate(StratifiedKFold(y, n_folds = 5, random_state=0)):
        print('Fold {}'.format(i))
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        i_train, i_valid = ids[train_index], ids[valid_index]
        clf, evals = train_xgboost(X_train, y_train, i_train, X_valid, y_valid, i_valid)
        break