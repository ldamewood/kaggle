#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb

np.random.seed(0)

def train_xgboost(fold, rounds = 1000):
    folds = pd.HDFStore('data/train_folds.h5', mode='r')
    target = 'Expected'
    train_df = 'train_fold_{}'.format(fold)
    valid_df = 'valid_fold_{}'.format(fold)
    xg_train = xgb.DMatrix(folds.select(train_df).drop(target, axis=1).values.astype('float32'),
                           label=folds.select_column(train_df, target).values.clip(0,70).astype('int'),
                           missing=np.nan)
    xg_valid = xgb.DMatrix(folds.select(valid_df).drop(target, axis=1).values.astype('float32'),
                           label=folds.select_column(valid_df, target).values.clip(0,70).astype('int'),
                           missing=np.nan)
                
    ## setup parameters for xgboost
    evals = dict()
    params = {
            'eta': 0.1,
            'max_depth': 5,
            'subsample': 0.7,
            'colsample_bytree': 0.5,
            'target': 'target',
#            'validation_set': xg_valid,
            'num_class' : 71,
            'objective': 'multi:softprob',
            'eval:metric': 'mlogloss',
            'silent': 1,
            }
    
    watchlist = [ (xg_train, 'train'), (xg_valid, 'valid') ]
    print('Training...')
    bst = xgb.train(params, xg_train, 20, watchlist,
                    early_stopping_rounds=100, evals_result=evals)
    return bst, min(evals['valid'])

