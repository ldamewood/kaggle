#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script uses the xgboost library to train and predict the rain data once
it is separated into time steps and missing values filled. It will automatically
do cross-validation and then create a submit file.

Try to be memory efficient by deleting variables that are no longer needed: it
requires around 32GB of ram depending on the number of extra features.
"""

from os.path import join
import random

import xgboost as xgb
import numpy as np
import pandas as pd
import rain

from subprocess import check_call

from sklearn.cross_validation import train_test_split
#from sklearn.ensemble import GradientBoostingClassifier

random.seed(0)

def load_dataset(filename):
    print('Loading data')
    df = pd.read_csv(filename, compression = 'gzip')
    
    if 'Expected' in df.columns:
        y = np.array(df.Expected, dtype = int).clip(0,70)
        del df['Expected']
    else:
        y = None
    
    ids = np.array(df.Id, dtype = int)
    del df['Id']
    
    # Remove derivative features
    for feature in df.columns:
        if '_deriv' in feature: del df[feature]
    
    X = np.array(df)
    
    return X, y, ids

def split_dataset(X, y, ids, split = 0.7):
    print('Splitting dataset')
    # This part is really slow for some reason
    uids = np.unique(ids).flatten().tolist()
    uids_train, uids_valid = train_test_split(uids, random_state = 0, train_size = 0.7)
    uids_train = set(uids_train)
    uids_valid = set(uids_valid)
    train_idx = np.array([i for i in ids if i in uids_train])
    valid_idx = np.array([i for i in ids if i in uids_valid])
    train_X = X[train_idx, :]
    train_y = y[train_idx]
    train_ids = ids[train_idx]
    valid_X = X[valid_idx, :]
    valid_y = y[valid_idx]
    valid_ids = ids[valid_idx]
    return train_X, train_y, train_ids, valid_X, valid_y, valid_ids

def train_xgboost(train_X, train_y, rounds = 1):
    print('Loading dataset into xgboost')
    xg_train = xgb.DMatrix( train_X, label=train_y, missing = np.nan )
    ## setup parameters for xgboost
    params = {
            'eta': 0.2,
            'target': 'target',
            'max_depth': 11,
            'min_child_weight': 4,
            'subsample': 1,
            'min_loss_reduction': 1,
            'column_subsample': 1,
            'validation_set': None,
            'num_class' : 71,
            'objective': 'multi:softprob',
            'eval:metric': 'mlogloss',
            }
    
    watchlist = [ (xg_train, 'train') ]
    print('Training')
    bst = xgb.train(params, xg_train, rounds, watchlist )
    return bst

def train_skgb(train_X, train_y):
    #bst = GradientBoostingClassifier()
    raise NotImplementedError

def score_xgboost(bst, valid_X, valid_y, valid_ids):
    xg_valid = xgb.DMatrix( valid_X, label=valid_y, missing = np.nan )
    prob_y = bst.predict( xg_valid )
    return rain.score_crp(prob_y, valid_y, valid_ids).sum()

def score_skgb(bst, valid_X, valid_y, valid_ids):
    prob_y = bst.predict_proba( valid_X )
    return rain.score_crp(prob_y, valid_y, valid_ids).sum()

def predict_testset_and_save(bst, testfile, outfile):
    test_X, _, test_ids = load_dataset(testfile)
    xg_test = xgb.DMatrix( test_X, missing = np.nan )
    prob_y = bst.predict( xg_test )
    probabilities = rain.group_and_mean(prob_y, test_ids)
    ids = np.array(rain.group_and_mean(test_ids, test_ids), dtype=int).flatten()
    crp = probabilities.cumsum(axis=1)[:,:70]
    out = pd.DataFrame(crp, index=ids, columns =  ['Predicted{}'.format(i) for i in range(70)])
    out.to_csv(outfile, index_label='Id', float_format = '%0.4f')
    check_call(['gzip', '-f',  outfile])

if __name__ == '__main__':
    X, y, ids = load_dataset(join(rain.datapath, 'train_with_na.csv.gz'))
    train_X, train_y, train_ids, valid_X, valid_y, valid_ids = split_dataset(X, y, ids)
    del X, y, ids
    if True:
        bst = train_xgboost(train_X, train_y, rounds = 16)
        score = score_xgboost(bst, valid_X, valid_y, valid_ids)
    else:
        bst = train_skgb(train_X, train_y)
        score = score_skgb(bst, valid_X, valid_y, valid_ids)
    print(score)
        
    print('Press [enter] to save test set answers to disk:')
    raw_input()

    outdf = predict_testset_and_save(bst, 
                join(rain.datapath, 'test_with_na.csv.gz'), 
                join(rain.datapath, 'out.csv'))
