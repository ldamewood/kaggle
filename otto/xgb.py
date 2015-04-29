#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

np.random.seed(0)

def train_xgboost(train_X, train_y, valid_X, valid_y, rounds = 1000):
    xg_train = xgb.DMatrix( train_X, label=train_y )
    xg_valid = xgb.DMatrix( valid_X, label=valid_y )
    ## setup parameters for xgboost
    evals = dict()
    params = {
            'eta': 0.1,
            'gamma': 0,
            'max_depth': 11,
            'min_child_weight': 1,
            'subsample': 1,
            'colsample_bytree': 0.5,
            'target': 'target',
            'validation_set': xg_valid,
            'num_class' : 9,
            'objective': 'multi:softprob',
            'eval:metric': 'mlogloss',
            'silent': 1,
            }
    
    watchlist = [ (xg_train, 'train'), (xg_valid, 'valid') ]
    bst = xgb.train(params, xg_train, rounds, watchlist,
                    early_stopping_rounds=100, evals_result=evals)
    return bst, evals

if __name__ == '__main__':
    X, y = OttoCompetition.load_data(train=True, tsne=False)
    le = LabelEncoder().fit(y)
    train_idx, valid_idx = next(iter(StratifiedKFold(y, 5)))
    X_train, X_valid = X[train_idx], X[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    clf, evals = train_xgboost(X_train, le.transform(y_train),
                               X_valid, le.transform(y_valid))
    print(min(evals['valid']))
    X_test, _ = OttoCompetition.load_data(train=False, tsne=False)
    y_pred = clf.predict(xgb.DMatrix(X_test))
    OttoCompetition.save_data(y_pred)