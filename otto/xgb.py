#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition
import numpy as np
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
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
    X, y = OttoCompetition.load_data(train=True)
    X_test, _ = OttoCompetition.load_data(train=False)
    le = LabelEncoder().fit(y)

    all_hold = []
    all_hold_predict = []
    all_test_predict = []
    all_weights = []
    # 20% holdout
    for i, (data_index, hold_index) in enumerate(StratifiedKFold(y, n_folds = 5, random_state=0)):
        X_data, X_hold = X[data_index], X[hold_index]
        y_data, y_hold = y[data_index], y[hold_index]
        y_hold_predict = []
        y_test_predict = []
        # train with 50%, validation with 5%
        for j, (train_index, valid_index) in enumerate(StratifiedShuffleSplit(y_data, 20, test_size = 0.05, train_size = 0.5, random_state=0)):
            X_train, X_valid = X_data[train_index], X_data[valid_index]
            y_train, y_valid = y_data[train_index], y_data[valid_index]

            clf, evals = train_xgboost(X_train, le.transform(y_train),
                                       X_valid, le.transform(y_valid))

            y_hold_predict.append(clf.predict(xgb.DMatrix(X_hold)))
            y_test_predict.append(clf.predict(xgb.DMatrix(X_test)))
            
        weights = OttoCompetition.prediction_weights(y_hold_predict, y_hold)
        all_hold.append(y_hold)
        all_hold_predict.append(y_hold_predict)

    