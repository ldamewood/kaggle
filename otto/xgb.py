#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition
import numpy as np
from sklearn.cross_validation import StratifiedKFold
#from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

#from xgboost import XGBClassifier
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
    encoder = LabelEncoder()
    X, y = OttoCompetition.load_data(train=True)
    y = encoder.fit_transform(y).astype('int32')
    num_classes = len(encoder.classes_)
    num_features = X.shape[1]

    bags, losses = [], []
    n_folds = 10
    print("Fitting...")
    for i, (train_index, valid_index) in enumerate(StratifiedKFold(y, n_folds = n_folds, random_state=0)):
        print('Fold {}'.format(i))
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        clf, evals = train_xgboost(X_train, y_train, X_valid, y_valid)
        bags.append(clf)
        losses.append(min(evals['valid']))

    losses = np.array(losses, dtype='float')
    print(np.mean(losses), np.std(losses))

    X_test, _ = OttoCompetition.load_data(train=False)
    X_test = xgb.DMatrix(X_test)
    y_preds = []
    for bag in bags:
        y_preds.append(bag.predict(X_test))
    y_pred = sum(y_preds)/n_folds

    OttoCompetition.save_data(y_pred)