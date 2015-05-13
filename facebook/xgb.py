#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import pandas as pd
import numpy as np

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

from facebook import FacebookCompetition
import xgboost as xgb

def auc_score(y_real, y_pred, ids):
    df = pd.DataFrame({ 'y_real' : y_real, 'y_pred' : y_pred, 'bidder_id': ids})
    df = df.groupby('bidder_id').mean()
    return roc_auc_score(df['y_real'].values, df['y_pred'].values)   

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

if __name__ == '__main__':
    drop_features = ['address', 'payment_account']
    mindex = ['bidder_id', 'bid_id']
    
    print('Loading data')
    bids = pd.read_csv('data/bids_with_features.csv')
    test = pd.read_csv(FacebookCompetition.__data__['test']).drop(drop_features, axis=1)
    full = pd.read_csv(FacebookCompetition.__data__['train']).drop(drop_features, axis=1)
    
    print('Splitting data/hold sets')
    data_idx, hold_idx = next(iter(StratifiedShuffleSplit(full['outcome'].values.astype('int'), 1, random_state=0)))
    data = full.iloc[data_idx]
    hold = full.iloc[hold_idx]

    print('Merging test/hold with bids')
    te = pd.merge(test, bids, how='left').set_index(mindex)
    hd = pd.merge(hold, bids).set_index(mindex)
    da = pd.merge(data, bids).set_index(mindex)

    X_hold = hd.drop('outcome', axis=1).values.astype('float')
    y_hold = hd['outcome'].values.astype('int')
    X_test = te.values.astype('float')
    
    xg_hold = xgb.DMatrix(X_hold, label=y_hold, missing=np.nan)
    xg_test = xgb.DMatrix(X_test, missing=np.nan)
    
    n_iter = 10
    rounds = 1000
    
    params = {
            'eta': 0.0001,
            'gamma': 0.5,
            'max_depth': 20,
            'min_child_weight': 10,
            'subsample': 1.00,
            'colsample_bytree': 0.50,
            'objective': 'binary:logistic',
            'eval:metric': 'auc',
            'silent': 1,
            }
    
    print('Training')
    clfs, scores = [], []
#    y_preds_hold = np.zeros((n_iter, X_hold.shape[0], 2))
#    X_train = da.drop('outcome', axis=1).values.astype('float')
#    y_train = da['outcome'].values.astype('int')
#    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    
    skf = StratifiedKFold(data['outcome'].astype('int').values, n_iter, random_state=0, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(skf):
#        print('Fold {}'.format(i))
        tr = pd.merge(data.iloc[train_idx], bids).set_index(mindex)
        va = pd.merge(data.iloc[valid_idx], bids).set_index(mindex)

        X_train = tr.drop('outcome', axis=1).values.astype('float')
        y_train = tr['outcome'].values.astype('int')
        X_valid = va.drop('outcome', axis=1).values.astype('float')
        y_valid = va['outcome'].values.astype('int')
        bidder_ids = map(lambda x: x[0], va.index.ravel())
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train==1)
        
        params['scale_pos_weight'] = ratio        
        dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
        dvalid = xgb.DMatrix(X_valid, missing=np.nan)
        clf = xgb.train(params, dtrain, num_boost_round=rounds, early_stopping_rounds=25)
        scores.append(auc_score(y_valid, clf.predict(dvalid), bidder_ids))
#        print(scores[-1])
    print(np.mean(scores), np.std(scores))
    
    # bs = 2**15   

#    y_preds_hold = np.zeros([n_iter, X_hold.shape[0], 2])
##    hold_X = xgb.DMatrix( X_hold, missing=np.nan )
#    for i, clf in enumerate(clfs):
#        y_preds_hold[i, :, 0] = clf.predict_proba(X_hold)
#        y_preds_hold[i, :, 1] = 1. - y_preds_hold[i, :, 0]
#        
#    A = np.exp(np.log(y_preds_hold).mean(axis=0))
#    row_sums = A.sum(axis=1)
#    A = A / row_sums[:, np.newaxis]
#    
#    B = y_preds_hold.mean(axis=0)
#    
#    print(auc_score(y_hold, A[:,0], map(lambda x: x[0], hd.index)))
#    print(auc_score(y_hold, B[:,0], map(lambda x: x[0], hd.index)))
#    
#    y_preds_test = np.zeros([n_iter, X_test.shape[0], 2])
##    test_X = xgb.DMatrix( X_test, missing=np.nan )
#    for i, clf in enumerate(clfs):
#        y_preds_test[i, :, 0] = clf.predict_proba(X_test)
#        y_preds_test[i, :, 1] = 1. - y_preds_test[i, :, 0]
#        
#    A = np.exp(np.log(y_preds_test).mean(axis=0))
#    row_sums = A.sum(axis=1)
#    A = A / row_sums[:, np.newaxis]
#    
#    B = y_preds_test.mean(axis=0)
#    
#    df = pd.DataFrame({'prediction': A[:,0], 'bidder_id': map(lambda x: x[0], te.index)})
#    df.groupby('bidder_id').mean().to_csv('data/facebook_20150511_2.csv')