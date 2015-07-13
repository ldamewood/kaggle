#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import pandas as pd
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

from facebook import FacebookCompetition
import xgboost as xgb

def auc_score(y_real, y_pred, ids):
    yr = collapse(y_real, ids)
    yp = collapse(y_pred, ids)
    return roc_auc_score(yr, yp)

def collapse(X, ids):
    df = pd.DataFrame({ 'X' : X, 'id': ids})
    df = df.groupby('id').mean()
    return df['X'].values

def do_train(data, bids, params, eval_size=0.05):
    mindex = ['bidder_id', 'bid_id']
    y = data['outcome'].values.astype('int')
    sss = StratifiedShuffleSplit(y, 1, test_size=eval_size, random_state=0)
    train_idx, evalu_idx = next(iter(sss))
    train = data.iloc[train_idx]
    evalu = data.iloc[evalu_idx]
    tr = pd.merge(train, bids).set_index(mindex)
    ev = pd.merge(evalu, bids).set_index(mindex)

    X_train = tr.drop('outcome',axis=1).values.astype('float')
    y_train = tr['outcome'].values.astype('int')
    X_eval = ev.drop('outcome',axis=1).values.astype('float')
    y_eval = ev['outcome'].values.astype('int')
    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    deval = xgb.DMatrix(X_eval, label=y_eval, missing=np.nan)
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train==1)
    params['scale_pos_weight'] = ratio
    
    def logregobj(preds, dtrain):
        labels = dtrain.get_label()
        preds = 1.0 / (1.0 + np.exp(-preds))
        grad = preds - labels
        hess = preds * (1.0-preds)
        return grad, hess
    
    def evalerror(preds, dtrain):
        labels = dtrain.get_label()
        if len(labels) == tr.shape[0]:
            ids = map(lambda x: x[0], tr.index.ravel())
        else:
            ids = map(lambda x: x[0], ev.index.ravel())
        return 'auc', -auc_score(labels, preds, ids)
    
    params['validation_set'] = deval
    evals = dict()
    watchlist = [ (dtrain, 'train'), (deval, 'eval') ]
    return xgb.train(params, dtrain, 1000, watchlist, feval=evalerror, obj=logregobj, 
                    early_stopping_rounds=25, evals_result=evals)

if __name__ == '__main__':
    drop_features = ['address', 'payment_account']
    mindex = ['bidder_id', 'bid_id']
    
    print('Loading data')
    bids = pd.read_csv('data/bids_with_features.csv')
    test = pd.read_csv(FacebookCompetition.__data__['test']).drop(drop_features, axis=1)
    full = pd.read_csv(FacebookCompetition.__data__['train']).drop(drop_features, axis=1)
    
    print('Splitting data/hold sets')
    sss = StratifiedShuffleSplit(full['outcome'].values.astype('int'), 1, test_size=0.1, random_state=0)
    data_idx, hold_idx = next(iter(sss))
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
            'eta': 0.03,
            'gamma': 0.0,
            'max_depth': 6,
            'min_child_weight': 1,
            'max_delta_step': 0,
            'subsample': 1.00,
            'colsample_bytree': 0.25,
            'objective': 'rank:pairwise',
            'eval:metric': 'auc',
            'silent': 1,
            'seed': 0,
            }
    
    print('Training')
    clfs, scores = [], []
#    y_preds_hold = np.zeros((n_iter, X_hold.shape[0], 2))
#    X_train = da.drop('outcome', axis=1).values.astype('float')
#    y_train = da['outcome'].values.astype('int')
#    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    
    skf = StratifiedKFold(data['outcome'].astype('int').values, n_iter, random_state=0, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(skf):
        print('Fold {}'.format(i))
        tran = data.iloc[train_idx]
        vald = data.iloc[valid_idx]        
        clf = do_train(tran, bids, params)
        va = vald.merge(bids).set_index(mindex)
        bidder_ids = map(lambda x: x[0], va.index.ravel())
        X_valid = va.drop('outcome',axis=1).values.astype('float')
        y_valid = va['outcome'].values.astype('int')
        dvalid = xgb.DMatrix(X_valid, label=y_valid, missing=np.nan)
        scores.append(auc_score(y_valid, clf.predict(dvalid, ntree_limit=clf.best_iteration), bidder_ids))
        print(scores[-1])
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