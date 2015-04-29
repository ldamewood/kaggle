#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rain import RainCompetition

import numpy as np
import pandas as pd
import xgboost as xgb

np.random.seed(0)

def load_data(shuffle=True):
    print('Loading training set...')
    df = pd.read_csv(RainCompetition.__data__['train_ravel_missing'],
                     index_col=['Id', 'Index'])
        
    if shuffle:
        print('Shuffle data...')
        n_ids = len(df.index.levels[0].values)
        neworder = np.array(range(n_ids))
        np.random.shuffle(neworder)
        newindex = df.index[ np.argsort(neworder[df.index.labels[0]],
                                        kind='mergesort') ]
        return df.reindex(newindex)
    else:
        return df

def train_xgboost(df, test_size = 0.2, rounds = 1000):
    print('Training...')
    features = [col for col in df.columns if col not in ['Expected']]
    target = 'Expected'
    split = int(df.shape[0]*test_size)
    xg_train = xgb.DMatrix( df.iloc[:split][features].values, label=df.iloc[:split][target].values.clip(0,70).astype('int32'), missing=np.nan )
    xg_valid = xgb.DMatrix( df.iloc[split:][features].values, label=df.iloc[split:][target].values.clip(0,70).astype('int32'), missing=np.nan )
                
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
            'num_class' : 71,
            'objective': 'multi:softprob',
            'eval:metric': 'mlogloss',
            'silent': 1,
            }
    
    watchlist = [ (xg_train, 'train'), (xg_valid, 'valid') ]
    bst = xgb.train(params, xg_train, rounds, watchlist, 
                    early_stopping_rounds=100, evals_result=evals)
    print(min(evals['valid']))
    return bst

def do_predict(clf, outfile):
    print('Predictions...')
    df_it = pd.read_csv(RainCompetition.__data__['test_ravel_missing'],
                        index_col=['Group','Index'], iterator=True,
                        chunksize=2048)
    y_preds = []
    ycols = ['Predicted{}'.format(i) for i in range(70)]
    for i,cnk in enumerate(df_it):
        print(i)
        features = [col for col in cnk.columns if col not in ['Expected','Id']]
        xg_test = xgb.DMatrix(cnk[features].values)
        y_pred = pd.DataFrame(clf.predict(xg_test)[:,:70], columns=ycols)
        y_pred['Id'] = cnk['Id'].values
        y_preds.append(y_pred)
    y_pred = pd.concat(y_preds).groupby('Id').mean().cumsum(axis=1)
    y_pred.to_csv(outfile, index_label='Id', float_format='%5.3f')

if __name__ == '__main__':
    df = load_data(shuffle=True)
    model = train_xgboost(df)
    do_predict(model, 'data/rain_20150430.csv')