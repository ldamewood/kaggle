#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import pandas as pd
import numpy as np
import progressbar

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
import scipy.sparse

from facebook import FacebookCompetition

def to_ffm(df, outfile, ycol, num_columns = []):
    df = df.copy()
    one_based = True
    hasher = FeatureHasher(input_type='string', non_negative=True)
    bs = 2**10
    value_pattern = u'%d:%d:%.16g'
    line_pattern = u'%d %s\n'
    with open(outfile, 'w') as out:
        pb = progressbar.ProgressBar(maxval=(df.shape[0]+bs+1) // bs).start()
        for i in xrange((df.shape[0]+bs+1) // bs):
            pb.update(i)
            s = slice(i*bs, (i+1)*bs)
            if ycol in df.columns:
                Xh = np.asarray(df.iloc[s].drop([ycol], axis=1).drop(num_columns,axis=1).astype('str'))
                Xv = np.asarray(df.iloc[s][num_columns].astype('float'))
                y = df.iloc[s][ycol].values.astype('int')
            else:
                Xh = np.asarray(df.iloc[s].drop(num_columns,axis=1).astype('str'))
                Xv = np.asarray(df.iloc[s][num_columns].astype('float'))
                y = np.zeros((bs,))
            Xt = scipy.sparse.hstack([Xv,hasher.transform(Xh)]).tocsr()
            for j in xrange(Xt.shape[0]):
                span = slice(Xt.indptr[j], Xt.indptr[j+1])
                row = zip(range(len(Xt.indices[span])), Xt.indices[span], Xt.data[span])
                st = " ".join(value_pattern % (j + one_based, fe + one_based, x) for j, fe, x in row if np.isnan(x) == False)
                feat = (y[j], st)
                out.write((line_pattern % feat).encode('ascii'))
        pb.finish()

if __name__ == '__main__':
    drop_features = ['address', 'payment_account']
    mindex = ['bidder_id', 'bid_id']
    
    print('Loading data')
    bids = pd.read_csv('data/bids_with_features.csv')
    test = pd.read_csv(FacebookCompetition.__data__['test']).drop(drop_features, axis=1)
    full = pd.read_csv(FacebookCompetition.__data__['train']).drop(drop_features, axis=1)
    
    te = pd.merge(test, bids, how='left').set_index(mindex)
    tr = pd.merge(full, bids).set_index(mindex)
    te = te.fillna(0)
    tr = tr.fillna(0)
    count_features = [c for c in te.columns if 'num' in c]
    time_features = [c for c in te.columns if 'time' in c]
    scale_features = count_features+time_features
    te[scale_features] = np.log(1+te[scale_features])
    tr[scale_features] = np.log(1+tr[scale_features])
    
    ss = StandardScaler().fit(tr[scale_features].values)
    
    te.loc[:,scale_features] = ss.transform(te[scale_features].values)
    tr.loc[:,scale_features] = ss.transform(tr[scale_features].values)
    
    print('Converting testing to FFM')
    to_ffm(te, 'data/facebook.te.txt', scaler)

    y = tr['outcome'].values.astype('int')
    sss = StratifiedShuffleSplit(y, 1, random_state=0)
    data_idx, hold_idx = next(iter(sss))

    print('Converting holdout set to FFM')    
    to_ffm(tr.iloc[hold_idx], 'data/facebook.hd.txt', scaler)

    print('Converting training to FFM')
    skf = StratifiedKFold(y[data_idx], 10, random_state=0, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(skf):
        print('Fold {}'.format(i))
        to_ffm(tr.iloc[data_idx].iloc[train_idx], 'data/facebook.tr.{}.txt'.format(i), scaler)
        to_ffm(tr.iloc[data_idx].iloc[valid_idx], 'data/facebook.va.{}.txt'.format(i), scaler)