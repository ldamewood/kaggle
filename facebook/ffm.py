#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

import pandas as pd
import numpy as np
import progressbar

from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import StandardScaler
import scipy.sparse

from facebook import FacebookCompetition

def to_ffm(df, outfile, scaler = None):
    df = df.copy()
    ycol = 'outcome'
    cnt_columns = [u'country_count', u'ip_count', u'url_count', 
                   u'device_count', u'merchandise_count', u'auction_count', 
                   u'num_prev_bids']
    num_columns = [u'time_since_last_bid', u'time'] + cnt_columns
    one_based = True
    hasher = FeatureHasher(input_type='string', non_negative=True)
    if scaler is None:
        scaler = StandardScaler()
    bs = 2**10
    value_pattern = u'%d:%d:%.16g'
    line_pattern = u'%d %s\n'
    df.loc[:,cnt_columns] = np.log(1+df[cnt_columns].values)
    df.loc[:,num_columns] = scaler.transform(df[num_columns].values)
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

def make_scaler(te, tr):
    cnt_columns = [u'country_count', u'ip_count', u'url_count', 
                   u'device_count', u'merchandise_count', u'auction_count', 
                   u'num_prev_bids']
    num_columns = [u'time_since_last_bid', 'time'] + cnt_columns
    df = pd.concat([te,tr])
    return StandardScaler().fit(df[num_columns].values)

if __name__ == '__main__':
    print('Loading bids')
    bids = pd.read_csv(FacebookCompetition.__data__['bids'])
    bids.loc[bids['country'].isnull(), 'country'] = 'unk'
    test = pd.read_csv(FacebookCompetition.__data__['test'])
    train = pd.read_csv(FacebookCompetition.__data__['train'])
    bidder_groups = bids.groupby('bidder_id')
    bidder_keys = bidder_groups.groups.keys()

    pb = progressbar.ProgressBar(maxval=test.shape[0]).start()
    print('Feature Engineering')
    for i, bidder_id in enumerate(test.bidder_id.values):
        pb.update(i)
        if bidder_id not in bidder_keys:
            for j in ['country', 'ip', 'url', 'device', 'merchandise', 'auction']:
                test.loc[i,'{}_count'.format(j)] = 0
                test.loc[i,'{}_group'.format(j)] = ''
        else:
            grp = bidder_groups.get_group(bidder_id)
            for j in ['country', 'ip', 'url', 'device', 'merchandise', 'auction']:
                test.loc[i,'{}_count'.format(j)] = grp[j].unique().size
                test.loc[i,'{}_group'.format(j)] = ' '.join(map(str, grp[j].unique()))
            bids.loc[grp.bid_id, 'time_since_last_bid'] = np.r_[0,np.diff(np.sort(grp['time'])/1.e10)]
            bids.loc[grp.bid_id, 'num_prev_bids'] = range(grp.shape[0])
    pb.finish()

    pb = progressbar.ProgressBar(maxval=train.shape[0]).start()
    for i, bidder_id in enumerate(train.bidder_id.values):
        pb.update(i)
        if bidder_id not in bidder_keys:
            for j in ['country', 'ip', 'url', 'device', 'merchandise', 'auction']:
                train.loc[i,'{}_count'.format(j)] = 0
                train.loc[i,'{}_group'.format(j)] = ''
        else:
            grp = bidder_groups.get_group(bidder_id)
            for j in ['country', 'ip', 'url', 'device', 'merchandise', 'auction']:
                train.loc[i,'{}_count'.format(j)] = grp[j].unique().size
                train.loc[i,'{}_group'.format(j)] = ' '.join(map(str, grp[j].unique()))
            bids.loc[grp.bid_id, 'time_since_last_bid'] = np.r_[0,np.diff(grp['time']/1.e10)]
            bids.loc[grp.bid_id, 'num_prev_bids'] = range(grp.shape[0])
    pb.finish()

    te = pd.merge(test, bids, how='left').set_index(['bidder_id', 'bid_id'])
    te = te.fillna(0)
    tr = pd.merge(train, bids).set_index(['bidder_id', 'bid_id'])
    scaler = make_scaler(tr, te)    
    
    print('Converting testing to FFM')
    to_ffm(te, 'data/facebook.te.txt', scaler)
    print('Converting training to FFM')
    y = tr['outcome'].values.astype('int')
    skf = StratifiedKFold(y, 10, random_state=0, shuffle=True)
    for i, (train_idx, valid_idx) in enumerate(skf):
        print('Fold {}'.format(i))
        to_ffm(tr.iloc[train_idx], 'data/facebook.tr.{}.txt'.format(i), scaler)
        to_ffm(tr.iloc[valid_idx], 'data/facebook.va.{}.txt'.format(i), scaler)