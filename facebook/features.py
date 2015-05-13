# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer

from facebook import FacebookCompetition

def time_since_last(x):
    return np.r_[np.nan,np.diff(x/1.e10)]

def num_prev_log1px(x):
    return np.log(1+np.array(range(len(x)))) 

def num_prev(x):
    return np.array(range(len(x)))

def count(x):
    return x.count()

if __name__ == '__main__':  
    group_features = ['device', 'country', 'ip', 'url']

    print("Loading bids")
    bids = pd.read_csv(FacebookCompetition.__data__['bids'])

    print("Sorting by time")
    bids.sort('time', inplace=True)

    print("Normalizing countries")
    bids.loc[bids['country'].isnull(), 'country'] = 'zz'
    bids.loc[bids.country == 'gb', 'country'] = 'uk'

    print("Bidder groups")
    groups = bids.groupby('bidder_id')
    bids['bidder_time_since_last_bid'] = groups['time'].transform(time_since_last)
    bids['bidder_num_prev_bids'] = groups['time'].transform(num_prev)
    bids['bidder_total_num_bids'] = groups['time'].transform(count)

    for c in group_features:
        print("Bidder/{} groups".format(c))
        groups = bids.groupby(['bidder_id', 'auction', c])
        bids['bidder_time_since_last_bid_with_{}'.format(c)] = groups['time'].transform(time_since_last)
        bids['bidder_num_prev_bids_with_{}'.format(c)] = groups['time'].transform(num_prev)
        bids['bidder_total_num_{}'.format(c)] = groups['time'].transform(count)

    print("Auction groups")
    groups = bids.groupby('auction')
    bids['time_since_last_bid_in_auction'] = groups['time'].transform(time_since_last)
    bids['num_prev_bids_in_auction'] = groups['time'].transform(num_prev)

    for c in group_features:
        print("Bidder/auction/{} groups".format(c))
        groups = bids.groupby(['bidder_id', 'auction', c])
        bids['bidder_time_since_last_bid_with_{}_in_auction'.format(c)] = groups['time'].transform(time_since_last)
        bids['bidder_num_prev_bids_with_{}_in_auction'.format(c)] = groups['time'].transform(num_prev)
        bids['bidder_total_num_{}_in_auction'.format(c)] = groups['time'].transform(count)

    for c in group_features:
        bids.drop(c, axis=1, inplace=True)

    lb = LabelBinarizer()
    merch = pd.DataFrame(lb.fit_transform(bids.merchandise), columns=lb.classes_, index=bids.index)
    bids = bids.merge(merch, left_index=True, right_index=True).drop('merchandise', axis=1)
    
    bids = bids.drop(['auction', 'time'], axis=1).set_index('bid_id')
    bids.to_csv('data/bids_with_features.csv')
    
    bids.fillna(0, inplace=True)
    