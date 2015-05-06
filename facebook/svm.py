# -*- coding: utf-8 -*-

import pandas as pd

from facebook import FacebookCompetition

from sklearn.feature_extraction import FeatureHasher

if __name__ == '__main__':
    bids = pd.read_csv(FacebookCompetition.__data__['bids'], index_col='bid_id')
    train = pd.read_csv(FacebookCompetition.__data__['train'])
    test = pd.read_csv(FacebookCompetition.__data__['test'])
    tr = bids.merge(train)
    te = bids.merge(test)