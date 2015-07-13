# -*- coding: utf-8 -*-
from os.path import join, dirname, realpath
import pandas as pd
from kaggle import KaggleCompetition
import numpy as np


class FacebookCompetition(KaggleCompetition):
    __short_name__ = 'facebook'
    __full_name__ = 'facebook-recruiting-iv-human-or-bot'
    __data_path__ = 'input'
    __data__ = {
        'train': join(dirname(realpath(__file__)), __data_path__, 'train.csv'),
        'test': join(dirname(realpath(__file__)), __data_path__, 'test.csv'),
        'bids': join(dirname(realpath(__file__)), __data_path__, 'bids.csv'),
    }


if __name__ == '__main__':
    bids = pd.read_csv(FacebookCompetition.__data__['bids'], index_col='bid_id')
    train = pd.read_csv(FacebookCompetition.__data__['train'])
    test = pd.read_csv(FacebookCompetition.__data__['test'])
    tr = bids.merge(train)
    te = bids.merge(test)

    cols = ['auction', 'merchandise', 'device', 'country', 'ip', 'url']