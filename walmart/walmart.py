# -*- coding: utf-8 -*-

import pandas as pd

class WalmartCompetition:
    __name__ = 'walmart'
    __data__ = {
        'train' : 'data/train.csv',
        'test' : 'data/test.csv',
        'key' : 'data/key.csv',
        'weather' : 'data/weather.csv'
    }

if __name__ == '__main__':
    
    train = pd.read_csv(WalmartCompetition.__data__['train'])
    key = pd.read_csv(WalmartCompetition.__data__['key'])
    weather = pd.read_csv(WalmartCompetition.__data__['weather'])