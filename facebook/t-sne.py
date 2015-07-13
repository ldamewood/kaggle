# -*- coding: utf-8 -*-

import pandas as pd
from tsne import bh_sne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#test = pd.read_csv('data/full_testing.csv', index_col=['bidder_id', 'bid_id'])
data = pd.read_csv('data/full_training.csv', index_col=['bidder_id', 'bid_id'])
data.fillna(0, inplace=True)
data = data.groupby(level=0).mean()
drop = [c for c in data.columns if 'prev' not in c and 'bid' in c]
data.drop(drop, axis=1, inplace=True)
y = data['outcome'].astype('int').values
X = data.drop('outcome', axis=1).astype('float').values
features = data.drop('outcome',axis=1).columns.values

X_2d = bh_sne(X, theta=0.5)
plt.figure()
plt.scatter(X_2d[:,0], X_2d[:,1], c=y)
plt.savefig('fb_1.png', format='png', dpi=600)