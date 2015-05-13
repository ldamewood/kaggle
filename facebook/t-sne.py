# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from tsne import bh_sne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier

#test = pd.read_csv('data/full_testing.csv', index_col=['bidder_id', 'bid_id'])
data = pd.read_csv('data/full_training.csv', index_col=['bidder_id', 'bid_id'])
data.fillna(0, inplace=True)
y = data['outcome'].astype('int').values
X = data.drop('outcome', axis=1).astype('float').values
features = data.drop('outcome',axis=1).columns.values
clf = ExtraTreesClassifier(n_estimators=500, verbose=True, n_jobs=-1)
clf.fit(X, y)
imp = pd.DataFrame({'feature': features, 'importance': clf.feature_importances_})
imp.sort('importance', inplace=True)

_, sample_idx = next(iter(StratifiedShuffleSplit(y, 1, test_size=0.01)))
print(sample_idx.shape)

y_sample = data.iloc[sample_idx]['outcome'].values.astype('int')
X_sample = data.iloc[sample_idx].drop(['outcome'], axis=1).values.astype('float')
X_2d = bh_sne(X_sample, theta=0.5)
plt.figure()
plt.scatter(X_2d[:,0], X_2d[:,1], c=y_sample)
plt.savefig('fb_1.png', format='png', dpi=600)