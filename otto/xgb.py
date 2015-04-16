#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os.path import join, dirname, realpath

import sys

import otto
import pandas as pd

from sklearn.decomposition import NMF
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

from xgboost import XGBClassifier

if __name__ == '__main__':
    
    components = 20
    
    print('Loading train data...')
    data = pd.read_csv(otto.train, index_col = 'id')
    print('Number of training samples: %d' % len(data))
    feats = [col for col in data.columns if col not in ['target']]
    
    data['count_0'] = (data[feats]==0).sum(axis=1)
    data['count_1'] = (data[feats]==1).sum(axis=1)
    data['count_2'] = (data[feats]==2).sum(axis=1)
    
    print('Clustering...')
    names = ['cluster_{}'.format(i+1) for i in range(components)]
    clusterer = NMF(n_components = components)
    df_components = pd.DataFrame(clusterer.fit_transform(data[feats].values), columns = names)
    df_components.index += 1
    data = data.join(df_components)
    y = data['target'].values
    del data['target']
    X = data.values
    
    print('K-Fold Cross Validation')
    folds = 10
    clfs= [XGBClassifier(max_depth=11, learning_rate=0.1, n_estimators=300, subsample = 0.5) for _ in range(folds)]
    skf = StratifiedKFold(y, n_folds = folds, shuffle = True, random_state = 0)
    print("Learning...")
    for clf, (train_index, test_index) in zip(clfs,skf):
        clf.fit(X[train_index], y[train_index])
        y_pred = clf.predict_proba(X[test_index])
        print(log_loss(y[test_index], y_pred))