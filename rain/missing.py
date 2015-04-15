# -*- coding: utf-8 -*-
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

def train_missing_values(df, epochs = 0, init_fill = 'mean'):
    """
    1. Keep a record of the location of missing values.
    2. Fill in the missing values using the mean or median.
    3. Create a model for the missing values and predict new values.
    4. Return the models.
    """
    nan_mask = df.isnull().to_sparse()
    nan_counts = nan_mask.sum(axis=0)
    nan_counts.sort(ascending=False)

    if init_fill == 'median':
        df.fillna(df.median(), inplace = True)
    else:
        df.fillna(df.mean(), inplace = True)
    
    # Note: Extremely high memory usage here. (~24GB)
    for epoch in range(epochs):
        gbrs = {} # Reset trainer each time.
        for feature in nan_counts.index:
            if np.all(np.array(nan_counts[[feature]]) == 0): continue
            print('Training "{}" on epoch #{}'.format(feature, epoch))
            gb = nan_mask.groupby(feature)
            cols = [f for f in nan_counts.index if f not in ['Expected', feature]]
            train_X = df.loc[gb.groups[0.0],cols]
            train_y = df.loc[gb.groups[0.0],feature]
            test_X = df.loc[gb.groups[1.0],cols]
            # test_y is not used, but it may be used to calculate the change
            # in the missing feature values, which could be used as stopping
            # criteria for the epochs.
            test_y = df.loc[gb.groups[1.0],feature]
            gbrs[feature] = GradientBoostingRegressor(subsample = 0.5)
            gbrs[feature].fit(train_X, train_y)
            pred_y = gbrs[feature].predict(test_X)
            delta = np.linalg.norm(np.array(test_y) - pred_y) / len(pred_y)
            print('Change in feature: {}'.format(delta))
            df.loc[gb.groups[1.0],feature] = pred_y
    return gbrs
    
def predict_missing_values(df, gbrs, init_fill = 'mean'):
    """
    Use models for the features to predict the missing values.
    """
    nan_mask = df.isnull().to_sparse()
    nan_counts = nan_mask.sum(axis=0)
    nan_counts.sort(ascending=False)

    if init_fill == 'median':
        df.fillna(df.median(), inplace = True)
    else:
        df.fillna(df.mean(), inplace = True)

    for feature, gbr in gbrs.iteritems():
        print('Predicting for "{}"'.format(feature))
        if np.all(np.array(nan_counts[[feature]]) == 0): continue
        gb = nan_mask.groupby(feature)
        cols = [f for f in nan_counts.index if f not in ['Expected', feature]]
        test_X = df.loc[gb.groups[1.0],cols]
        df.loc[gb.groups[1.0],feature] = gbrs[feature].predict(test_X)