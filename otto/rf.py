# -*- coding: utf-8 -*-

from otto import OttoCompetition, OttoScaler
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

np.random.seed(0)

if __name__ == '__main__':
    encoder = LabelEncoder()
    scaler = OttoScaler()
    X, y = OttoCompetition.load_data(train=True)
    y = encoder.fit_transform(y).astype('int32')
    X = scaler.transform(X).astype('float32')
    n_classes = np.unique(y).shape[0]
    n_features = X.shape[1]
    data_idx, hold_idx = next(iter(StratifiedShuffleSplit(y, 1, test_size = 0.2, random_state=1)))
    X_data, X_hold = X[data_idx], X[hold_idx]
    y_data, y_hold = y[data_idx], y[hold_idx]

    params = dict(
        n_estimators=5000,
        n_jobs=-1,
        random_state=1,
        max_features='auto',
        max_depth=None,
        oob_score=True,
        bootstrap=True,
        class_weight='subsample',
        verbose=True,
    )

    print("Fitting...")
    df = pd.read_csv('data/test.csv', index_col='id')
    imps = []
    for train_idx, valid_idx in StratifiedKFold(y_data, 10, random_state=0, shuffle=True):
        X_train, X_valid = X_data[train_idx], X_data[valid_idx]
        y_train, y_valid = y_data[train_idx], y_data[valid_idx]
        imps.append(ExtraTreesClassifier(**params).fit(X_train, y_train).feature_importances_)