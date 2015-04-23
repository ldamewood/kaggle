#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition, OttoTransform
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import copy

from xgboost import XGBClassifier

if __name__ == '__main__':
    train_df = OttoCompetition.load_data()
    labels = train_df['target']
    del train_df['target']
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)

    test_df = OttoCompetition.load_data(train=False)

    full_df = train_df.append(test_df)
    
    print("Fitting...")
    tr = OttoTransform(rescale=False)
    tr.fit(full_df)    
    X = tr.transform(train_df).values
    skf = StratifiedKFold(y, n_folds=20, random_state=0, shuffle=True)
    bag = []
    losses = []
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        bag.append(XGBClassifier(max_depth=11, learning_rate=0.1, n_estimators=1000, silent=True).fit(X_train,y_train))
        losses.append(log_loss(y_test, bag[-1].predict_proba(X_test)))
        print(losses[-1])