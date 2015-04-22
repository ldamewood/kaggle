#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition, OttoTransform
import numpy as np
from sklearn.ensemble import BaggingClassifier

from xgboost import XGBClassifier

if __name__ == '__main__':
    train_df = OttoCompetition.load_data()
    y = train_df['target']
    del train_df['target']

    test_df = OttoCompetition.load_data(train=False)

    full_df = train_df.append(test_df)
    
    print("Fitting...")
    tr = OttoTransform(rescale=False)
    tr.fit(full_df)    
    X = tr.transform(train_df).values
    clfs = XGBClassifier(max_depth=11, learning_rate=0.1, n_estimators=300)
    bags = BaggingClassifier(clfs, n_estimators=10, verbose=True)
    bags.fit(X,y)
    for i in range(10):
        bags.estimators_[i].classes_ = np.unique(y).tolist()
    y_pred = bags.predict_proba(X)