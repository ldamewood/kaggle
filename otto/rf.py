# -*- coding: utf-8 -*-

from otto import OttoCompetition, OttoScaler
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import progressbar
import pandas as pd

from os.path import basename

np.random.seed(0)

if __name__ == '__main__':
    scaler = OttoScaler()
    X, y = OttoCompetition.load_data(train=True)
    X = scaler.fit_transform(X).astype('float32')
    classes = np.unique(y)
    num_classes=len(classes)
    num_features = X.shape[1]
    holdout = OttoCompetition.holdout()
    X_hold, y_hold = X[holdout], y[holdout]
    X_data, y_data = np.delete(X,holdout,axis=0), np.delete(y,holdout,axis=0)
    X_test, _ = OttoCompetition.load_data(train=False)
    X_test = scaler.fit_transform(X_test).astype('float32')

    params = dict(
        n_estimators=200,
        n_jobs=-1,
        random_state=0,
        class_weight='auto',
        max_features='auto',
        max_depth=None,
        oob_score=True,
    )

    n_folds = 20
    print("Fitting...")
    y_preds, y_tests = [], []
    earlystop = 10
    for i, (train_index, valid_index) in enumerate(StratifiedKFold(y_data, n_folds = n_folds, random_state=2)):
        print('Fold {}'.format(i))
        X_train, X_valid = X_data[train_index], X_data[valid_index]
        y_train, y_valid = y_data[train_index], y_data[valid_index]
        clf = RandomForestClassifier(**params)
        y_scores = []
#        while True:
        clf.fit(X_train, y_train)
        y_train_score = log_loss(y_train, clf.predict_proba(X_train))
        y_scores.append(log_loss(y_valid, clf.predict_proba(X_valid)))
        print(y_train_score, y_scores[-1])
#            if len(y_scores) - np.argmin(y_scores) > earlystop:
#                break
#            clf.n_estimators += 10
#        clf.n_estimators -= 10*earlystop
        
        y_preds.append(clf.predict_proba(X_hold))
#        y_tests.append(clf.predict_proba(X_test))
    weights = OttoCompetition.prediction_weights(y_preds, y_hold)