# -*- coding: utf-8 -*-

from otto import OttoCompetition, OttoScaler
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import log_loss

np.random.seed(0)

if __name__ == '__main__':
    encoder = LabelEncoder()
    scaler = OttoScaler()
    X, y = OttoCompetition.load_data(train=True)
    X = scaler.fit_transform(X).astype('float32')
    y = encoder.fit_transform(y).astype('int32')
    num_classes = len(encoder.classes_)
    num_features = X.shape[1]

    kncs, losses = [], []
    n_folds = 10
    print("Fitting...")
    for i, (train_index, valid_index) in enumerate(StratifiedKFold(y, n_folds = n_folds, random_state=0)):
        print('Fold {}'.format(i))
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        kncs.append(SVC().fit(X_train, y_train))
        y_pred = kncs[-1].predict_proba(X_valid)
        losses.append(log_loss(y_valid, y_pred))
        print(losses[-1])

    losses = np.array(losses, dtype='float')
    print(np.mean(losses), np.std(losses))

#    X_test, _ = OttoCompetition.load_data(train=False)
#    X_test = scaler.fit_transform(X_test).astype('float32')
#    y_preds = []
#    for knc in kncs:
#        y_preds.append(knc.predict(X_test))
#    y_pred = sum(y_preds)/n_folds
#
#    OttoCompetition.save_data(y_pred)