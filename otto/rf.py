# -*- coding: utf-8 -*-

from otto import OttoCompetition
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

np.random.seed(0)

if __name__ == '__main__':
    encoder = LabelEncoder()

    # Training data
    X, y = OttoCompetition.load_data(train=True)
    y = encoder.fit_transform(y).astype('int32')
    n_classes = np.unique(y).shape[0]
    n_features = X.shape[1]

    # Split a holdout set
    sss = StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=0)
    data_idx, hold_idx = next(iter(sss))
    X_data, X_hold = X[data_idx], X[hold_idx]
    y_data, y_hold = y[data_idx], y[hold_idx]
    print(hash(','.join(map(str, y_hold))))

    # Test data
    X_test, _ = OttoCompetition.load_data(train=False)

    np.random.seed(0)

    scores = []
    n_iter = 10
    y_hold_pred = np.zeros([n_iter, X_hold.shape[0], 9])
    y_test_pred = np.zeros([n_iter, X_test.shape[0], 9])
    for i, (train_index, valid_index) in enumerate(StratifiedKFold(y_data, n_iter, shuffle=True, random_state=0)):
        print('Fold {}'.format(i))
        X_train, X_valid = X_data[train_index], X_data[valid_index]
        y_train, y_valid = y_data[train_index], y_data[valid_index]
        clf = RandomForestClassifier(n_estimators=300, max_features=0.5, max_depth=4, oob_score=True, n_jobs=-1, random_state=0, class_weight='auto').fit(X_train, y_train)
        y_valid_pred = clf.predict_proba(X_valid)
        scores.append(log_loss(y_valid, y_valid_pred))
        print(scores[-1])
#        y_hold_pred[i,:,:] = clf.predict_proba(X_hold)
#        y_test_pred[i,:,:] = clf.predict_proba(X_test)
    print('CV:',np.mean(scores), np.std(scores))