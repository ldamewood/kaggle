#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition
import numpy as np
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import xgboost as xgb

np.random.seed(0)

#params = {
#        'eta': 0.3,
#        'gamma': 0.5,
#        'max_depth': 4,
#        'min_child_weight': 4,
#        'subsample': 0.8,
#        'colsample_bytree': 0.5,
#        'target': 'target',
#        'num_class' : 9,
#        'objective': 'multi:softprob',
#        'eval:metric': 'mlogloss',
#        'silent': 1,
#}

params = {
        'eta': 0.1,
        'gamma': 0.5,
        'max_depth': 4,
        'min_child_weight': 4,
        'subsample': 1.0,
        'colsample_bytree': 0.5,
        'target': 'target',
        'num_class' : 9,
         'silent': 1,
}

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'logloss', log_loss(labels, preds)

def train(X, y, params):
    train_idx, eval_idx = next(iter(StratifiedShuffleSplit(y, 1, test_size=0.1, random_state=0)))
    X_train, X_eval = X[train_idx], X[eval_idx]
    y_train, y_eval = y[train_idx], y[eval_idx]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)
    params['validation_set'] = deval
    evals = dict()
    watchlist = [ (dtrain, 'train'), (deval, 'eval') ]
    return xgb.train(params, dtrain, 10000, watchlist, feval=evalerror, 
                    early_stopping_rounds=100, evals_result=evals)

if __name__ == '__main__':
    encoder = LabelEncoder()
    X, y = OttoCompetition.load_data(train=True)
    y = encoder.fit_transform(y).astype('int32')
    X_test, _ = OttoCompetition.load_data(train=False)
    le = LabelEncoder().fit(y)

    sss = StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=0)
    data_index, hold_index = next(iter(sss))

    X_data, X_hold = X[data_index], X[hold_index]
    y_data, y_hold = y[data_index], y[hold_index]
    print(hash(','.join(map(str, y_hold))))
    
        # train with 50%, validation with 5%
    scores = []
    n_iter = 10
#    y_hold_pred = np.zeros([n_iter, X_hold.shape[0], 9])
#    y_test_pred = np.zeros([n_iter, X_test.shape[0], 9])
    scores = np.zeros([10,n_iter])
    for j,css in enumerate([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
        np.random.seed(0)
        for i, (train_index, valid_index) in enumerate(StratifiedKFold(y_data, n_iter, shuffle=True, random_state=0)):
            print('Fold {}'.format(i))
            X_train, X_valid = X_data[train_index], X_data[valid_index]
            y_train, y_valid = y_data[train_index], y_data[valid_index]
            params['colsample_bytree'] = css
            clf = train(X_train, y_train, params)
            y_valid_pred = clf.predict(xgb.DMatrix(X_valid))
            scores[j,i] = log_loss(y_valid, y_valid_pred)
            print(scores[j,i])
#            y_hold_pred[i,:,:] = clf.predict(xgb.DMatrix(X_hold))
#            y_test_pred[i,:,:] = clf.predict(xgb.DMatrix(X_test))
    print(scores.mean(axis=1), scores.std(axis=1))
#    np.save('xgb_hold_pred', y_hold_pred)
#    np.save('xgb_test_pred', y_test_pred)