#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition
import numpy as np
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import xgboost as xgb

from scipy.optimize import minimize

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
        'max_depth': 10,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'target': 'target',
        'num_class' : 9,
        'objective': 'multi:softprob',
        'silent': 1,
}

def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'logloss', log_loss(labels, preds)

def train(X, y, params, deval, eval_size=0.1):
    if deval is None and eval_size > 0:
        sss = StratifiedShuffleSplit(y, 1, test_size=eval_size, random_state=0)
        train_idx, eval_idx = next(iter(sss))
        X_train, X_eval = X[train_idx], X[eval_idx]
        y_train, y_eval = y[train_idx], y[eval_idx]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = xgb.DMatrix(X_eval, label=y_eval)
    else:
        dtrain = xgb.DMatrix(X, label=y)
    params['validation_set'] = deval
    evals = dict()
    watchlist = [ (dtrain, 'train'), (deval, 'eval') ]
    return xgb.train(params, dtrain, 10000, watchlist, feval=evalerror, 
                    early_stopping_rounds=100, evals_result=evals)

if __name__ == '__main__':
    # Set this to True to predict the calibration and test set.
    # Set to False to do Kfold cross-validation to test the parameters
    predict = True

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
    if predict:
        y_hold_pred = np.zeros([n_iter, X_hold.shape[0], 9])
        y_test_pred = np.zeros([n_iter, X_test.shape[0], 9])
    scores = []
    np.random.seed(0)
    for i, (train_index, valid_index) in enumerate(StratifiedKFold(y_data, n_iter, shuffle=True, random_state=0)):
        print('Fold {}'.format(i))
        X_train, X_valid = X_data[train_index], X_data[valid_index]
        y_train, y_valid = y_data[train_index], y_data[valid_index]
        deval = xgb.DMatrix(X_valid, label=y_valid) if predict else None
        clf = train(X_train, y_train, params, deval)
        y_valid_pred = clf.predict(xgb.DMatrix(X_valid, label=y_valid), ntree_limit=clf.best_iteration)
        scores.append(log_loss(y_valid, y_valid_pred))
        print(scores[-1])
        if predict:
            y_hold_pred[i,:,:] = clf.predict(xgb.DMatrix(X_hold), ntree_limit=clf.best_iteration)
            y_test_pred[i,:,:] = clf.predict(xgb.DMatrix(X_test), ntree_limit=clf.best_iteration)
    print(scores.mean(axis=1), scores.std(axis=1))
    if predict:
        print('Warning: validation scores are meaningless when predict=True')
        np.save('xgb_hold_pred', y_hold_pred)
        np.save('xgb_test_pred', y_test_pred)
      
from sklearn.preprocessing import LabelBinarizer      
      
def isotonic_fit(y_true, y_pred):
    yp = LabelBinarizer().fit_transform(y_true)
    w = np.random.random((1,y_pred.shape[0]))
    cost = ((np.tensordot(w, y_pred, axes=(0,0)) - yp)**2).sum()
    
def prediction_weights(y_preds, y_real):
    eps = 1.e-15
    yr = LabelBinarizer().fit_transform(y_real).clip(eps, 1-eps)

    def log_loss_func(weights):
        weights /= np.sum(weights)
        yp = sum([w*y for w,y in zip(weights, y_preds)])
        return log_loss(y_real, yp)
    
    bounds = [(0,1)]*len(y_preds)
    starting = [1.]*len(y_preds)
    res = minimize(log_loss_func, starting, method='L-BFGS-B', bounds=bounds)
    return res