# -*- coding: utf-8 -*-

from otto import OttoCompetition, OttoScaler
import numpy as np
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone

np.random.seed(0)

def _fit(estimator, X, y, train, valid):
    X_train, y_train = X[train], y[train]
    X_valid, y_valid = X[valid], y[valid]
    estimator.fit(X_train, y_train)
    score = log_loss(y_valid, estimator.predict_proba(X_valid))
    return estimator, score

"""
C = 10, score = 0.74926798298890462
C = 100, score = 
"""

if __name__ == '__main__':
    params = dict(
        C=100.,
        probability=True,
        random_state=0,
    )
    
    # Identical to StandardScaler using all train and test data.
    scaler = OttoScaler()

    # Training data
    X, y = OttoCompetition.load_data(train=True)
    X = scaler.transform(X).astype('float')
    n_classes = np.unique(y).shape[0]
    
    # Split a holdout set
    data_idx, hold_idx = next(iter(StratifiedShuffleSplit(y, 1, test_size = 0.1, random_state=0)))
    X_data, X_hold = X[data_idx], X[hold_idx]
    y_data, y_hold = y[data_idx], y[hold_idx]

    # Test data
    X_test, _ = OttoCompetition.load_data(train=False)
    X_test = scaler.transform(X_test).astype('float')

    # CV
    clf = SVC(**params)
    skf = StratifiedKFold(y_data, n_folds=8, random_state=0)
    parallel = Parallel(n_jobs=8, verbose=True, pre_dispatch='2*n_jobs')
    blocks = parallel(delayed(_fit)(clone(clf), X_data, y_data, train, valid) for train, valid in skf)

    clfs = [c for c, _ in blocks]
    scores = np.array([s for _, s in blocks])
    print(scores)
    
    # Do predictions on test set
    bs = 2**10
    y_preds = np.zeros([len(clfs), X_hold.shape[0], n_classes])
    for i, clf in enumerate(clfs):
        for j in xrange((X_hold.shape[0] + bs + 1) // bs):
            s = slice(bs * j, bs * (j+1))
            y_preds[i, s,:] = clf.predict_proba(X_hold[s, :])

    # Average the softmax outcomes from n_estimators
    # y_preds => [n_estimators, n_rows, n_classes]
    eps = 1.e-15
    y_preds_all = np.array([[[0.,1.,0.]],[[1./3,1./3,1./3]],[[1./2,0.,1./2]]])
    y_preds_avg = np.exp(np.log(y_preds_all.clip(eps, 1-eps)).mean(axis=0))
    row_sums = y_preds_avg.sum(axis=1)
    y_preds_avg = y_preds_avg / row_sums[:, np.newaxis]
    
#    for i, (train_index, valid_index) in enumerate(skf):
#        print('Fold {}'.format(i))
#        X_train, X_valid = X_data[train_index], X_data[valid_index]
#        y_train, y_valid = y_data[train_index], y_data[valid_index]
#        
#        losses = []
#        minloss = np.nan
#        iminloss = 0
#        for epoch in xrange(10000):
#            clf.fit(X_train, y_train)
#            y_pred = np.zeros([X_valid.shape[0], num_classes])
#            for j in xrange(X_valid.shape[0] // bs):
#                s = slice(bs * j, bs * (j+1))
#                y_pred[s,:] = clf.predict_proba(X_valid[s, :])
#            losses.append(log_loss(y_valid, y_pred))
#            if losses[-1] < minloss:
#                minloss = losses[-1]
#                iminloss = len(losses)
#            if len(losses) > iminloss + 100:
#                break
#            print(losses[-1])
#        
#        y_pred = np.zeros([X_hold.shape[0], num_classes])
#        for j in xrange(X_hold.shape[0] // bs):
#            s = slice(bs * j, bs * (j+1))
#            y_pred[s,:] = clf.predict_proba(X_hold[s,:])
#        pd.DataFrame(y_pred).to_csv('data/holdout_{}_{}.csv'.format(basename(__file__), i), index=False)
#    
#        y_pred = np.zeros([X_test.shape[0], num_classes])
#        for j in xrange(X_test.shape[0] // bs):
#            s = slice(bs * j, bs * (j+1))
#            y_pred[s,:] = clf.predict_proba(X_test[s,:])
#        pd.DataFrame(y_pred).to_csv('data/test_{}_{}.csv'.format(basename(__file__), i), index=False)