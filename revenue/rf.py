# -*- coding: utf-8 -*-

from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint
import numpy as np

import copy
import itertools
import progressbar

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.grid_search import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import LeaveOneOut, ShuffleSplit, StratifiedKFold, train_test_split
from sklearn.metrics import log_loss

from kaggle.util import RemoveVarianceless
from revenue import RevenueCompetition, RevenueTransform

params_svr = dict(
    C=[1.,10.,100.],
    epsilon=[0.1,0.2],
    kernel=['linear','rbf','sigmoid'],
    random_state=[0],
)

params_rfr = dict(
    n_estimators=[10,100,1000],
    max_features=['auto'],
    max_depth=[2,5,10,None],
    min_samples_split=[2,4],
    min_samples_leaf=[1,2,4],
    random_state=[0],
)

if __name__ == '__main__':
    train_df_orig = RevenueCompetition.load_data()
    y = train_df_orig['revenue'].values
    del train_df_orig['revenue']

    test_df_orig = RevenueCompetition.load_data(train=False)

    full_df = train_df_orig.append(test_df_orig)
    
    print("Transforming...")
    tr = RevenueTransform(rescale=True)
    tr.fit(full_df)

    print('Searching for outliers...')
    ly = np.log(y)
    ym = ly.mean()
    ys = ly.std()
    sets = np.empty(ly.shape[0])
    sets[np.logical_and((ly-ym)/ys > -3,(ly-ym)/ys <= -2)] = 0
    sets[np.logical_and((ly-ym)/ys > -2,(ly-ym)/ys <= -1)] = 1
    sets[np.logical_and((ly-ym)/ys > -1,(ly-ym)/ys <= 1)] = 2
    sets[np.logical_and((ly-ym)/ys > 1,(ly-ym)/ys <= 2)] = 3
    sets[np.logical_and((ly-ym)/ys > 2,(ly-ym)/ys <= 3)] = 4
    sets[(ly-ym)/ys > 3] = 5                         
    rfc = GradientBoostingClassifier(n_estimators=50, random_state=0)
    rfc.fit(tr.transform(train_df_orig).values,sets)

    train_df = train_df_orig.copy()
    test_df = test_df_orig.copy()
    train_df['prob0'] = rfc.predict_proba(tr.transform(train_df_orig).values)[:,0]
    train_df['prob1'] = rfc.predict_proba(tr.transform(train_df_orig).values)[:,1]
    train_df['prob2'] = rfc.predict_proba(tr.transform(train_df_orig).values)[:,2]
    train_df['prob3'] = rfc.predict_proba(tr.transform(train_df_orig).values)[:,3]
    train_df['prob4'] = rfc.predict_proba(tr.transform(train_df_orig).values)[:,4]
    train_df['prob5'] = rfc.predict_proba(tr.transform(train_df_orig).values)[:,5]
    test_df['prob0'] = rfc.predict_proba(tr.transform(test_df_orig).values)[:,0]
    test_df['prob1'] = rfc.predict_proba(tr.transform(test_df_orig).values)[:,1]
    test_df['prob2'] = rfc.predict_proba(tr.transform(test_df_orig).values)[:,2]
    test_df['prob3'] = rfc.predict_proba(tr.transform(test_df_orig).values)[:,3]
    test_df['prob4'] = rfc.predict_proba(tr.transform(test_df_orig).values)[:,4]
    test_df['prob5'] = rfc.predict_proba(tr.transform(test_df_orig).values)[:,5]
    
    X = tr.transform(train_df).values
    Xt = tr.transform(test_df).values
    
    print("Fitting...")
    rfr = RandomForestRegressor(n_estimators=1000, max_features='auto',
                                random_state=0, max_depth=None, 
                                min_samples_split=2, min_samples_leaf=1,
                                max_leaf_nodes=None,
                                oob_score=False, n_jobs=-1)
                                
    svr = SVR()

    gs = GridSearchCV(rfr, params_rfr, scoring='mean_squared_error', iid=True, cv = 10,
                      refit=True, verbose=True, n_jobs=-1)

#    p = progressbar.ProgressBar(maxval=len(y)).start()
#    rfs, mse = [], []
#    for i,(train_index, test_index) in enumerate(LeaveOneOut(len(y))):
#        X_train, X_test = X[train_index], X[test_index]
#        y_train, y_test = y[train_index], y[test_index]
#        rfs.append(copy.deepcopy(rfr).fit(X_train, y_train))
#        y_pred = rfs[-1].predict(X_test)
#        mse.append(np.sqrt(np.mean((y_pred - y_test)**2)))
#        p.update(i)
##        print(mse[-1])
#    print(np.mean(mse))
    
    Xv, Xf, yv, yf = train_test_split(X, y, test_size=0.15, random_state=2)
    
    iters = 50
    p = progressbar.ProgressBar(maxval=iters).start()
    rfs, mse = [], []
    for i,(train_index, test_index) in enumerate(ShuffleSplit(len(yv), n_iter=iters, train_size=0.5, test_size=0.5, random_state=0)):
        X_train, X_test = Xv[train_index], Xv[test_index]
        y_train, y_test = yv[train_index], yv[test_index]
        rfs.append(copy.deepcopy(gs).fit(X_train, y_train))
        y_pred = rfs[-1].predict(X_test)
        mse.append(np.sqrt(np.mean((y_pred - y_test)**2)))
        p.update(i)
    p.update(iters)
#        print(mse[-1])
    print()
    print(np.mean(mse))
    
    yss = np.array([rfs[i].predict(Xf) for i in range(iters)])
    A = 1. - np.array(mse)/np.array(mse).sum()
    yy = np.average(yss,axis=0, weights=A)
    print()
    print(np.sqrt(np.mean((yy - yf)**2)))
    
    
#    y_pred = np.zeros(X.shape[0])
#    for i, irf in enumerate(rfs):
#        y_pred += irf.predict(X)
#    y_pred /= (i+1.)
    
#    y_pred = np.zeros(Xt.shape[0])
#    for i, irf in enumerate(rfs):
#        y_pred += irf.predict(Xt)
#    y_pred /= (i+1.)