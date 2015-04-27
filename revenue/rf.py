# -*- coding: utf-8 -*-

import numpy as np
import progressbar

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import StratifiedShuffleSplit

from revenue import RevenueCompetition, RevenueTransform
from xgboost import XGBClassifier

if __name__ == '__main__':
        
    train_size = 0.75
    cls = XGBClassifier(max_depth=10, learning_rate=0.05, n_estimators=100,
                        silent=False)
    reg = RandomForestRegressor(n_estimators=20, max_features=5,
                                 random_state=0, max_depth=4,
                                 min_samples_split=2, min_samples_leaf=1,
                                 max_leaf_nodes=None, bootstrap=True,
                                 oob_score=False, n_jobs=-1)
    train_df_orig = RevenueCompetition.load_data()
    y = train_df_orig['revenue'].values
    del train_df_orig['revenue']

    test_df_orig = RevenueCompetition.load_data(train=False)

    full_df = train_df_orig.append(test_df_orig)
    
    print("Transforming...")
    tr = RevenueTransform(rescale=True)
    tr.fit(full_df)
    X = tr.transform(train_df_orig).values

    print('Classify the outliers...')
    ly = np.log(y)
    ym = ly.mean()
    ys = ly.std()
    s = np.empty(ly.shape[0])
    s[(ly-ym)/ys <= -2] = 0
    s[np.logical_and((ly-ym)/ys > -2,(ly-ym)/ys <= -1)] = -1
    s[np.logical_and((ly-ym)/ys > -1,(ly-ym)/ys <= 1)] = 2
    s[np.logical_and((ly-ym)/ys > 1,(ly-ym)/ys <= 2)] = 3
    s[(ly-ym)/ys > 2] = 4

    train_index, valid_index = list(StratifiedShuffleSplit(s, n_iter=1, train_size=train_size, random_state=0))[0]
    X_train, X_valid = X[train_index], X[valid_index]
    y_train, y_valid = y[train_index], y[valid_index]
    s_train, s_valid = s[train_index], s[valid_index]

    assert (np.unique(s_valid) == np.unique(s_train)).all()

    print("Train outlier model...")
    cls.fit(X_train,s_train)
#
    print("Appending outlier predictions...")
    train_df = train_df_orig.copy()
    predstd = cls.predict(tr.transform(train_df_orig).values)
#    train_df['prob0'] = predstd == 0
#    train_df['prob1'] = predstd == 1
#    train_df['prob2'] = predstd == 2
#    train_df['prob3'] = predstd == 3
    train_df['prob4'] = predstd == 4
    X = tr.transform(train_df).values
      
    print("Training regression model...")
    reg.fit(X_train, y_train)
    
    print("Validating regression model...")
    p = progressbar.ProgressBar(maxval=len(y_valid)).start()
    mse = []
    for i in range(len(y_valid)):
        X_sample = X_valid[i]
        y_sample = y_valid[i]
        y_pred = reg.predict([X_sample])
        mse.append(np.sqrt(np.mean((y_pred - y_sample)**2)))
        p.update(i)
    p.update(len(y_valid))
    print('')
    print("Regression mse:")
    print(np.mean(mse), np.std(mse)/np.sqrt(len(y_valid)))

    print('Fit with all data')    
    reg.fit(X,y)
    imporder = reg.feature_importances_.argsort()[::-1]
    for c,v in zip(tr.transform(train_df).columns.values[imporder], reg.feature_importances_[imporder]):
        print('{} : \t {}'.format(c,v))
    
    print('Transform test set...')
    test_df = test_df_orig.copy()
    predstd = cls.predict(tr.transform(test_df_orig).values)
#    test_df['prob0'] = predstd == 0
#    test_df['prob1'] = predstd == 1
#    test_df['prob2'] = predstd == 2
#    test_df['prob3'] = predstd == 3
    test_df['prob4'] = predstd == 4
    X = tr.transform(test_df).values

    print('Predict test set...')
    yp = reg.predict(X)
    RevenueCompetition.save_data(yp, 'data/submit_20150426_01.csv')