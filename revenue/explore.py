# -*- coding: utf-8 -*-

from scipy.stats import uniform as sp_uniform
from scipy.stats import randint as sp_randint
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

from kaggle.util import RemoveVarianceless
from revenue import RevenueCompetition, RevenueTransform

params1 = dict(
    rf__n_estimators=sp_randint(10,1000),
    rf__max_features=['auto','sqrt','log2'],
    rf__random_state=[0],
    rf__max_depth=sp_randint(3,20),
    rf__oob_score=[True],
#    gb__learning_rate=[1,0.1,0.01,0.001],
#    gb__n_estimators=sp_randint(10,1000),
#    gb__max_depth=sp_randint(3,20),
#    gb__subsample=sp_uniform(0.25,0.75),
#    gb__max_features=['auto','sqrt','log2'],
#    gb__random_state=[0],
)

params2 = dict(
    nn__n_components=sp_randint(10,100),
    nn__learning_rate=[1.,0.1,0.01,0.001],
    nn__n_iter=sp_randint(1,20),
    nn__batch_size=sp_randint(1,1000),
    nn__random_state=[0],
    sg__loss=['squared_loss','huber'],
    sg__penalty=['elasticnet'],
    sg__alpha=[1.,0.1,0.01,0.001],
    sg__l1_ratio=sp_uniform(0,1),
    sg__n_iter=sp_randint(1,20),
    sg__shuffle=[True],
    sg__random_state=[0],
)

if __name__ == '__main__':
    train_df = RevenueCompetition.load_data()
    y = train_df['revenue']
    del train_df['revenue']

    test_df = RevenueCompetition.load_data(train=False)

    full_df = train_df.append(test_df)
    
    print("Fitting 1...")
    tr1 = RevenueTransform(rescale=False)
    tr1.fit(full_df)    
    X1 = tr1.transform(train_df).values
    cv1 = RandomizedSearchCV(Pipeline([('rf',RandomForestRegressor()),]),
                            params1, n_iter=50, verbose=True, cv=10, n_jobs=4)
    reg1 = BaggingRegressor(cv1, n_estimators=100, verbose=True, random_state=0,
                           oob_score=True)
    reg1.fit(X1, y)
    y_pred1 = reg1.predict(X1)
#    y_pred1 = reg1.predict(tr1.transform(test_df).values)
    
    print("Fitting 2...")
    tr2 = make_pipeline(RevenueTransform(rescale=True),MinMaxScaler())
    tr2.fit(full_df)    
    X2 = tr2.transform(train_df)
    cv2 = RandomizedSearchCV(Pipeline([('nn',BernoulliRBM()),('sg',SGDRegressor())]),
                            params2, n_iter=50, verbose=True, cv=10, n_jobs=4)
    reg2 = BaggingRegressor(cv2, n_estimators=100, verbose=True, random_state=0,
                           oob_score=True)
    reg2.fit(X2, y)
    y_pred2 = reg2.predict(X2)
    
    yp = np.vstack([y_pred1,y_pred2])
    A = np.linalg.inv((yp.T.dot(yp))).dot(np.dot(yp.T,y))
    
    y_pred1 = reg1.predict(tr1.transform(test_df).values)
    y_pred2 = reg2.predict(tr2.transform(test_df).values)
    yp = np.vstack([y_pred1,y_pred2]).T
    y_pred = yp.dot(A)
