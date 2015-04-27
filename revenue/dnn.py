# -*- coding: utf-8 -*-

from revenue import RevenueCompetition, RevenueTransform

import pandas as pd
import numpy as np
import theano
import copy

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, adagrad, adadelta
from nolearn.lasagne import NeuralNet

layers0 = [('input', InputLayer),
           ('dense1', DenseLayer),
#           ('dropout1', DropoutLayer),
#           ('dense2', DenseLayer),
#           ('dropout2', DropoutLayer),
#           ('dense3', DenseLayer),
#           ('dropout3', DropoutLayer),
#           ('dense4', DenseLayer),
           ('output', DenseLayer)]

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

if __name__ == '__main__':
    train_df_orig = RevenueCompetition.load_data()
    y = train_df_orig['revenue'].values
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
    del train_df_orig['revenue']

    test_df_orig = RevenueCompetition.load_data(train=False)

    full_df = train_df_orig.append(test_df_orig)
    
    print("Transforming...")
    tr = RevenueTransform(rescale=True)
    tr.fit(full_df)

    print('Searching for outliers...')                               
    rfc = GradientBoostingClassifier(n_estimators=10, random_state=0)
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
    
    np.random.seed(0)
    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, X.shape[1]),
                     dense1_num_units=512,
#                     dropout1_p=0.5,
#                     dense2_num_units=512,
#                     dropout2_p=0.5,
#                     dense3_num_units=512,
#                     dropout3_p=0.5,
#                     dense4_num_units=1600,
                     regression=True,
                     output_num_units=1,
                     output_nonlinearity=None,
                     update=nesterov_momentum,
                     eval_size=0.1,
                     verbose=1,
                     update_learning_rate=theano.shared(float32(0.000001)),
                     update_momentum=theano.shared(float32(0.9)),
                     on_epoch_finished=[
#                         AdjustVariable('update_learning_rate', start=0.01, stop=0.00001),
#                         AdjustVariable('update_momentum', start=0.9, stop=0.999),
                     ],
                     max_epochs=10,)
    
    y.shape = (y.shape[0],1)
    y = y.astype('float32')
    X = X.astype('float32')
    net0.fit(X,y)
    
#    print('Training')
#    data = X
#    model = RnnRbm()
#    model.train(Xl, batch_size=1, num_epochs=200)
#    print(model.generate())

