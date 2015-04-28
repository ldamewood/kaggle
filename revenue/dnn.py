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
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum, adagrad, adadelta
from nolearn.lasagne import NeuralNet

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
    train_size = 0.75
    cls = RandomForestClassifier()
    train_df_orig = RevenueCompetition.load_data()
    y = train_df_orig['revenue'].values.astype('float32')
    del train_df_orig['revenue']

    test_df_orig = RevenueCompetition.load_data(train=False)

    full_df_orig = train_df_orig.append(test_df_orig)
    
    print("Transforming...")
    tr = make_pipeline(RevenueTransform(rescale=True), StandardScaler())
    tr.fit(full_df_orig)

    print('Classify the outliers...')
    ly = np.log(y)
    ym = ly.mean()
    ys = ly.std()
    s = np.empty(ly.shape[0])
    s[(ly-ym)/ys <= -2] = 0
    s[np.logical_and((ly-ym)/ys > -2,(ly-ym)/ys <= -1)] = 1
    s[np.logical_and((ly-ym)/ys > -1,(ly-ym)/ys <= 1)] = 2
    s[np.logical_and((ly-ym)/ys > 1,(ly-ym)/ys <= 2)] = 3
    s[(ly-ym)/ys > 2] = 4

    print("Train outlier model...")
    X = tr.transform(train_df_orig)
    cls.fit(X,s)
    X = np.vstack([X.T, (s==4).T]).T.astype('float32')
      
    print("Training regression model...")
    net0 = NeuralNet(layers=[('input', InputLayer),
                             ('dense1', DenseLayer),
                             ('dropout1', DropoutLayer),
                             ('dense2', DenseLayer),
                             ('dropout2', DropoutLayer),
                             ('dense3', DenseLayer),
                             ('output', DenseLayer)],
                     input_shape=(None, X.shape[1]),
                     dense1_num_units=512,
                     dropout1_p=0.5,
                     dense2_num_units=512,
                     dropout2_p=0.5,
                     dense3_num_units=512,
                     regression=True,
                     output_num_units=1,
                     output_nonlinearity=None,
                     update=nesterov_momentum,
                     eval_size=None,
                     verbose=1,
                     update_learning_rate=theano.shared(float32(0.0001)),
                     # update_momentum=theano.shared(float32(0.9)),
                     on_epoch_finished=[
                        # AdjustVariable('update_learning_rate', start=0.01, stop=0.00001),
                        # AdjustVariable('update_momentum', start=0.9, stop=0.999),
                     ],
                     max_epochs=100,)
    ly.shape = (ly.shape[0],1)
    net0.fit(X, (ly-ym)/ys)
    
    
    print('Transform test set...')
    X = tr.transform(test_df_orig)
    s = cls.predict(X)
    X = np.vstack([X.T, (s==4).T]).T.astype('float32')
    
    yp1 = np.exp((net0.predict(X)*ys)+ym)
    yp2 = pd.read_csv('data/rf11_svr.csv', index_col='Id').values
# #    test_df['prob0'] = predstd == 0
# #    test_df['prob1'] = predstd == 1
# #    test_df['prob2'] = predstd == 2
# #    test_df['prob3'] = predstd == 3
#     test_df['prob4'] = predstd == 4
#     X = ss.transform(tr.transform(test_df).values)
#     yp1 = np.exp(net0.predict(X))
#     yp2 = pd.read_csv('data/rf11_svr.csv', index_col='Id').values
    
#
#    print('Predict test set...')
#    yp = net0.predict(X)
#    RevenueCompetition.save_data(yp, 'data/revenue_20150428_01.csv')