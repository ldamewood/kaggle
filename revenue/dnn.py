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
from sklearn.ensemble import BaggingClassifier
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
           ('dense2', DenseLayer),
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
    
    train_df = RevenueCompetition.load_data()
    y = train_df['revenue'].values
    del train_df['revenue']

    test_df = RevenueCompetition.load_data(train=False)

    full_df = train_df.append(test_df)
    
    X = full_df[[c for c in full_df.columns if c[0] == 'P']].values
    
    # Encode
    tr = make_pipeline(RevenueTransform(), StandardScaler())
    print("Encoding...")
    tr.fit(full_df)

    X = tr.transform(train_df)
    
    np.random.seed(0)
    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, X.shape[1]),
                     dense1_num_units=400,
#                     dropout1_p=0.5,
                     dense2_num_units=400,
#                     dropout2_p=0.2,
#                     dense3_num_units=400,
#                     dropout3_p=0.5,
#                     dense4_num_units=1600,
                     regression=True,
                     output_num_units=1,
                     output_nonlinearity=None,
                     update=adadelta,
                     eval_size=0.5,
                     verbose=1,
                     update_learning_rate=theano.shared(float32(0.0001)),
#                     update_momentum=theano.shared(float32(0.9)),
                     on_epoch_finished=[
                         AdjustVariable('update_learning_rate', start=0.0001, stop=0.00001),
#                         AdjustVariable('update_momentum', start=0.9, stop=0.999),
                     ],
                     max_epochs=3000,)
    
    y.shape = (y.shape[0],1)
    y = y.astype('float32')
    net0.fit(X,y)
    
#    print('Training')
#    data = X
#    model = RnnRbm()
#    model.train(Xl, batch_size=1, num_epochs=200)
#    print(model.generate())

