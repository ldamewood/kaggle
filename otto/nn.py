#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition, OttoTransform

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
from lasagne.updates import nesterov_momentum, adagrad
from nolearn.lasagne import NeuralNet

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids

layers0 = [('input', InputLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
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
    X, y, encoder, scaler = load_train_data('data/train.csv')
    X_test, ids = load_test_data('data/test.csv', scaler)
    num_classes = len(encoder.classes_)
    num_features = X.shape[1]
    
    params = dict(
        dense1_num_units=[100],
        dense2_num_units=[100],
    )    
    
    np.random.seed(0)
    net0 = NeuralNet(layers=layers0,
                     input_shape=(None, num_features),
                     dense1_num_units=400,
                     dropout1_p=0.1,
                     dense2_num_units=400,
#                     dropout2_p=0.2,
#                     dense3_num_units=400,
#                     dropout3_p=0.4,
#                     dense4_num_units=800,
                     output_num_units=num_classes,
                     output_nonlinearity=softmax,
                     update=adagrad,
                     eval_size=0.2,
                     verbose=1,
                     update_learning_rate=theano.shared(float32(0.001)),
#                     update_momentum=theano.shared(float32(0.9)),
                     on_epoch_finished=[
                         AdjustVariable('update_learning_rate', start=0.001, stop=0.0001),
#                         AdjustVariable('update_momentum', start=0.9, stop=0.999),
                     ],
                     max_epochs=1000,)
    
    net0.fit(X,y)