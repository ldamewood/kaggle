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
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit, train_test_split, StratifiedKFold
from sklearn.metrics import log_loss

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax, sigmoid, rectify
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

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(y, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def float32(k):
    return np.cast['float32'](k)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

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
    num_classes = len(encoder.classes_)
    num_features = X.shape[1]
#    
    np.random.seed(0)
    
    print("Training model...")

    nets, ll = [], []
    for i, (train_index, valid_index) in enumerate(StratifiedKFold(y, n_folds = 10, random_state=0)):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        net0 = NeuralNet(layers= [ ('input', InputLayer),
                                   ('dense1', DenseLayer),
                                   ('dropout1', DropoutLayer),
                                   ('dense2', DenseLayer),
                                   ('dropout2', DropoutLayer),
                                   ('dense3', DenseLayer),
#                                   ('dropout3', DropoutLayer),
#                                   ('dense4', DenseLayer),
                                   ('output', DenseLayer)],
                 input_shape=(None, num_features),
                 dense1_num_units=512,
                 dropout1_p=0.5,
                 dense2_num_units=512,
                 dropout2_p=0.5,
                 dense3_num_units=512,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=nesterov_momentum,
                 eval_size=0.05,
                 verbose=1,
                 update_learning_rate=theano.shared(float32(0.001)),
                 update_momentum=theano.shared(float32(0.9)),
                 on_epoch_finished=[
#                         AdjustVariable('update_learning_rate', start=0.001, stop=0.0001),
#                         AdjustVariable('update_momentum', start=0.9, stop=0.999),
                     EarlyStopping(),
                 ],
                 max_epochs=10000,)
        nets.append(net0.fit(X_train, y_train))
        y_pred = nets[-1].predict_proba(X_valid)
        ll.append(log_loss(y_valid, y_pred))
    
    
    X_test, ids = load_test_data('data/test.csv', scaler)
    y_preds = []
    for net in nets:
        y_preds.append(net.predict_proba(X_test))
    y_pred = sum(y_preds)/10.
    
#    OttoCompetition.save_data(y_pred, 'data/otto_submit_20150424_1.csv')