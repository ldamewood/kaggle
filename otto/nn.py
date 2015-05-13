#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition
from kaggle.network import AdjustVariable, EarlyStopping, float32

import numpy as np
import theano

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import log_loss
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adagrad, nesterov_momentum
from nolearn.lasagne import NeuralNet

if __name__ == '__main__':
    encoder = LabelEncoder()
    
    # Identical to StandardScaler using all train and test data.
    scaler = StandardScaler()

    # Training data
    X, y = OttoCompetition.load_data(train=True)
    y = encoder.fit_transform(y).astype('int32')
    X = scaler.fit_transform(X).astype('float32')
    n_classes = np.unique(y).shape[0]
    n_features = X.shape[1]

    # Split a holdout set
    sss = StratifiedShuffleSplit(y, 1, test_size=0.2, random_state=0)
    data_idx, hold_idx = next(iter(sss))
    X_data, X_hold = X[data_idx], X[hold_idx]
    y_data, y_hold = y[data_idx], y[hold_idx]
    print(hash(','.join(map(str, y_hold))))

    # Test data
    X_test, _ = OttoCompetition.load_data(train=False)
    X_test = scaler.transform(X_test).astype('float32')


    np.random.seed(0)
    
    print("Training model...")
    net = NeuralNet(layers= [ ('input', InputLayer),
                              ('dense1', DenseLayer),
                              ('dropout1', DropoutLayer),
                              ('dense2', DenseLayer),
                              ('dropout2', DropoutLayer),
                              ('dense3', DenseLayer),
                              ('output', DenseLayer)],
             input_shape=(None, n_features),
             dense1_num_units=512,
             dropout1_p=0.5,
             dense2_num_units=512,
             dropout2_p=0.5,
             dense3_num_units=512,
             output_num_units=n_classes,
             output_nonlinearity=softmax,
             update=nesterov_momentum,
             eval_size=0.1,
             verbose=1,
             update_learning_rate=theano.shared(float32(0.001)),
             update_momentum=theano.shared(float32(0.9)),
             on_epoch_finished=[
                 AdjustVariable('update_learning_rate', start=0.001, stop=0.000001),
                 AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 EarlyStopping(patience=100),
             ],
             max_epochs=10000,)

    scores = []
    n_iter = 10
    y_hold_pred = np.zeros([n_iter, X_hold.shape[0], 9])
    y_test_pred = np.zeros([n_iter, X_test.shape[0], 9])
    for i, (train_index, valid_index) in enumerate(StratifiedKFold(y_data, n_iter, shuffle=True, random_state=0)):
        X_train, X_valid = X_data[train_index], X_data[valid_index]
        y_train, y_valid = y_data[train_index], y_data[valid_index]
        clf = clone(net).fit(X_train, y_train)
        y_valid_pred = clf.predict_proba(X_valid)
        y_hold_pred[i,:,:] = clf.predict_proba(X_hold)
        y_test_pred[i,:,:] = clf.predict_proba(X_test)
        scores.append(log_loss(y_valid, y_valid_pred))
    print('CV:',np.mean(scores), np.std(scores))
    np.save('nn_hold_pred', y_hold_pred)
    np.save('nn_test_pred', y_test_pred)