#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition, OttoScaler
from kaggle.network import AdjustVariable, EarlyStopping, float32, AdaptiveVariable

import numpy as np
import theano

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
from sklearn.metrics import log_loss
from sklearn.base import clone

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

if __name__ == '__main__':
    encoder = LabelEncoder()
    
    # Identical to StandardScaler using all train and test data.
    scaler = OttoScaler()

    # Training data
    X, y = OttoCompetition.load_data(train=True)
    y = encoder.fit_transform(y).astype('int32')
    X = scaler.transform(X).astype('float32')
    n_classes = np.unique(y).shape[0]
    n_features = X.shape[1]

    # Split a holdout set
    data_idx, hold_idx = next(iter(StratifiedShuffleSplit(y, 1, test_size = 0.2, random_state=0)))
    X_data, X_hold = X[data_idx], X[hold_idx]
    y_data, y_hold = y[data_idx], y[hold_idx]

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
             eval_size=0.2,
             verbose=1,
             update_learning_rate=theano.shared(float32(0.001)),
             update_momentum=theano.shared(float32(0.9)),
             on_epoch_finished=[
                 AdjustVariable('update_learning_rate', start=0.001, stop=0.000001),
                 AdjustVariable('update_momentum', start=0.9, stop=0.999),
                 EarlyStopping(),
             ],
             max_epochs=10000,)

    bs = 2**10
    n_iter = 1
    y_preds = np.zeros([n_iter, X_hold.shape[0], n_classes])
    scores, clfs = [], []
#    for i, (train_index, valid_index) in enumerate(StratifiedKFold(y_data, n_iter, random_state=1, shuffle=True)):
    for i, (train_index, valid_index) in enumerate(StratifiedShuffleSplit(y_data, n_iter, test_size=0.2, random_state=0)):
        print('Iter {}'.format(i))
        X_train, X_valid = X_data[train_index, :], X_data[valid_index, :]
        y_train, y_valid = y_data[train_index], y_data[valid_index]

        clfs.append(clone(net).fit(X_train, y_train))
        scores.append(log_loss(y_valid, clfs[-1].predict_proba(X_valid)))

#        for j in xrange((X_hold.shape[0] + bs + 1) // bs):
#            s = slice(bs * j, bs * (j+1))
#            y_preds[i, s,:] = clfs[-1].predict_proba(X_hold[s, :])

    print(np.mean(scores))
#    A = np.exp(np.log(y_preds).mean(axis=0))
#    row_sums = A.sum(axis=1)
#    A = A / row_sums[:, np.newaxis]
#    print(scores)
#    print('softmax avg', log_loss(y_hold, A))
#    
#    B = y_preds.mean(axis=0)
#    print('just avg', log_loss(y_hold, B))

#    X_test, _ = OttoCompetition.load_data(train=False)
#    X_test = scaler.fit_transform(X_test).astype('float32')
#    y_preds = []
#    for net in nets:
#        y_preds.append(net.predict_proba(X_test))
#    y_pred = sum(y_preds)/10.
#    
#    OttoCompetition.save_data(y_pred)