#!/usr/bin/env python
# -*- coding: utf-8 -*-

from otto import OttoCompetition, OttoScaler
from kaggle.network import AdjustVariable, EarlyStopping, float32, AdaptiveVariable

import numpy as np
import theano

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

if __name__ == '__main__':
    encoder = LabelEncoder()
    n_folds = 10       
    
    # Identical to StandardScaler using all train and test data.
    scaler = OttoScaler()

    # Training data
    X, y = OttoCompetition.load_data(train=True)
    y = encoder.fit_transform(y).astype('int32')
    X = scaler.transform(X).astype('float32')
    n_classes = np.unique(y).shape[0]
    n_features = X.shape[1]

    # Test data
    X_test, _ = OttoCompetition.load_data(train=False)
    X_test = scaler.transform(X_test).astype('float32')


    np.random.seed(0)
    
    print("Training model...")

    bs = 2**10
    y_preds = np.zeros([n_folds, X_test.shape[0], n_classes])
    scores = []
    for i, (train_index, valid_index) in enumerate(StratifiedKFold(y, n_folds = n_folds, random_state=1)):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        clf = NeuralNet(layers= [ ('input', InputLayer),
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
                     AdjustVariable('update_learning_rate', start=0.001, stop=0.00001),
                     AdjustVariable('update_momentum', start=0.9, stop=0.999),
                     EarlyStopping(),
                 ],
                 max_epochs=10000,)
        clf.fit(X_train, y_train)
        scores.append(log_loss(y_valid, clf.predict_proba(X_valid)))

        for j in xrange((X_test.shape[0] + bs + 1) // bs):
            s = slice(bs * j, bs * (j+1))
            y_preds[i, s,:] = clf.predict_proba(X_test[s, :])

#    X_test, _ = OttoCompetition.load_data(train=False)
#    X_test = scaler.fit_transform(X_test).astype('float32')
#    y_preds = []
#    for net in nets:
#        y_preds.append(net.predict_proba(X_test))
#    y_pred = sum(y_preds)/10.
#    
#    OttoCompetition.save_data(y_pred)