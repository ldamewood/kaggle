#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rain import RainCompetition

import pandas as pd
import numpy as np
import theano

from kaggle.network import AdjustVariable, EarlyStopping, float32

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold, train_test_split, StratifiedShuffleSplit
from sklearn.metrics import log_loss

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

def load():
    train_df = pd.read_csv('data/train_normalize.csv', index_col=['Id', 'Group'])
    y = train_df['Expected'].values.clip(0,71).astype('int32')
    del train_df['Expected']
    X = train_df.values.astype('float32')
    ids = train_df.index.values
    return X, y, ids

if __name__ == '__main__':
    print('loading...')
    X, y, ids = load()
    np.random.seed(0)
    num_features = X.shape[1]
    num_classes = 72
    
    print("Training model...")
    nets, ll = [], []
    
    train_index, valid_index = tuple(*list(StratifiedShuffleSplit(y, n_iter=1, test_size=0.2, random_state = 0)))
    
#    for i, (train_index, valid_index) in enumerate(StratifiedKFold(y, n_folds = 5, random_state=0)):
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
             eval_size=0.2,
             verbose=1,
             update_learning_rate=theano.shared(float32(0.001)),
             update_momentum=theano.shared(float32(0.9)),
             on_epoch_finished=[
                     AdjustVariable('update_learning_rate', start=0.001, stop=0.0001),
                     AdjustVariable('update_momentum', start=0.9, stop=0.999),
                     EarlyStopping(),
             ],
             max_epochs=10000,)
    net0.fit(X_train, y_train)
    
    
#    X_test, ids = load_test_data('data/test.csv', scaler)
#    y_preds = []
#    for net in nets:
#        y_preds.append(net.predict_proba(X_test))
#    y_pred = sum(y_preds)/10.
    
#    OttoCompetition.save_data(y_pred, 'data/otto_submit_20150424_1.csv')
