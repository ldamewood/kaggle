#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rain import RainCompetition

import pandas as pd
import numpy as np
import theano
from time import time

from kaggle.network import AdjustVariable, EarlyStopping, float32

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

class ansi:
    BLUE = '\033[94m'
    GREEN = '\033[32m'
    ENDC = '\033[0m'

def do_fit(self, filename):
    on_epoch_finished = self.on_epoch_finished
    if not isinstance(on_epoch_finished, (list, tuple)):
        on_epoch_finished = [on_epoch_finished]

    on_training_finished = self.on_training_finished
    if not isinstance(on_training_finished, (list, tuple)):
        on_training_finished = [on_training_finished]

    epoch = 0
    info = None
    best_valid_loss = np.inf
    best_train_loss = np.inf

    if self.verbose:
        print("""
 Epoch  |  Train loss  |  Valid loss  |  Train / Val  |  Valid acc  |  Dur
--------|--------------|--------------|---------------|-------------|-------\
""")

    while epoch < self.max_epochs:
        it = pd.read_csv(filename, index_col=['Id', 'Index'], iterator=True,
                         chunksize=2**15)

        epoch += 1

        train_losses = []
        valid_losses = []
        valid_accuracies = []

        t0 = time()

        for chunk in it:
            y = chunk['Expected'].values.clip(0,70).astype('int32')
            X = chunk[[c for c in chunk.columns if c not in ['Expected']]].values.astype('float32')
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=self.eval_size)
    
            for Xb, yb in self.batch_iterator_train(X_train, y_train):
                batch_train_loss = self.train_iter_(Xb, yb)
                train_losses.append(batch_train_loss)
    
            for Xb, yb in self.batch_iterator_test(X_valid, y_valid):
                batch_valid_loss, accuracy = self.eval_iter_(Xb, yb)
                valid_losses.append(batch_valid_loss)
                valid_accuracies.append(accuracy)

        avg_train_loss = np.mean(train_losses)
        avg_valid_loss = np.mean(valid_losses)
        avg_valid_accuracy = np.mean(valid_accuracies)

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss

        if self.verbose:
            best_train = best_train_loss == avg_train_loss
            best_valid = best_valid_loss == avg_valid_loss
            print(" {:>5}  |  {}{:>10.6f}{}  |  {}{:>10.6f}{}  "
                  "|  {:>11.6f}  |  {:>9}  |  {:>3.1f}s".format(
                      epoch,
                      ansi.BLUE if best_train else "",
                      avg_train_loss,
                      ansi.ENDC if best_train else "",
                      ansi.GREEN if best_valid else "",
                      avg_valid_loss,
                      ansi.ENDC if best_valid else "",
                      avg_train_loss / avg_valid_loss,
                      "{:.2f}%".format(avg_valid_accuracy * 100)
                      if not self.regression else "",
                      time() - t0,
                      ))

        info = dict(
            epoch=epoch,
            train_loss=avg_train_loss,
            valid_loss=avg_valid_loss,
            valid_accuracy=avg_valid_accuracy,
            )
        self.train_history_.append(info)
        try:
            for func in on_epoch_finished:
                func(self, self.train_history_)
        except StopIteration:
            break

    for func in on_training_finished:
        func(self, self.train_history_)

if __name__ == '__main__':
    np.random.seed(0)
    num_features = 43
    num_classes = 71
    
    net0 = NeuralNet(layers= [ ('input', InputLayer),
                               ('dense1', DenseLayer),
#                               ('dropout1', DropoutLayer),
#                               ('dense2', DenseLayer),
#                               ('dropout2', DropoutLayer),
#                               ('dense3', DenseLayer),
                               ('output', DenseLayer)],
             input_shape=(None, num_features),
             dense1_num_units=512,
#             dropout1_p=0.5,
#             dense2_num_units=512,
#             dropout2_p=0.5,
#             dense3_num_units=512,
             output_num_units=num_classes,
             output_nonlinearity=softmax,
             update=nesterov_momentum,
             eval_size=0.2,
             verbose=1,
             update_learning_rate=theano.shared(float32(0.0001)),
             update_momentum=theano.shared(float32(0.9)),
             on_epoch_finished=[
                     AdjustVariable('update_learning_rate', start=0.0001, stop=0.00001),
                     AdjustVariable('update_momentum', start=0.9, stop=0.999),
                     EarlyStopping(),
             ],
             max_epochs=10000,)
    net0.initialize()
    do_fit(net0, 'data/train_impu_norm_shuf.csv')