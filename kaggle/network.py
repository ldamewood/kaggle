# -*- coding: utf-8 -*-

import numpy as np

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

class AdaptiveVariable(object):
    def __init__(self, name, start=0.03, stop=0.000001, inc=1.1, dec=0.5):
        self.name = name
        self.start, self.stop = start, stop
        self.inc, self.dec = inc, dec
    
    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        if len(train_history) > 1:
            previous_valid = train_history[-2]['valid_loss']
        else:
            previous_valid = np.inf
        current_value = getattr(nn, self.name).get_value()
        if current_value < self.stop:
            raise StopIteration()
        if current_valid > previous_valid:
            getattr(nn, self.name).set_value(float32(current_value*self.dec))
        else:
            getattr(nn, self.name).set_value(float32(current_value*self.inc))

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