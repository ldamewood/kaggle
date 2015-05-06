# -*- coding: utf-8 -*-
from __future__ import print_function, division
import pandas as pd
import numpy as np
from rain import RainCompetition
from sklearn.cross_validation import StratifiedKFold
import progressbar

def do_fold_h5():
    # Requires a lot of memory!
    print('Loading')
    store = pd.HDFStore('data/train_folds.h5', mode='w')
    df = pd.read_hdf(RainCompetition.__data__['train_missing'], 'df')
    y = df['Expected'].clip(0,70).astype('int')
    y_grp = y.groupby(level=0).mean()
    skf = StratifiedKFold(y_grp.values.ravel(), 10, shuffle=True, random_state=0)
    for i, (train_id, valid_id) in enumerate(skf):
        print('Fold {}'.format(i))
        order = y.index.levels[0][np.hstack([train_id,valid_id])]
        split = y.index.levels[0][train_id].shape[0] - 1
        newindex = y.index[np.argsort(order[y.index.labels[0]],kind='mergesort')]
        df.reindex(newindex)
        store.append('train_fold_{}'.format(i), df.loc[:split-1], data_columns=True)
        store.append('valid_fold_{}'.format(i), df.loc[split:], data_columns=True)

def do_fold_libsvm(bs = 2**15):
    # Low memory
    print('Loading')
    read = pd.HDFStore('data/train_folds.h5', mode='r')
    for i in xrange(10):
        print('Fold {}'.format(i))
        
        print('Training set')
        name = 'train_fold_{}'.format(i)
        nrows = read.get_storer(name).nrows
        pb = progressbar.ProgressBar(maxval=nrows//bs).start()
        with open('train_folds_{}.txt'.format(i), 'wb') as out:
            for j,chunk in enumerate(read.select(name, iterator=True, chunksize=bs)):
                pb.update(j)
                X = chunk.drop('Expected', axis=1).values.astype('float')
                y = chunk['Expected'].values.clip(0,70).astype('int')
                RainCompetition.to_libsvm(X, y, out)
            pb.finish()
            
        print('Validation set')
        name = 'valid_fold_{}'.format(i)
        nrows = read.get_storer(name).nrows
        pb = progressbar.ProgressBar(maxval=nrows//bs).start()
        with open('valid_folds_{}.txt'.format(i), 'wb') as out:
            for j,chunk in enumerate(read.select(name, iterator=True, chunksize=bs)):
                pb.update(j)
                X = chunk.drop('Expected', axis=1).values.astype('float')
                y = chunk['Expected'].values.clip(0,70).astype('int')
                RainCompetition.to_libsvm(X, y, out)
            pb.finish()
            
if __name__ == '__main__':
    do_fold_libsvm()