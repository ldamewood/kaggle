#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from os.path import join, dirname, realpath

import pandas as pd
import numpy as np

from kaggle.file import ProgressDictReader

class RainCompetition:
    __name__ = 'how-much-did-it-rain'
    __short_name__ = 'rain'
    __data__ = {
        'train': join(dirname(realpath(__file__)), 'data', 'train_2013.csv'),
        'train_ravel_missing': join(dirname(realpath(__file__)), 'data', 'train_miss.csv'),
        'train_ravel_imputed': join(dirname(realpath(__file__)), 'data', 'train_inpu.csv'),
        'train_ravel_imputed_normalized': join(dirname(realpath(__file__)), 'data', 'train_impu_norm.csv'),
        'test': join(dirname(realpath(__file__)), 'data', 'test_2014.csv'),
        'test_ravel_missing': join(dirname(realpath(__file__)), 'data', 'test_miss.csv'),
        'test_ravel_imputed': join(dirname(realpath(__file__)), 'data', 'test_inpu.csv'),
        'test_ravel_imputed_normalized': join(dirname(realpath(__file__)), 'data', 'test_impu_norm.csv'),
    }
    
    htypes = ['no echo', 'moderate rain', 'moderate rain2', 'heavy rain',
    'rain/hail', 'big drops', 'AP', 'Birds', 'unknown', 'no echo2',
    'dry snow', 'wet snow', 'ice crystals', 'graupel', 'graupel2']

    nan = float('nan')
    nan_values = [ '-99900.0', '-99901.0', '-99903.0', '999.0' ]
    not_features = [ 'Expected', 'Id' ]
    
    @classmethod    
    def process_row_(cls, row, deriv=True, ignore_keys=[]):
        """
        Process the csv file and do transformations:
            * Split the time series.
            * Standardize the NAN values. This must be done before taking the
              derivative, otherwise the derivative may be a number when it should
              be nan.
            * Calculate time derivatives if deriv == True.
            * Add NAN categories (removed)
        """
    
        # Extract the time series
        ntime = len(row['TimeToEnd'].split())
    
        # Split the row into a time series
        record = {}
        for key, value in row.iteritems():
            if key in ignore_keys:
                continue
            vals = value.split()
            vals = ['nan' if v in cls.nan_values else v for v in vals]
            if len(vals) > 1:
                record[key] = map(int, vals) if key in ['Id'] else map(float, vals)
            else:
                # Rows that contain a common value for all time steps
                record[key] = ntime * map(int, vals) if key in ['Id'] else ntime * map(float, vals)
        
        x = record['TimeToEnd']
        dx = [x[i] - x[i-1] for i in xrange(ntime)]
        
        # Add group index
#        j = -1
#        record['Group'] = []
#        for i in xrange(ntime):
#            if dx[i] > 0: j+=1
#            record['Group'].append(j)
        record['Index'] = range(ntime)
        
        for f, value in record.items():
            # Skip Id and Expected columns
            if f in RainCompetition.not_features: continue
            
            record[f] = value
            
            if deriv:
                # Don't take derivative of some features
                if f in [ 'TimeToEnd', 'HydrometeorType', 'Index', 'Id' ]: continue
                
                # Segment may contain multiple time series. They are separated 
                # by an increase in the TimeToEnd value. If dx < 0, then
                # take the derivative, otherwise make dy/dx = 0.
#                record['{}_deriv'.format(f)] = [0]*ntime
#                for i in range(ntime):
#                    if dx[i] >= 0 or np.isnan(value[i]) or np.isnan(value[i-1]):
#                        record['{}_deriv'.format(f)][i] = 0
#                    else:
#                        record['{}_deriv'.format(f)][i] = (value[i] - value[i-1])/dx[i]
                    
                record['{}_deriv'.format(f)] = [0 if (dx[i] >= 0 or np.isnan(value[i]) or np.isnan(value[i-1])) else (value[i] - value[i-1])/dx[i] for i in range(ntime)]
        
        # This part is akward. ;)
        for ht in cls.htypes:
            record[ht] = []
        for htval in record['HydrometeorType']:
            for ht in cls.htypes:
                record[ht].append(True) if cls.htypes[int(htval)] == ht else record[ht].append(False)
        del record['HydrometeorType']
    
        return record
    
    @classmethod
    def read_csv_(cls, filename, deriv=True, group=False, ignore_keys=[]):
        for row in ProgressDictReader(open(filename)):
            yield pd.DataFrame(cls.process_row_(row))

    @classmethod
    def ravel_(cls, train=True):
        index = ['Id', 'Index']
        filein = cls.__data__['train'] if train else cls.__data__['test']
        fileout = cls.__data__['train_ravel_missing'] if train else cls.__data__['test_ravel_missing']
        with open(fileout, 'w') as out:
            for i, chunk in enumerate(cls.read_csv_(filein)):
                chunk.set_index(index, inplace=True)
                if i==0:
                    chunk.to_csv(out, float_format='%e', index_label=index, header=True)
                else:
                    chunk.to_csv(out, float_format='%e', index_label=index, header=False)

    @classmethod
    def impute_(cls, train=True):
        index = ['Id', 'Index']
        filein = cls.__data__['train_ravel_missing'] if train else cls.__data__['test_ravel_missing']
        fileout = cls.__data__['train_ravel_imputed'] if train else cls.__data__['test_ravel_imputed']
        df = pd.read_csv(filein, index_col=index)
        print('Removing features without variance:')
        remove = []
        for col in df.columns:
            if col in index + ['Expected']:
                continue
            if col in cls.htypes:
                df[col].fillna(False, inplace='True')
            else:
                df[col].fillna(df[col].mean(), inplace='True')
            if df[col].std() < 1.e-5:
                remove.append(col)
                print('Removing column {}'.format(col))
                del df[col]
        df.to_csv(fileout, float_format='%e', index_label=index, header=True)

    @classmethod
    def normalize_(cls, train=True):
        index = ['Id', 'Index']
        filein = cls.__data__['train_ravel_imputed'] if train else cls.__data__['test_ravel_imputed']
        fileout = cls.__data__['train_ravel_imputed_normalized'] if train else cls.__data__['test_ravel_imputed_normalized']
        df = pd.read_csv(filein, index_col=index)
        for col in df.columns:
            if col in index + ['Expected'] + cls.htypes:
                continue
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean)/std
        df.to_csv(fileout, float_format='%e', index_label=index, header=True)

    @classmethod
    def shuffle_(cls, df):
        n_ids = len(df.index.levels[0].values)
        neworder = np.array(range(n_ids))
        np.random.shuffle(neworder)
        newindex = df.index[ np.argsort(neworder[df.index.labels[0]], kind='mergesort') ]
        return df.reindex(newindex)
        
    @classmethod
    def collapse(cls, y, ids):
        cols = ['Predicted{}'.format(i) for i in range(y.shape[1])]
        cols.append('Id')
        df = pd.DataFrame(np.hstack([y, ids]), columns=cols)
        return df.groupby('Id').mean().values
        
    @classmethod
    def score(cls, ypred, ytrue):
        x = range(70)
        return np.array([(ypred[:,n] - (n >= ytrue[:,0]))**2 for n in x]).sum()/len(x)/len(ytrue)

def shuffle(filein, fileout):
    index = ['Id', 'Index']
    df = RainCompetition.shuffle_(pd.read_csv(filein, index_col=index))
    df.to_csv(fileout, index_label=index, header=True)

if __name__ == '__main__':
    print('Creating training data...')
    print('\tRavelling...')
    RainCompetition.ravel_(train=True)
    print('\tImputing...')
    RainCompetition.impute_(train=True)
    print('\tNormalizing...')
    RainCompetition.normalize_(train=True)
    print('\tShuffling...')
    shuffle(RainCompetition.__data__['train_ravel_imputed_normalized'],
            'data/train_impu_norm_shuf.csv')
    print('Creating test data...')
    print('\tRavelling...')
    RainCompetition.ravel_(train=False)
    print('\tImputing...')
    RainCompetition.impute_(train=False)
    print('\tNormalizing...')
    RainCompetition.normalize_(train=False)