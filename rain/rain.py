#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from os.path import join, dirname, realpath

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from kaggle.file import ProgressDictReader
import xgboost as xgb
import progressbar

from kaggle import KaggleCompetition

class RainCompetition(KaggleCompetition):
    __full_name__ = 'how-much-did-it-rain'
    __short_name__ = 'rain'
    __data_path__ = 'data'
    __data__ = {
        'train': join(dirname(realpath(__file__)), 'data', 'train_2013.csv'),
        'train_missing': join(dirname(realpath(__file__)), 'data', 'train_miss.h5'),
        'train_imputed': join(dirname(realpath(__file__)), 'data', 'train_impu.h5'),
        'train_normalized': join(dirname(realpath(__file__)), 'data', 'train_norm.h5'),
        'test': join(dirname(realpath(__file__)), 'data', 'test_2014.csv'),
        'test_missing': join(dirname(realpath(__file__)), 'data', 'test_miss.h5'),
        'test_imputed': join(dirname(realpath(__file__)), 'data', 'test_impu.h5'),
        'test_normalized': join(dirname(realpath(__file__)), 'data', 'test_norm.h5'),
    }
    
    htypes = ['no echo', 'moderate rain', 'moderate rain2', 'heavy rain',
    'rain/hail', 'big drops', 'AP', 'Birds', 'unknown', 'no echo2',
    'dry snow', 'wet snow', 'ice crystals', 'graupel', 'graupel2']
    
    htypes_tr = ['no_echo', 'moderate_rain', 'moderate_rain2', 'heavy_rain',
    'rain_hail', 'big_drops', 'AP', 'Birds', 'unknown', 'no_echo2',
    'dry_snow', 'wet_snow', 'ice_crystals', 'graupel', 'graupel2']
    
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
            httr = ht.replace('/','_').replace(' ','_')
            record[httr] = []
        for htval in record['HydrometeorType']:
            for ht in cls.htypes:
                httr = ht.replace('/','_').replace(' ','_')
                record[httr].append(True) if cls.htypes[int(htval)] == ht else record[httr].append(False)
        del record['HydrometeorType']

        return [{k:v[i] for k, v in record.items()} for i in range(len(record.values()[0]))]
    
    @classmethod
    def read_csv_(cls, filename, batch=2**15):
        index = ['Id', 'Index']
        rows = []
        for i,row in enumerate(ProgressDictReader(open(filename))):
            rows.extend(cls.process_row_(row))
            if i>0 and i%batch==0:
                yield pd.DataFrame(rows).set_index(index)
                rows = []
        yield pd.DataFrame(rows).set_index(index)

    @classmethod
    def ravel_(cls, train=True):
        filein = cls.__data__['train' if train else 'test']
        fileout = cls.__data__['train_missing' if train else 'test_missing']

        with pd.HDFStore(fileout, mode='w') as store:
            for chunk in cls.read_csv_(filein):
                store.append('df', chunk, data_columns=True)

    @classmethod
    def impute_(cls, train=True):
        filein = cls.__data__['train_missing' if train else 'test_missing']
        fileout = cls.__data__['train_imputed' if train else 'test_imputed']
        remove = []
        with pd.HDFStore(fileout, mode='w') as store, pd.HDFStore(filein, mode='r') as read:
            means, stds = {}, {}
            print('Reading column-wise:')
            for col in read.get_storer('df').data_columns:
                print('Column: {}'.format(col))
                column = read.select_column('df',col)
                means[col], stds[col] = column.mean(), column.std()
                if stds[col] < 1.e-5:
                    remove.append(col)
                    print('  -- To remove')
            print('Imputing file:')
            bs = 2**15
            pb = progressbar.ProgressBar(maxval=read.get_storer('df').nrows//bs).start()
            for i,chunk in enumerate(read.select('df', chunksize = bs)):
                pb.update(i)
                for col in chunk.columns:
                    if col in ['Expected']:
                        continue
                    if col in cls.htypes_tr:
                        chunk[col].fillna(False, inplace=True)
                    else:
                        chunk[col].fillna(means[col], inplace=True)
                    if col in remove:
                        del chunk[col]
                store.append('df', chunk, data_columns=True)
            pb.finish()

    @classmethod
    def normalize_(cls, train=True):
        filein = cls.__data__['train_imputed'] if train else cls.__data__['test_imputed']
        fileout = cls.__data__['train_normalized'] if train else cls.__data__['test_normalized']
        with pd.HDFStore(fileout, mode='w') as store, pd.HDFStore(filein, mode='r') as read:
            means, stds = {}, {}
            print('Reading column-wise:')
            for col in read.get_storer('df').data_columns:
                print('Column: {}'.format(col))
                column = read.select_column('df',col)
                means[col], stds[col] = column.mean(), column.std()
            print('Imputing file:')
            bs = 2**15
            pb = progressbar.ProgressBar(maxval=read.get_storer('df').nrows//bs).start()
            for i,chunk in enumerate(read.select('df', chunksize = bs)):
                pb.update(i)
                for col in chunk.columns:
                    if col in ['Expected']:
                        continue
                    if col not in cls.htypes_tr:
                        chunk[col] = (chunk[col] - means[col]) / stds[col]
                store.append('df', chunk, data_columns=True)
            pb.finish()

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
    def score(cls, ytrue, ypred):
        x = range(70)
        return np.array([(ypred[:,n] - (n >= ytrue[:,0]))**2 for n in x]).sum()/len(x)/len(ytrue)

    @classmethod
    def prediction_weights(cls, y_preds, y_real):

        def log_loss_func(weights):
            weights /= np.sum(weights)
            yp = sum([w*y for w,y in zip(weights, y_preds)])
            return cls.score(y_real, yp)
        
        bounds = [(0,1)]*len(y_preds)
        starting = [1.]*len(y_preds)
        res = minimize(log_loss_func, starting, method='L-BFGS-B', bounds=bounds)
        return res

    @classmethod
    def collapse_file(cls, filein, fileout):
        it = pd.read_csv(filein, iterator=True, chunksize=1, index_col='Index', compression='gzip')
        Id = 1
        rows = []
        header=True
        with open(fileout, 'w') as out:
            for i,chunk in enumerate(it):
                if i%1000==0: print(i)
                if Id != chunk['Id'].values.ravel()[0]:
                    df = pd.concat(rows)
#                    print(df['Id'].values.ravel())
                    if df.shape[0] > 1:
                        df = df.groupby('Id').mean().cumsum(axis=1)
                    else:
                        df.index = df['Id']
                        df = df.drop('Id', axis=1).cumsum(axis=1)
                    df.to_csv(out, index_label='Id', header=header)
                    header=False
                    rows = []
                    Id = chunk['Id'].values.ravel()[0]
                rows.append(chunk.copy())

    @classmethod
    def do_predict(cls, clf, hdf_file, fileout):
        ycols = ['Predicted{}'.format(i) for i in range(70)]
        header = True
        with pd.HDFStore(hdf_file, mode='r') as read, open(fileout, 'w') as out:
            print('Grouping...')
            groups = read.df.groupby(level=0)
            pb = progressbar.ProgressBar(maxval=groups.ngroups).start()
            idx = []
            rows = []
            bs=2**15
            for i,k in enumerate(groups.groups.keys()):
                pb.update(i)
                y_pred = clf.predict_proba(groups.get_group(k).values).mean(axis=0).cumsum()[:70]
                idx.append(k)
                rows.append(y_pred)
                if i > 0 and i % bs==0:
                    df = pd.DataFrame(rows, columns=ycols, index=idx)
                    df.to_csv(out, index_label='Id', header=header)
                    header=False
                    rows, idx = [], []
            df = pd.DataFrame(rows, columns=ycols, index=idx)
            df.to_csv(out, index_label='Id', header=header)
            pb.finish()

    @classmethod
    def do_predict2(cls, clf, hdf_file, fileout):
        ycols = ['Predicted{}'.format(i) for i in range(70)]
        header = True
        with pd.HDFStore(hdf_file, mode='r') as read, open(fileout, 'w') as out:
            print('Grouping...')
            groups = read.df.groupby(level=0)
            pb = progressbar.ProgressBar(maxval=groups.ngroups).start()
            idx = []
            rows = []
            bs=2**15
            for i,k in enumerate(groups.groups.keys()):
                pb.update(i)
                y_pred = clf.predict(xgb.DMatrix(groups.get_group(k).values, missing=np.nan)).mean(axis=0).cumsum()[:70]
                idx.append(k)
                rows.append(y_pred)
                if i > 0 and i % bs==0:
                    df = pd.DataFrame(rows, columns=ycols, index=idx)
                    df.to_csv(out, index_label='Id', header=header)
                    header=False
                    rows, idx = [], []
            df = pd.DataFrame(rows, columns=ycols, index=idx)
            df.to_csv(out, index_label='Id', header=header)
            pb.finish()
    
    @classmethod
    def to_libsvm(cls, X, y, f):
        if hasattr(X, "tocsr"):
            raise ValueError    
        
        if X.dtype.kind == 'i':
            value_pattern = u"%d:%d"
        else:
            value_pattern = u"%d:%.16g"
    
        if y.dtype.kind == 'i':
            line_pattern = u"%d"
        else:
            line_pattern = u"%.16g"
        line_pattern += u" %s\n"
        
        for i in range(X.shape[0]):
            nz = np.isnan(X[i]) == False
            row = zip(np.where(nz)[0], X[i, nz])
            
            s = " ".join(value_pattern % (j + 1, x) for j, x in row)
            feat = (y[i], s)
            f.write((line_pattern % feat).encode('ascii'))

def shuffle(filein, fileout):
    index = ['Id', 'Index']
    df = RainCompetition.shuffle_(pd.read_csv(filein, index_col=index))
    df.to_csv(fileout, index_label=index, header=True)

