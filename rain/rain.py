#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from os.path import join, dirname, realpath

import pystats
import pandas as pd
import numpy as np
import scipy.stats
import itertools

from csv import DictReader

from kaggle.file import ProgressDictReader
from kaggle.util import records_groupby
from subprocess import check_call
import multiprocessing as mp

class RainCompetition:
    __name__ = 'how-much-did-it-rain'
    __train__ = join(dirname(realpath(__file__)), 'data', 'train_2013.csv')
    __test__ = join(dirname(realpath(__file__)), 'data', 'test_2014.csv')
    __train_split__ = join(dirname(realpath(__file__)), 'data', 'train_split_rows.csv')
    __test_split__ = join(dirname(realpath(__file__)), 'data', 'test_split_rows.csv')

    nan = float('nan')
    nan_values = [ '-99900.0', '-99901.0', '-99903.0', '999.0' ]
    not_features = [ 'Expected', 'Id' ]
    
    @classmethod    
    def process_row_(cls, row, deriv=True, group=False, ignore_keys=[]):
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
        j = 0
        record['Group'] = []
        for i in xrange(ntime):
            if dx[i] < 0: j+=1
            record['Group'].append(j)
        
        for f, value in record.items():
            # Skip Id and Expected columns
            if f in RainCompetition.not_features: continue
            
            record[f] = value
            
            if deriv:
                # Don't take derivative of some features
                if f in [ 'TimeToEnd', 'HydrometeorType' ]: continue
                
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
                
        
        htypes = ['no echo', 'moderate rain', 'moderate rain2', 'heavy rain',
            'rain/hail', 'big drops', 'AP', 'Birds', 'unknown', 'no echo2',
            'dry snow', 'wet snow', 'ice crystals', 'graupel', 'graupel2']
        
        # This part is akward. ;)
        for ht in htypes:
            record[ht] = []
        for htval in record['HydrometeorType']:
            for ht in htypes:
                record[ht].append(True) if htypes[int(htval)] == ht else record[ht].append(False)
        del record['HydrometeorType']
    
        return record
    
    @classmethod
    def read_csv_(cls, filename, deriv=True, group=False, ignore_keys=[]):
        for row in ProgressDictReader(open(filename)):
            yield pd.DataFrame(cls.process_row_(row))
    
#    @classmethod
#    def group_data_(cls, records):
#        for r in records:
#            for grp in records_groupby(r, 'group'):
#                df = {}
#                t = grp['TimeToEnd']
#                for k,v in grp.iteritems():
#                    df['{}_mean'.format(k)] = np.mean(v)
#                    df['{}_std'.format(k)] = np.std(v)
#                    df['{}_min'.format(k)] = np.min(v)
#                    df['{}_max'.format(k)] = np.max(v)
#                    df['{}_range'.format(k)] = np.max(v) - np.min(v)
#                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(v,t)
#                    df['{}_slope'.format(k)] = slope
#                    df['{}_intercept'.format(k)] = intercept
#                yield df    
    
#    @classmethod
#    def find_ignorable_features(cls, deriv=True, group=False, stdmin=1.e-5, n_threads=2):
#        fstats={}        
#        for df in cls.read_csv_(cls.__train__, deriv=deriv):
#            for key,val in df.iteritems():
#                if key not in fstats:
#                    fstats[key] = pystats.Accumulator()
#                for v in val:
#                    fstats[key].push(v)
#                # TODO: Update Accumulator to accept push(<list>)
#        return [key for key,col in fstats.iteritems() if col.std() < stdmin]

    @classmethod
    def process_df(cls, outfile, train=True, ignore_keys=[], deriv=True, group=False):
        with open(outfile, 'w') as out:
            for i, chunk in enumerate(cls.read_csv_(RainCompetition.__train__ if train else RainCompetition.__test__, ignore_keys=ignore_keys, deriv=deriv, group=group)):
                chunk.set_index(['Id','Group'], inplace=True)
                if i==0: 
                    chunk.to_csv(out, index_label=['Id','Group'], header=True)
                else:
                    chunk.to_csv(out, index_label=['Id','Group'], header=False)
#        check_call(['gzip', '-f', outfile])
#                    
#
#    @classmethod
#    def read_df_(cls, train=True, ignore_keys=[], deriv=True, group=False):
#        for df in cls.read_csv_(cls.__train__ if train else cls.__test__,
#                                deriv=deriv, group=group,
#                                ignore_keys=ignore_keys):
#            yield pd.DataFrame(df)
#
#    @classmethod
#    def load_data(cls, train=True, ignore_keys=[], deriv=True, group=False):
#        return pd.concat(cls.read_df_(train=train, ignore_keys=ignore_keys,
#                                      deriv=deriv, group=group))
#
#    @classmethod
#    def save_data(cls, df, filename):
#        df.to_csv(filename, index = False)
#        check_call(['gzip', '-f',  filename])
#        
    @classmethod
    def collapse(cls, y, ids):
        cols = ['Predicted{}'.format(i) for i in range(71)]
        cols.append('Id')
        df = pd.DataFrame(np.hstack(y, ids), columns=cols)
        return df.groupby('Id').mean()
        
    @classmethod
    def score(cls, yp, yr):
        x = range(70)
        return np.array([(yp[:,n] - (n >= yr))**2 for n in x]).T / len(x) / len(yr)             

if __name__ == '__main__':
    print("Pass #2: Processing training data")
    RainCompetition.process_df(RainCompetition.__train_split__, train=True)
    print("Pass #3: Processing testing data")
    RainCompetition.process_df(RainCompetition.__test_split__, train=False)
    print("Complete!")