#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from os.path import join, dirname, realpath

import pystats
import pandas as pd
import numpy as np
import scipy.stats

from kaggle.util import ProgressDictReader
from subprocess import check_call

class RainCompetition:
    __name__ = 'how-much-did-it-rain'
    __train__ = join(dirname(realpath(__file__)), 'data', 'train_2013.csv')
    __test__ = join(dirname(realpath(__file__)), 'data', 'test_2014.csv')

    nan = float('nan')
    nan_values = [ -99900.0, -99901.0, -99903.0, 999.0, float('nan') ]
    not_features = [ 'Expected', 'Id' ]

    @classmethod
    def read_csv_(cls, filename, deriv=True, group=True, ignore_keys=[]):
        """
        Process the csv file and do transformations:
            * Split the time series.
            * Standardize the NAN values. This must be done before taking the
              derivative, otherwise the derivative may be a number when it should
              be nan.
            * Calculate time derivatives if deriv == True.
            * Add NAN categories (removed)
        """

        for row in ProgressDictReader(open(filename, 'r')):
            # Extract the time series
            ntime = len(row['TimeToEnd'].split())
    
            # Split the row into a time series
            record = {}
            for key, value in row.iteritems():
                if key in ignore_keys: continue
                if len(value.split()) > 1:
                    record[key] = map(float, value.split())
                else:
                    # Rows that contain a common value for all time steps
                    record[key] = ntime * [float(value)]
            
            x = record['TimeToEnd']
            dx = [x[i] - x[i-1] for i in xrange(ntime)]
    
            # Add group index
            j = 0
            record['Group'] = []
            for i in xrange(ntime):
                if dx[i] > 0: j+=1
                record['Group'].append(j)
            
            for f, value in record.items():
                # Skip Id and Expected columns
                if f in cls.not_features: continue
                
                # Convert values to float or nan
                record[f] = [float('nan') if v in cls.nan_values else v for v in value]
                
                if deriv:
                    # Don't take derivative of some features
                    if f in [ 'TimeToEnd', 'HydrometeorType' ]: continue
                    
                    # Segment may contain multiple time series. They are separated 
                    # by an increase in the TimeToEnd value. If dx < 0, then
                    # take the derivative, otherwise make dy/dx = 0.
                    record['{}_deriv'.format(f)] = [(value[i] - value[i-1])/dx[i] if dx[i] < 0 else 0 for i in range(ntime)]
            
            cls.process_htypes_(record)
    
            yield record    
    
    @classmethod
    def process_htypes_(cls, d):
        """
        Manually create HydrometeorType features.
        """
        htypes = ['no echo', 'moderate rain', 'moderate rain2', 'heavy rain',
            'rain/hail', 'big drops', 'AP', 'Birds', 'unknown', 'no echo2',
            'dry snow', 'wet snow', 'ice crystals', 'graupel', 'graupel2']
        
        # This part is akward. ;)
        for ht in htypes:
            d[ht] = []
        for htval in d['HydrometeorType']:
            for ht in htypes:
                d[ht].append(True) if htypes[int(htval)] == ht else d[ht].append(False)
        del d['HydrometeorType']
    
    @classmethod
    def group_data_(cls, records):
        for r in records:
            for grp in util.records_groupby(r, 'group'):
                df = {}
                t = grp['TimeToEnd']
                for k,v in grp.iteritems():
                    df['{}_mean'.format(k)] = np.mean(v)
                    df['{}_std'.format(k)] = np.std(v)
                    df['{}_min'.format(k)] = np.min(v)
                    df['{}_max'.format(k)] = np.max(v)
                    df['{}_range'.format(k)] = np.max(v) - np.min(v)
                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(v,t)
                    df['{}_slope'.format(k)] = slope
                    df['{}_intercept'.format(k)] = intercept
                yield df    
    
    @classmethod
    def find_ignorable_features(cls, deriv=True, group=False, stdmin=1.e-5):
        fstats = {}
        for df in cls.read_csv_(cls.__train__, deriv=deriv):
            for key,val in df.iteritems():
                if key not in fstats:
                    fstats[key] = pystats.Accumulator()
                for v in val:
                    fstats[key].push(v)
                # TODO: Update Accumulator to accept push(val)
        return [key for key,col in fstats.iteritems() if col.std() < stdmin]

    @classmethod
    def read_df_(cls, train=True, ignore_keys=[], deriv=True, group=False):
        for df in cls.read_csv_(cls.__train__ if train else cls.__test__,
                                deriv=deriv, group=group,
                                ignore_keys=ignore_keys):
            yield pd.DataFrame(df)

    @classmethod
    def load_data(cls, train=True, ignore_keys=[], deriv=True, group=False):
        return pd.concat(cls.read_df_(train=train, ignore_keys=ignore_keys,
                                      deriv=deriv, group=group))

    @classmethod
    def save_data(cls, df, filename):
        df.to_csv(filename, index = False)
        check_call(['gzip', '-f',  filename])
        
    @classmethod
    def score(cls, yp, yr):
        x = range(70)
        return np.array([(yp[:,n] - (n >= yr))**2 for n in x]).T / len(x) / len(yr)     

if __name__ == '__main__':
    print("Pass #1: Find ignorable features")
    ignore_keys = RainCompetition.find_ignorable_features()
    print("Pass #2: Loading training data")
    train_df = RainCompetition.load_data(train=True, ignore_keys=ignore_keys)
    print("Saving new training data")
    RainCompetition.save_data(train_df, join('data', 'train_with_deriv.csv'), index=False)
    print("Loading testing data")
    test_df = RainCompetition.load_data(train=False, ignore_keys=ignore_keys)
    print("Saving testing data")
    RainCompetition.save_data(test_df, join('data', 'test_with_deriv.csv'), index=False)
    print("Complete!")