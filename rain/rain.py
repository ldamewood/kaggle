#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

from csv import DictReader
from os.path import join, dirname, realpath

import pystats
import pandas as pd
import numpy as np
import scipy.stats

from kaggle import util

datapath = join(dirname(realpath(__file__)), 'data')

class RainCompetition:

    train = join(datapath, 'train_2013.csv')
    test = join(datapath, 'test_2014.csv')
    
    nan = float('nan')
    nan_values = [ -99900.0, -99901.0, -99903.0, 999.0, float('nan') ]
    not_features = [ 'Expected', 'Id' ]
    
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
    def find_ignoreable_features_(cls, deriv = True, stdmin = 1.e-5):
        size = util.wccount(cls.train) - 1
        fstats = {}
        for df in util.progress(cls.process_csv_(cls.train, deriv = deriv), size):
            for key,val in df.iteritems():
                if key not in fstats:
                    fstats[key] = pystats.Accumulator()
                for v in val:
                    fstats[key].push(v)
                # Update Accumulator to accept push(val)
        return [key for key,col in fstats.iteritems() if col.std() < stdmin]

    @classmethod
    def load_data_(cls, filename, ignore_keys = [], deriv = True):
        size = util.wccount(filename) - 1
        for df in util.progress(cls.process_csv_(filename, deriv = deriv), size):
            for key in ignore_keys:
                del df[key]
            yield pd.DataFrame(df)
    
    @classmethod
    def load_dataframe(cls, train = True, ignore_keys = []):
        return pd.concat(cls.load_data_(cls.train if train else cls.test,
                                        ignore_keys = ignore_keys))

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
    def process_csv_(cls, filepath, deriv = True, group = True):
        """
        Process the csv file and do transformations:
            * Split the time series.
            * Standardize the NAN values. This must be done before taking the
              derivative, otherwise the derivative may be a number when it should
              be nan.
            * Calculate time derivatives if deriv == True.
            * Add NAN categories (removed)
        """
        # features that cannot have nan and are not used in poly2 or deriv
        
        for row in DictReader(open(filepath)):
            # Extract the time series
            ntime = len(row['TimeToEnd'].split())
    
            # Split the row into a time series
            d = {}
            for key, value in row.iteritems():
                if len(value.split()) > 1:
                    d[key] = map(float, value.split())
                else:
                    # Rows that contain a common value for all time steps
                    d[key] = ntime * [float(value)]
            
            x = d['TimeToEnd']
            dx = [x[i] - x[i-1] for i in xrange(ntime)]
    
            # Add group index
            j = 0
            d['Group'] = []
            for i in xrange(ntime):
                if dx[i] > 0: j+=1
                d['Group'].append(j)
            
            for f, value in d.items():
                # Skip Id and Expected columns
                if f in cls.not_features: continue
                
                # Convert values to float or nan
                d[f] = [float('nan') if v in cls.nan_values else v for v in value]
                
                if deriv:
                    # Don't take derivative of some features
                    if f in [ 'TimeToEnd', 'HydrometeorType' ]: continue
                    
                    # Segment may contain multiple time series. They are separated 
                    # by an increase in the TimeToEnd value. If dx < 0, then
                    # take the derivative, otherwise make dy/dx = 0.
                    d['{}_deriv'.format(f)] = [(value[i] - value[i-1])/dx[i] if dx[i] < 0 else 0 for i in range(ntime)]
            
            cls.process_htypes_(d)
    
            yield d

#def group_and_mean(array, group_ids, logit = False):
#    """
#    Group by "group_ids" and take the mean.
#    can this be easily done without pandas? Of course!
#    """
#    
#    # Check if ids are given in sorted order
#    #assert sum([group_ids[i] != j for i,j in enumerate(sorted(group_ids))]) == 0
#    df = pd.DataFrame(array)
#    df['Group'] = group_ids
#    if logit:
#        return np.array(df.groupby('Group').agg(lambda x: 1./(1+np.exp(-np.mean(np.log(x/(1-x)))))))
#    else:
#        return np.array(df.groupby('Group').mean())
#
#def score_crp(y_pred, y_real, ids):
#    """
#    Gives the score based on the classification probability and expected values.
#    """
#    yp = np.array(group_and_mean(y_pred, ids)).cumsum(axis=1)
#    ya = np.array(group_and_mean(y_real, ids)).flatten()
#    x = range(70)
#    return np.array([(yp[:,n] - (n >= ya))**2 for n in x]).T / len(x) / len(ya)       

if __name__ == '__main__':
    print('Pass #1: Search for bad features')
    ignore = RainCompetition.find_ignoreable_features_()
    print('Pass #2: Load data into DataFrame object')
    train_df = RainCompetition.load_dataframe(train=True, ignore_keys=ignore)
#print(util.wccount(RainCompetition.train))