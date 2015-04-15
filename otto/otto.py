#!/usr/bin/env python

from csv import DictReader
from os.path import join, dirname, realpath
from math import log
import gzip
import random

import numpy as np

random.seed(0)

datapath = join(dirname(realpath(__file__)), 'data')

#def _data(dataset):
#    """
#    Read the datafile and return the Id, features and outcome.
#
#    `features` is a dict of feature:value combinations.
#    """
#
#    for row in DictReader(open(dataset)):
#        Id = row['id']
#        # Ignore features that are zero
#        features = {i:int(j) for i,j in row.iteritems() if i[:5] == 'feat_' and int(j) > 0}
#        outcome = row['target'] if 'target' in row else None
#        yield Id,features,outcome

def logloss(y_true, y_pred, classes):
    """ Log Loss for Kaggle. """
    idx = [list(classes).index(y) for y in y_true]
    logloss = sum([-log(max(min(y[i],1. - 1.e-15),1.e-15)) for y,i in zip(y_pred,idx)])
    return logloss / len(y_true)

#def create_submit_file(clf, filename):
#    with gzip.open(filename, 'wb') as fout:
#        fout.write('id,')
#        fout.write(','.join(clf.classes_))
#        fout.write('\n')
#        for Id,X,y in test:
#            probs = {cls:pr for cls,pr in zip(clf.classes_,clf.predict_proba(X)[0])}
#            fout.write('%s,' % Id)
#            fout.write(','.join(['%0.4f' % probs[cls] for cls in clf.classes_]))
#            fout.write('\n')

#def equalize(allX, ally, method = 'expand'):
#    classes = list(set(ally))
#    counts = {}
#    indexes = {}
#    for cls in classes:
#        counts[cls] = ally.count(cls)
#        indexes[cls] = [i for i,y in enumerate(ally) if y == cls]
#    newX = []
#    newy = []
#    if method == 'expand':
#        k = max(counts.values())
#        newX = [x for x in allX]
#        newy = [y for y in ally]
#        for cls in classes:
#            for i in range(counts[cls], k):
#                newX.append(allX[random.choice(indexes[cls])])
#                newy.append(cls)
#    else:
#        # Shrink
#        k = len(ally) / len(classes)
#        for cls in classes:
#            for i in range(k): 
#                newX.append(allX[random.choice(indexes[cls])])
#                newy.append(cls)
#    return allX,ally

train = join(datapath, 'train.csv')
test = join(datapath, 'test.csv')