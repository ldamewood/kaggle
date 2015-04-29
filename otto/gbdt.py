#!/usr/bin/env python
from __future__ import print_function

import otto
import random
import numpy

from poisson import PoissonPolynomialFeatures

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline

print('Loading data...')
allX = []
ally = []
for Id,X,y in otto._data(otto.train):
    allX.append(X)
    ally.append(y)
print('Number of samples: %d' % len(ally))

params = dict()
step1 = DictVectorizer(sparse=False)
step2 = PoissonPolynomialFeatures()
step3 = GradientBoostingClassifier(random_state = 0, max_features = None, n_estimators = 300, verbose = True)
clf = make_pipeline(step1, step2)

#print('Grid Search')
#gsearch = GridSearchCV(clf, param_grid=params,n_jobs=4,cv=3,scoring="log_loss")
X = clf.fit_transform(allX)


#print('Writing submit file...')
#otto.create_submit_file(clf, 'submit_gsearch.csv.gz')