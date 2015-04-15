#!/usr/bin/env python
from __future__ import print_function

import otto
import random

from hashlib import sha1

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score

print('Loading data...')
allX = []
ally = []
for Id,X,y in otto.train:
    allX.append(X)
    ally.append(y)

print('Number of samples: %d' % len(ally))

params = dict(gradientboostingclassifier__subsample=[1.0,0.5])
dv = DictVectorizer(sparse=False)
gbdt = GradientBoostingClassifier(max_features = 5, random_state = 0, verbose = True)
clf = make_pipeline(dv, gbdt)

print('Grid Search')
gsearch = GridSearchCV(clf,param_grid=params,cv=2,refit=True,scoring="log_loss", n_jobs = -1)
gsearch.fit(allX,ally)
print(gsearch.best_estimator_)

print('Writing submit file...')
otto.create_submit_file(gsearch.best_estimator_, 'test.csv.gz')
print(sha1("otto\0" + open('test.csv.gz','rb').read()).hexdigest())
