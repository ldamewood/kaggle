# -*- coding: utf-8 -*-

from otto import OttoCompetition, OttoScaler
import numpy as np
from sklearn.cross_validation import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss

import progressbar

np.random.seed(0)

if __name__ == '__main__':
    encoder = LabelEncoder()
    scaler = OttoScaler()
    X, y = OttoCompetition.load_data(train=True)
    X = scaler.fit_transform(X).astype('float32')
    y = encoder.fit_transform(y).astype('int32')
    num_classes = len(encoder.classes_)
    num_features = X.shape[1]

    Xt, X_test, yt, y_test = train_test_split(X, y, test_size = 0.2)

    params = dict(
        loss='modified_huber',   
        penalty='elasticnet',
        alpha=0.0001,
        l1_ratio=0.15,
        n_iter=5,
        n_jobs=-1,
        warm_start=True,
    )

    kncs, losses = [], []
    n_folds = 10
    bs = 256
    print("Fitting...")
    for i, (train_index, valid_index) in enumerate(StratifiedKFold(yt, n_folds = n_folds, random_state=0)):
        print('Fold {}'.format(i))
        X_train, X_valid = Xt[train_index], Xt[valid_index]
        y_train, y_valid = yt[train_index], yt[valid_index]
        kncs.append(SGDClassifier(loss='modified_huber', random_state=0))
        pb = progressbar.ProgressBar(maxval=bs).start()
        for j in xrange(len(y_train) // bs):
            pb.update(j)
            s = slice(bs * j, bs * (j+1))
            Xp = PolynomialFeatures().fit_transform(X_train[s])
            kncs[-1].fit(Xp, y_train[s])
        pb.finish()
        y_pred = np.zeros([y_valid.shape[0], num_classes])
        for j in xrange(len(y_valid) // bs):
            s = slice(bs * j, bs * (j+1))
            Xp = PolynomialFeatures().fit_transform(X_valid[s,:])
            y_pred[s] = kncs[-1].predict_proba(Xp)
        losses.append(log_loss(y_valid, y_pred))
        print(losses[-1])

    losses = np.array(losses, dtype='float')
    print(np.mean(losses), np.std(losses))
    
    y_pred = np.zeros([y_test.shape[0], num_classes])
    for knc in kncs:
        for j in xrange(len(X_test.shpe[0]) // bs):
            s = slice(bs * j, bs * (j+1))
            Xp = PolynomialFeatures().fit_transform(X_test[s,:])
            y_pred[s,:] += knc.predict_proba(Xp)
    y_pred /= len(kncs)
    print('Test set log loss: {}'.format(log_loss(y_test, y_pred)))

#    X_test, _ = OttoCompetition.load_data(train=False)
#    X_test = scaler.fit_transform(X_test).astype('float32')
#    y_preds = []
#    for knc in kncs:
#        y_preds.append(knc.predict(X_test))
#    y_pred = sum(y_preds)/n_folds
#
#    OttoCompetition.save_data(y_pred)