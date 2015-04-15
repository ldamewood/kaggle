"""
sklearn helpers for Poisson processes.
"""

from sklearn.preprocessing import PolynomialFeatures

from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from sklearn.metrics import euclidean_distances
from sklearn.metrics.pairwise import check_pairwise_arrays
from sklearn.cross_validation import train_test_split

import numpy as np

def poisson_kernel(X, Y=None, gamma=None, Sigma_inv = None):
    """
    Compute the poisson kernel between X and Y::
        K(x, y) = exp(-gamma ||x-mu||^2/mu)
        mu = centroid of X (=X if X.shape[0] == 1)
    for each pair of rows x in X and y in Y.
    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)
    Y : array of shape (n_samples_Y, n_features)
    gamma : float
    Returns
    -------
    kernel_matrix : array of shape (n_samples_X, n_samples_Y)
    """
    X, Y = check_pairwise_arrays(X, Y)
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    if Sigma_inv is None:
        raise ValueError('Missing Sigma_inv')
    
    v = X - Y
    K = -0.5 * gamma * np.sqrt(v.dot(Sigma_inv).dot(v.T))
    np.exp(K, K)    # exponentiate K in-place
    return K

class PoissonPolynomialFeatures(PolynomialFeatures):
    def __init__(self, degree=2):
        # Note the different default values from PolynomialFeatures:
        # * bias will always be zero.
        # * self-interaction is linear.
        super(PoissonPolynomialFeatures, self).__init__(degree = degree, 
            interaction_only = True, include_bias = False)
    
    def transform(self, X, y=None):
        """Transform data to polynomial features
        Parameters
        ----------
        X : array with shape [n_samples, n_features]
            The data to transform, row by row.
        Returns
        -------
        XP : np.ndarray shape [n_samples, NP]
            The matrix of features, where NP is the number of polynomial
            features generated from the combination of inputs.
        """
        check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = check_array(X)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)

        combinations = self._combinations(n_features, self.degree,
                                          self.interaction_only,
                                          self.include_bias)
        for i, c in enumerate(combinations):
            # Change prod to sum
            XP[:, i] = X[:, c].sum(1)

        return XP

from os.path import join, dirname, realpath

import sys
sys.path.append(join(dirname(realpath(__file__)), 'libxgboost'))
import xgboost as xgb

import otto
import pandas as pd
import numpy as np
import itertools as it
import scipy.stats as stats

from sklearn.decomposition import FastICA
from sklearn.cross_validation import StratifiedKFold

from xgboost import XGBClassifier

if __name__ == '__main__':
    
    ica_components = 10
    
    print('Loading train data...')
    data = pd.read_csv(otto.train, index_col = 'id')
    print('Number of training samples: %d' % len(data))
    feats = [col for col in data.columns if col not in ['target']]
    
    data['count_0'] = (data[feats]==0).sum(axis=1)
    data['count_1'] = (data[feats]==1).sum(axis=1)
    data['count_2'] = (data[feats]==2).sum(axis=1)
    
    ica_names = ['ica_{}'.format(i+1) for i in range(ica_components)]
    ica = FastICA(n_components = ica_components)
    icas = pd.DataFrame(ica.fit_transform(data[feats].values), columns = ica_names)
    icas.index += 1
    data = data.join(icas)
    y = data['target'].values
    del data['target']
    X = data.values
    
    folds = 10
    clfs = [XGBClassifier() for _ in range(folds)]
    skf = StratifiedKFold(labels = y, n_folds = 10, shuffle = True, random_state = 0)
    for clf, (train_index, test_index) in zip(clfs,skf):
        break
        
    