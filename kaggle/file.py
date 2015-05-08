from __future__ import division

import subprocess
import progressbar
import csv
import pandas as pd
import numpy as np

from sklearn.feature_extraction import FeatureHasher

class ProgressDictReader(csv.DictReader):
    """
    A copy of DictReader but with a progressbar indicator
    """
    def __init__(self, f, fieldnames=None, restkey=None, restval=None,
                 dialect="excel", *args, **kwds):
        csv.DictReader.__init__(self, f, fieldnames=fieldnames,
            restkey=restkey, restval=restval, dialect=dialect, *args, **kwds)
        n_lines = line_count(f.name)
        self.progressbar = progressbar.ProgressBar(maxval=n_lines-1).start()
                 
    def __iter__(self):
        return self
    
    def next(self):
        self.progressbar.update(min(self.progressbar.currval+1, self.progressbar.maxval-1))
        try:
            return csv.DictReader.next(self)
        except StopIteration:
            self.progressbar.finish()
            raise StopIteration

def line_count(filename):
    """
    Count the lines in a file using wc.
    TODO: use pure python method if this one fails (i.e. Windows)
    """
    p = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE, 
                                                 stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

def to_libsvm(dfin, outfile, target_column, labelEncoder = None):
    pass

def dump_libffm_format(X, y, f):
    one_based = True
    hasher = FeatureHasher(input_type='string', non_negative=True)
    Xt = hasher.transform(X)
    value_pattern = u'%d:%d:%.16g'
    line_pattern = u'%d %s\n'
    for i in xrange(Xt.shape[0]):
        span = slice(Xt.indptr[i], Xt.indptr[i+1])
        row = zip(range(len(Xt.indices[span])), Xt.indices[span], Xt.data[span])
        s = " ".join(value_pattern % (j + one_based, fe, x) for j, fe, x in row)
        feat = (y[i], s)
        f.write((line_pattern % feat).encode('ascii'))
        
def combine_result_files(files, index_col='Id'):
    all_preds = np.array([pd.read_csv(f, index_col=index_col).values for f in files])
    all_preds = np.exp(np.log(all_preds).mean(axis=0))
    all_preds /= all_preds.sum(axis=1)[:, np.newaxis]
    return all_preds