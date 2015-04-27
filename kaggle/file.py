from __future__ import division

import os
import multiprocessing as mp

import subprocess
import progressbar
import csv

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
        return csv.DictReader.next(self)

def line_count(filename):
    """
    Count the lines in a file using wc.
    TODO: use pure python method if this one fails.
    """
    p = subprocess.Popen(['wc', '-l', filename], stdout=subprocess.PIPE, 
                                                 stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])