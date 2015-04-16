# -*- coding: utf-8 -*-

import subprocess
from progressbar import ProgressBar


def progress(source, count, batchsize = 100):
    """
    Progressbar iterator.
    """
    with ProgressBar(maxval=count) as progress:
        for i,s in enumerate(source):
            yield s
            if i%batchsize == 0:
                progress.update(i)


def wccount(filename):
    """
    Count the lines in a file using wc.
    """
    out = subprocess.Popen(['wc', '-l', filename],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT
                         ).communicate()[0]
    return int(out.strip().split()[0])


def records_groupby(dct, key):
    """
    Similar to pandas groupby method but for dicts of lists.
    """
    curval = dct[key][0]
    jstart = 0
    for j in xrange(len(dct[key])):
        if dct[key][j] != curval:
            yield {k: [v[i] for i in range(jstart, j)] for k,v in dct.iteritems()}
            jstart = j
            curval = dct[key][j]
    yield {k: [v[i] for i in range(jstart, j)] for k,v in dct.iteritems()}