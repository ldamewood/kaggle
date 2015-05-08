# -*- coding: utf-8 -*-

from rain import inv_cumsum
import pandas as pd
from itertools import izip
import numpy as np

files = ['data/rain_pred.csv','data/rain_20150430.csv']
its = [iter(pd.read_csv(f, index_col='Id', iterator=True, chunksize=2**10)) for f in files]
with open('data/rain_softmax_20150507.csv', 'w') as out:
    header = True
    for i,chunks in enumerate(izip(*its)):
        print(i)
        A = np.array([inv_cumsum(chunk.values) for chunk in chunks])
        A = np.exp(np.log(A).mean(axis=0))
        A /= A.sum(axis=1)[:, np.newaxis]
        A = A.cumsum(axis=1)[:,:70]
        df = pd.DataFrame(A)
        df.index = chunks[0].index
        df.columns = chunks[0].columns
        df.to_csv(out, index_label='Id', float_format='%8.6f', header=header)
        header = False