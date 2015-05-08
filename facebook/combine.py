# -*- coding: utf-8 -*-

import pandas as pd
from itertools import izip
import numpy as np
import glob
from facebook import FacebookCompetition

print('Loading test data')
bids = pd.read_csv(FacebookCompetition.__data__['bids'], index_col='bid_id')
test = pd.read_csv(FacebookCompetition.__data__['test'], index_col='bidder_id')
te = test.merge(bids, left_index=True, right_on='bidder_id', how='left')
del bids

files = glob.glob('data/facebook.te.*.txt')
its = [iter(pd.read_table(f, header=-1, iterator=True, chunksize=2**15)) for f in files]
#with open('data/facebook_softmax_20150506.csv', 'w') as out:
c = []
for i,chunks in enumerate(izip(*its)):
    print(i)
    A = np.array([np.c_[chunk.values,1-chunk.values] for chunk in chunks])
    A = np.exp(np.log(A).mean(axis=0))
    A /= A.sum(axis=1)[:, np.newaxis]
    A = A[:,0]
    df = pd.DataFrame(A)
    df.index = chunks[0].index
    df.columns = chunks[0].columns
    c.append(df)

df = pd.concat(c)
df.index = te.bidder_id
df = df.groupby(level=0).mean()
df.columns = ['prediction']