import sys
import os
import glob
import json

import numpy as np
import pandas as pd
import sklearn.model_selection

from utils.data_utils import get_iaga_data, get_omni_data

BUCKETS = 100
LENGTH = 600
LAG = 1

def get_sequences(df, length, lag):
    _df = df.copy()
    _df['cluster'] = (_df['seconds'].diff() != 60).cumsum()
    f = _df.groupby(['cluster']).apply(lambda x: x[:len(x)-length-lag]['index']).reset_index(drop=True)
    t = _df.groupby(['cluster']).apply(lambda x: x[length+lag:]['index']).reset_index(drop=True)
    
    assert (t-f).max() == (t-f).min() == (length+lag)

    return list(zip(f, t))

path = sys.argv[1]
print(f'loading from path {path}')

dates, data, features = get_iaga_data(path, load_data=False)

df = pd.DataFrame()
df['seconds'] = dates
df['dates'] = pd.to_datetime(df['seconds'], unit='s', errors='coerce')

df['index'] = range(len(df))
bucket_size = len(df)//BUCKETS
df['bucket'] = (((df['index']%bucket_size)==0) & (df['index'] > 0)).cumsum()

train, test_val = sklearn.model_selection.train_test_split(list(range(BUCKETS+1)), train_size=0.8)
test, val = sklearn.model_selection.train_test_split(test_val, train_size=0.5)

df.loc[df['bucket'].isin(train), 'split'] = 'train'
df.loc[df['bucket'].isin(test), 'split'] = 'test'
df.loc[df['bucket'].isin(val), 'split'] = 'val'

train_df = df.loc[df['split'] == 'train']
test_df = df.loc[df['split'] == 'test']
val_df = df.loc[df['split'] == 'val']

# make sure omni and iaga have the same dates
# test if it maches omni
for year in pd.unique(df['dates'].dt.year):
    print(f'testing {year}')
    omni = get_omni_data("data_local/omni/sw_data.h5", year=f'{year}')
    assert len(df.loc[df['dates'].dt.year==year]) == len(omni)
print('testing done')

train_idx = get_sequences(train_df, LENGTH, LAG)
test_idx = get_sequences(test_df, LENGTH, LAG)
val_idx = get_sequences(val_df, LENGTH, LAG)

with open('train.txt', 'w') as f: 
    json.dump({
        "length": LENGTH,
        "lag": LAG,
        "years": pd.unique(df['dates'].dt.year).tolist(),
        "idx": train_idx,
    }, f)

with open('test.txt', 'w') as f: 
    json.dump({
        "length": LENGTH,
        "lag": LAG,
        "years": pd.unique(df['dates'].dt.year).tolist(),
        "idx":test_idx
    }, f)

with open('val.txt', 'w') as f: 
    json.dump({
        "length": LENGTH,
        "lag": LAG,
        "years": pd.unique(df['dates'].dt.year).tolist(),
        "idx": val_idx
    }, f)