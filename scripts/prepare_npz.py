import glob
import json
import os

import numpy as np
import pandas as pd
import tqdm

import dask
import dask.dataframe as dd

import datetime as dt
import sys

year = sys.argv[1]

OMNI_PATH  = '/scratch/pankas/FDL/sw_data.h5'
SUPERMAG_PATH = f'/scratch/pankas/FDL/supermag_preprocessed_{year}/'

print("Loading OMNI data from " + OMNI_PATH)
sw_df = pd.read_hdf(OMNI_PATH, key=year)

print("Loading csv files from " + SUPERMAG_PATH)
files = list(glob.glob(SUPERMAG_PATH + '*.csv.gz'))

odf = dd.read_csv(files,
                    dtype={
                        'dt': 'float64',
                    },
                    compression='gzip',
                    blocksize=None)

print("Index by Date_UTC")
odf = odf.set_index('Date_UTC').compute()

print("Filtering and pre-processing")
# Keep only auroral
odf = odf[odf['MAGLAT'] >= 40]

NBINS_PER_MLT = 4
NBINS_PER_MAGLAT = 2

# Fixing MLT resolution to match OVATION Prime
odf.MLT = odf.MLT.replace(24, 0) # 24h = 0h
# 4's just because we're using integers.
# odf['MLTI'] = (((odf['MLT']*NBINS_PER_MLT).astype('uint8')))/NBINS_PER_MLT
# odf['MAGLAT'] = (((odf['MAGLAT']*NBINS_PER_MAGLAT).astype('uint8')))/NBINS_PER_MAGLAT

# clean up
odf.drop(columns=['MAGON', 'year', 'month', 'day',
                 'hour', 'prev_t', 'dt', 'LatBands', 'MLT_INT', 'date'],
        inplace=True)

odf['month'] = pd.to_datetime(odf.index, unit='s').month
odf['year'] = pd.to_datetime(odf.index, unit='s').year

sw_df['Date_UTC'] = (sw_df.index-dt.datetime(1970, 1, 1)).total_seconds()

for month in range(12):

    if os.path.isfile('/home/ptigkas/data/iaga/{year}/supermag_iaga_{year}_{month}.npz'):
        continue

    month += 1
    print(f"computing month {month}")

    df = odf[odf['month'] == month]

    min_date = df.index.min()
    max_date = df.index.max()

    # Keep omni for which supermag exists
    sw_df2 = sw_df[(sw_df['Date_UTC'] >= min_date) & (sw_df['Date_UTC'] <= max_date)]

    print("Pivoting table")
    dates = sw_df2.Date_UTC.unique()
    #MAGLATs = df.MAGLAT.unique()
    #MAGLATs = np.arange(60, 90, 1/NBINS_PER_MAGLAT)
    #MLTIs = df.MLTI.unique()

    stations = sorted(df.IAGA.unique())
    # st_map = {}
    # for i, station in enumerate(stations):
    #     st_map[station] = i

    features = ['dbe_nez', 'dbn_nez', 'ddbe_dt', 'ddbn_dt', 'MAGLAT', 'MLT']
    for f in features:
        df[f].astype('float32', copy=False)

    #tmp = pd.MultiIndex.from_product([MAGLATs, MLTIs], names=['MAGLAT', 'MLTI']).to_frame().drop(columns=['MAGLAT', 'MLTI']).reset_index()
    #df = pd.concat([df, tmp])
    #df = pd.concat([df,  pd.MultiIndex.from_product([MAGLATs, MLTIs], names=['MAGLAT', 'MLTI']).to_frame().drop(columns=['MAGLAT', 'MLTI']).reset_index()])
    df.index.name = 'Date_UTC' # the line above destroyed the index name
    sw_df2.set_index('Date_UTC', inplace=True)
    # np.count_nonzero(~np.isnan(df2.values))


    df2 = pd.pivot_table(df, values=features, index=['Date_UTC'], columns=['IAGA'], dropna=False)

    # df2 = pd.pivot_table(df,
    #                     values=features,
    #                     index=['Date_UTC'],
    #                     columns=['MAGLAT', 'MLTI'],
    #                     dropna=False,
    #                     aggfunc=lambda x: x if len(x) <= 1 else x.values[np.abs(x).argmin()])

    #df2 = df2[df2.index > (len(MAGLATs)*len(MLTIs)+1)]
    #data = df2.values.reshape(len(df.dates.unique()), len(MAGLATs), len(MLTIs), len(features))

    df2  = sw_df2.join(df2)
    df2.drop(columns=list(set(sw_df2.columns)), inplace=True)
    data =  df2.values.reshape(len(dates), len(features), len(df.IAGA.unique())).swapaxes(2, 1)

    assert np.count_nonzero(~np.isnan(df2.values)) != 0

    output_path = f'/scratch/pankas/FDL/iaga/{year}'
    os.makedirs(output_path, exist_ok=True)

    np.savez(f'{output_path}/supermag_iaga_{year}_{month}.npz',
                                            data=data,
                                            dates=dates,
                                            columns=df2.columns,
                                            stations=sorted(df.IAGA.unique()),
                                            features=sorted(features))

    np.savez(f'{output_path}/supermag_iaga_tiny_{year}_{month}.npz',
                                            data=data[:1000],
                                            dates=dates,
                                            columns=df2.columns,
                                            stations=sorted(df.IAGA.unique()),
                                            features=sorted(features))
