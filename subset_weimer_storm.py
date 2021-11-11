import os.path
import pickle
import os
from glob import glob
import h5py
import numpy as np
from astropy.time import Time
from torch.utils import data
import json
import pandas as pd

from utils.data_utils import get_iaga_data, get_omni_data, load_cached_data, get_wiemer_data,get_iaga_data_as_list
from utils.splitter import generate_indices
from dataloader import OMNIDataset, ShpericalHarmonicsDatasetBucketized,SuperMAGIAGADataset
#----------------------------

hyperparameter_best = dict(future_length = 1, past_omni_length = 120,
                                omni_resolution = 1, nmax = 20,lag = 30,
                                learning_rate = 5e-04,batch_size = 8500,
                                l2reg=5e-5,epochs = 1000, dropout_prob=0.7,n_hidden=8,
                                loss='MAE')
config = hyperparameter_best

future_length = config['future_length'] 
past_omni_length = config['past_omni_length']
omni_resolution = config['omni_resolution']
nmax = config['nmax']
targets = ["dbe_nez", "dbn_nez"] #config.targets
lag = config['lag']

yearlist = list(np.arange(2010,2012).astype(int))
supermag_data = SuperMAGIAGADataset(*get_iaga_data_as_list(base="full_data_panos/iaga/",year=yearlist))

yearlist = list(np.arange(2010,2012).astype(str))
omni_data = OMNIDataset(get_omni_data("data_local/omni/sw_data.h5", year=yearlist))

f107 = np.load('data_local/f107.npz')
f107_data,f107_dates = f107["f107"],f107["dates"]
weimer_year = 2011

subpath = f'Subset/Weimer/{weimer_year}'
os.makedirs(subpath,exist_ok=True)

print("Generating start and end times of Weimer nowcast....")
weimer = {}
with h5py.File(glob(f"data_local/weimer/{weimer_year}/*.h5")[0], "r") as f:
    for k in f.keys():
        weimer[k] = f.get(k)[:]

weimer_times_unix = Time(weimer['JDTIMES'],format='jd').to_value('unix')
N_weimer_data = len(weimer_times_unix)
start = np.argmin(np.abs(supermag_data.dates-weimer_times_unix[0]))
end = np.argmin(np.abs(supermag_data.dates-weimer_times_unix[-1]))+1

#Generate the correct indices
id_end = np.linspace(start,end,N_weimer_data,endpoint=False).astype(int)
id_start = id_end-past_omni_length-lag

data = supermag_data.data[id_start[0]:end,...]
dates = supermag_data.dates[id_start[0]:end,...]
features = supermag_data.features
idx = np.concatenate([id_start[:,None],id_end[:,None]],axis=-1)-id_start[0]

omnivalues = omni_data.data.values[id_start[0]:end,:]
omnifeatures = omni_data.data.columns.tolist()

np.savez(f"{subpath}/supermag_omni_data.npz",data=data,dates=dates,features=features,omni=omnivalues,
omni_features=omnifeatures,idx=idx)

#Generate the subset of f107 index.
tmp_dates = pd.to_datetime(np.array([dates[0],dates[-1]]).reshape(-1),unit='s').to_numpy().reshape([-1,1])
#Find the best matching f10.7 index along 2nd dimension
match = np.argmin(np.abs(tmp_dates-f107_dates.reshape([1,-1])),axis=-1)

np.savez(f"{subpath}/f107.npz",f107=f107_data[match[0]:match[-1]],dates=f107_dates[match[0]:match[-1]])