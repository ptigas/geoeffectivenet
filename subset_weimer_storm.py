import os.path
import pickle
import os
import h5py
import numpy as np
from astropy.time import Time
from torch.utils import data
import json

from utils.data_utils import get_iaga_data, get_omni_data, load_cached_data, get_wiemer_data,get_iaga_data_as_list
from utils.splitter import generate_indices
from dataloader import OMNIDataset, ShpericalHarmonicsDatasetBucketized,SuperMAGIAGADataset
#----------------------------

hyperparameter_best = dict(future_length = 1, past_omni_length = 599,
                                omni_resolution = 1, nmax = 20,lag = 1,
                                learning_rate = 5e-04,batch_size = 4096,
                                l2reg=1.6e-3,epochs = 1000, dropout_prob=0.1,n_hidden=64,
                                loss='MAE')
config = hyperparameter_best

future_length = config['future_length'] 
past_omni_length = config['past_omni_length']
omni_resolution = config['omni_resolution']
nmax = config['nmax']
targets = ["dbe_nez", "dbn_nez"] #config.targets
lag = config['lag']

yearlist = list(np.arange(2014,2016).astype(int))
supermag_data = SuperMAGIAGADataset(*get_iaga_data_as_list(base="full_data_panos/iaga/",year=yearlist))

yearlist = list(np.arange(2014,2016).astype(str))
omni_data = OMNIDataset(get_omni_data("data_local/omni/sw_data.h5", year=yearlist))

# with open("test.txt") as f:
#     test_idx = np.asarray(json.load(f)['idx'])
yearlist = list(np.arange(2014,2016).astype(int))

weimer_year = 2015
_,_,_,weimer_idx = generate_indices(base="full_data_panos/iaga/",year=yearlist,
                                                            LENGTH=past_omni_length,LAG=lag,
                                                            omni_path="data_local/omni/sw_data.h5",
                                                            weimer_path=f"data_local/weimer/{weimer_year}/")

test_idx = np.asarray(weimer_idx)
subpath = f'Subset/Weimer/{weimer_year}'
os.makedirs(subpath,exist_ok=True)

diffs = np.mean(test_idx[1:,:]-test_idx[:-1,:])
if diffs==1.0:
    start,end = test_idx[0,0],test_idx[-1,-1]+1
data = supermag_data.data[start:end,...]
dates = supermag_data.dates[start:end,...]
features = supermag_data.features
idx = test_idx-test_idx[0,0]

omnivalues = omni_data.data.values[start:end,:]
omnifeatures = omni_data.data.columns.tolist()

np.savez(f"{subpath}/supermag_omni_data.npz",data=data,dates=dates,features=features,omni=omnivalues,
omni_features=omnifeatures,idx=idx)