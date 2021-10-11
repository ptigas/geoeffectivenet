import os.path
import pickle

import h5py
import numpy as np
import pytorch_lightning as pl
import torch.optim
from astropy.time import Time
from torch.utils import data
import json

# from models.geoeffectivenet import *
from models.spherical_harmonics import SphericalHarmonics
from utils.data_utils import get_iaga_data, get_omni_data, load_cached_data, get_wiemer_data,get_iaga_data_as_list
from dataloader import OMNIDataset, ShpericalHarmonicsDatasetBucketized,SuperMAGIAGADataset
#----------------------------

torch.set_default_dtype(torch.float64)  # this is important else it will overflow

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
learning_rate = config['learning_rate']
batch_size = config['batch_size']
l2reg=config['l2reg']
max_epochs = config['epochs']
n_hidden=config['n_hidden']
dropout_prob=config['dropout_prob']
loss = config['loss']

yearlist = list(np.arange(2010,2019).astype(int))
supermag_data = SuperMAGIAGADataset(*get_iaga_data_as_list(base="full_data_panos/iaga/",year=yearlist))

yearlist = list(np.arange(2010,2019).astype(str))
omni_data = OMNIDataset(get_omni_data("data_local/omni/sw_data.h5", year=yearlist))

with open("test.txt") as f:
    test_idx = np.asarray(json.load(f)['idx'])

test_idx = test_idx[:100]
test_ds = ShpericalHarmonicsDatasetBucketized(supermag_data,omni_data,test_idx,
            f107_dataset="data_local/f107.npz",targets=targets,past_omni_length=past_omni_length,
            past_supermag_length=1,future_length=future_length,lag=lag,zero_omni=False,
            zero_supermag=False,scaler=None,training_batch=True,nmax=nmax)
vals = test_ds[10]