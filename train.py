import os.path
import pickle

import h5py
import numpy as np
import pytorch_lightning as pl
import torch.optim
import tqdm
from astropy.time import Time
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils import data
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.geoeffectivenet import *
from models.spherical_harmonics import SphericalHarmonics
from utils.data_utils import get_iaga_data, get_omni_data, load_cached_data, get_wiemer_data
from dataloader import (OMNIDataset, ShpericalHarmonicsDataset,
                        SuperMAGIAGADataset)

torch.set_default_dtype(torch.float64)  # this is important else it will overflow

wandb_logger = WandbLogger(project="geoeffectivenet", log_model=True)

future_length = 1
past_omni_length = 600 # minutes in the past
omni_resolution = 10 # 10 minutes
nmax = 20
targets = ["dbe_nez", "dbn_nez"]
lag = 1
learning_rate = 1e-04
batch_size = 256*5


if (
    not os.path.exists("cache/train_ds.p")
    or not os.path.exists("cache/test_ds.p")
    or not os.path.exists("cache/val_ds.p")
):
    supermag_data = SuperMAGIAGADataset(*get_iaga_data("data_local/iaga/2013/"))
    omni_data = OMNIDataset(get_omni_data("data_local/omni/sw_data.h5", year="2013"))

    idx = list(range(len(supermag_data.dates)))
    train_idx = idx[: int(len(idx) * 0.7)]
    test_val_idx = idx[int(len(idx) * 0.7) :]
    test_idx = test_val_idx[: len(test_val_idx) // 2]
    val_idx = test_val_idx[len(test_val_idx) // 2 :]
else:
    train_idx = None
    test_idx = None
    val_idx = None
    supermag_data = None
    omni_data = None


overfit = False
if overfit:
    nmax = 10
    train_idx = test_idx = val_idx = train_idx[:300]
    train_ds, scaler = load_cached_data("tiny_cache/train_ds.p", train_idx, None, supermag_data, omni_data, targets, past_omni_length, future_length)
    test_ds, _ = load_cached_data("tiny_cache/test_ds.p", test_idx, scaler, supermag_data, omni_data, targets, past_omni_length, future_length)
    val_ds, _ = load_cached_data("tiny_cache/val_ds.p", val_idx, scaler, supermag_data, omni_data, targets, past_omni_length, future_length)
else:
    train_ds, scaler = load_cached_data("cache/train_ds.p", train_idx, None, supermag_data, omni_data, targets, past_omni_length, future_length)
    test_ds, _ = load_cached_data("cache/test_ds.p", test_idx, scaler, supermag_data, omni_data, targets, past_omni_length, future_length)
    val_ds, _ = load_cached_data("cache/val_ds.p", val_idx, scaler, supermag_data, omni_data, targets, past_omni_length, future_length)

# load weimer data for debugging
if os.path.exists("cache/wiemer_ds.p"):
    wiemer_ds = pickle.load(open("cache/wiemer_ds.p", "rb"))
else:
    wiemer_ds = get_wiemer_data(targets, scaler, lag, past_omni_length, future_length)
    pickle.dump(wiemer_ds, open("cache/wiemer_ds.p", "wb"))

wiemer_loader = data.DataLoader(
    wiemer_ds, batch_size=batch_size, shuffle=False, num_workers=8
)
train_loader = data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=False, num_workers=8
)
val_loader = data.DataLoader(
    val_ds, batch_size=batch_size, shuffle=False, num_workers=8
)
plot_loader = data.DataLoader(val_ds, batch_size=4, shuffle=False)

targets_idx = [np.where(train_ds.supermag_features == target)[0][0] for target in targets]

# initialize model
model = NeuralRNNWiemer(
    past_omni_length,
    future_length,
    train_ds.omni_features,
    train_ds.supermag_features,
    omni_resolution,
    nmax,
    targets_idx,
)
model = model.double()

# add wiemer data to the model to debug
model.wiemer_data = wiemer_loader
model.scaler = scaler

# save the scaler to de-standarize prediction
pickle.dump(scaler, open('checkpoints/scalers.p', "wb"))

checkpoint_callback = ModelCheckpoint(dirpath="checkpoints")
trainer = pl.Trainer(
    gpus=-1,
    check_val_every_n_epoch=5,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, EarlyStopping(monitor='val_MSE')]
)
trainer.fit(model, train_loader, val_loader)
