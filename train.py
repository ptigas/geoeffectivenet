import tqdm
import pickle
from utils.data_utils import get_iaga_data, get_omni_data
from dataloader import SuperMAGIAGADataset, OMNIDataset, ShpericalHarmonicsDataset
from models.spherical_harmonics import SphericalHarmonics
from torch.utils import data
import torch.optim
import os.path
from functools import partial
import pytorch_lightning as pl
from models.geoeffectivenet import *
import numpy as np
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import h5py
from astropy.time import Time

torch.set_default_dtype(torch.float64) # this is important else it will overflow

wandb_logger = WandbLogger(project='geoeffectivenet')

future_length = 1
past_omni_length = 120
nmax = 20
targets = "dbn_nez"
lag = 1
learning_rate = 1e-04
batch_size = 2048



def get_wiemer_data(targets, scaler):

    weimer = {}
    with h5py.File("data/TimeStepGeomagnetic_20150317_1min.h5", "r") as f:
        for k in f.keys():
            weimer[k] = f.get(k)[:]

    sg_data = SuperMAGIAGADataset(*get_iaga_data("data_local/iaga/2015/"))
    omni_data = OMNIDataset(get_omni_data("data_local/omni/sw_data.h5", year="2015"))
    weimer_times_unix = Time(weimer['JDTIMES'],format='jd').to_value('unix')
    wstart = np.argmin(np.abs(weimer_times_unix[0]-sg_data.dates)) - past_omni_length
    wend = np.argmin(np.abs(weimer_times_unix[-1]-sg_data.dates)) + lag + future_length
    weimerinds = np.arange(wstart,wend).astype(int)
    return ShpericalHarmonicsDataset(sg_data, omni_data, weimerinds, scaler=scaler, targets=targets ,f107_dataset="data_local/f107.npz")

    import pdb; pdb.set_trace()

if (
    not os.path.exists("cache/train_ds.p")
    or not os.path.exists("cache/test_ds.p")
    or not os.path.exists("cache/val_ds.p")
):
    sg_data = SuperMAGIAGADataset(*get_iaga_data("data_local/iaga/2013/"))
    omni_data = OMNIDataset(get_omni_data("data_local/omni/sw_data.h5", year="2013"))

    idx = list(range(len(sg_data.dates)))
    train_idx = idx[: int(len(idx) * 0.7)]
    test_val_idx = idx[int(len(idx) * 0.7) :]
    test_idx = test_val_idx[: len(test_val_idx) // 2]
    val_idx = test_val_idx[len(test_val_idx) // 2 :]
else:
    train_idx = None
    test_idx = None
    val_idx = None


def load_data(filename, idx, scaler):
    if os.path.exists(filename):
        return pickle.load(open(filename, "rb")), None
    else:
        data = ShpericalHarmonicsDataset(sg_data, omni_data, idx, scaler=scaler, targets=targets ,f107_dataset="data_local/f107.npz")
        pickle.dump(data, open(filename, "wb"))
        return data, data.scaler


overfit = False
if overfit:
    nmax = 10
    train_idx = test_idx = val_idx = train_idx[:300]
    train_ds, scaler = load_data("tiny_cache/train_ds.p", idx=train_idx, scaler=None)
    test_ds, _ = load_data("tiny_cache/test_ds.p", idx=test_idx, scaler=scaler)
    val_ds, _ = load_data("tiny_cache/val_ds.p", idx=val_idx, scaler=scaler)
else:
    train_ds, scaler = load_data("cache/train_ds.p", idx=train_idx, scaler=None)
    test_ds, _ = load_data("cache/test_ds.p", idx=test_idx, scaler=scaler)
    val_ds, _ = load_data("cache/val_ds.p", idx=val_idx, scaler=scaler)


# load weimer data for debugging
if os.path.exists("cache/wiemer_ds.p"):
    wiemer_ds = pickle.load(open("cache/wiemer_ds.p", "rb"))
else:
    wiemer_ds = get_wiemer_data(targets, scaler)
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

targets_idx = np.where(train_ds.supermag_features == targets)[0][0]

# initialize model
model = NeuralRNNWiemer(
    past_omni_length,
    future_length,
    train_ds.omni_features,
    train_ds.supermag_features,
    nmax,
    targets_idx,
)
model = model.double()

# add wiemer data to the model to debug
model.wiemer_data = wiemer_loader

checkpoint_callback = ModelCheckpoint(dirpath='checkpoints')
trainer = pl.Trainer(gpus=-1, check_val_every_n_epoch=5, logger=wandb_logger, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)
