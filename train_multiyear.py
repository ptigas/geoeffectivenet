import os.path
import pickle

import h5py
import numpy as np
import pytorch_lightning as pl
import torch.optim
import wandb
from astropy.time import Time
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils import data
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from models.geoeffectivenet import *
from models.spherical_harmonics import SphericalHarmonics
from utils.data_utils import get_iaga_data_as_list, get_omni_data, load_cached_data, get_wiemer_data,get_weimer_data_indices
from dataloader import (OMNIDataset, ShpericalHarmonicsDataset,
                        SuperMAGIAGADataset)

torch.set_default_dtype(torch.float64)  # this is important else it will overflow

hyperparameter_best = dict(future_length = 1, past_omni_length = 240,
                                omni_resolution = 1, nmax = 20,lag = 1,
                                learning_rate = 5e-04,batch_size = 4096,
                                l2reg=1.6e-3,epochs = 1000, dropout_prob=0.1,n_hidden=64,
                                loss='MAE')
                                # learning_rate originally 1e-5

hyperparameter_defaults = hyperparameter_best
wandb.init(config=hyperparameter_defaults)
config = wandb.config
wandb.run.name = "MAE_2015_SMAG_Reg"

#----- Data loading also depends on the sweep parameters.
#----- Hence this process will be repeated per training cycle.
def train(config):
    # future_length = config['future_length'] 
    # past_omni_length = config['past_omni_length']
    # omni_resolution = config['omni_resolution']
    # nmax = config['nmax']
    # targets = ["dbe_nez", "dbn_nez"] #config.targets
    # lag = config['lag']
    # learning_rate = config['learning_rate']
    # batch_size = config['batch_size']
    # l2reg=config['l2reg']
    # max_epochs = config['epochs']
    # n_hidden=config['n_hidden']
    # dropout_prob=config['dropout_prob']
    # loss = config['loss']
    future_length = config.future_length 
    past_omni_length = config.past_omni_length
    omni_resolution = config.omni_resolution
    nmax = config.nmax
    targets = ["dbe_nez", "dbn_nez"] #config.targets
    lag = config.lag
    learning_rate = config.learning_rate
    batch_size = config.batch_size
    l2reg=config.l2reg
    max_epochs = config.epochs
    n_hidden=config.n_hidden
    dropout_prob=config.dropout_prob
    loss = config.loss
    
    train_yearlist = ['2015']
    storm_yearlist = ['2015']
    if (
        not os.path.exists("cache/train_ds.p")
        or not os.path.exists("cache/test_ds.p")
        or not os.path.exists("cache/val_ds.p")
    ):
        supermag_data = SuperMAGIAGADataset(*get_iaga_data_as_list(train_yearlist))
        omni_data = OMNIDataset(get_omni_data("data_local/omni/sw_data.h5", year=train_yearlist))
        weimer_indices = get_weimer_data_indices(targets, lag, past_omni_length, future_length,supermag_data,storm_yearlist)

        idx = list(range(len(supermag_data.dates)))
        idx = [i for i in idx if i not in weimer_indices]
        train_idx = idx[: int(len(idx) * 0.85)]
        val_idx = idx[int(len(idx) * 0.85) :]
    else:
        train_idx = None
        test_idx = None
        val_idx = None
        supermag_data = None
        omni_data = None

    train_ds, scaler = load_cached_data("cache/train_ds.p", train_idx, None, supermag_data, omni_data, targets, past_omni_length, future_length)
    val_ds, _ = load_cached_data("cache/val_ds.p", val_idx, scaler, supermag_data, omni_data, targets, past_omni_length, future_length)

    # load weimer data for debugging
    if os.path.exists("cache/wiemer_ds.p"):
        wiemer_ds = pickle.load(open("cache/wiemer_ds.p", "rb"))
    else:
        wiemer_ds = get_wiemer_data(targets, scaler, lag, past_omni_length, future_length,storm_yearlist)
        pickle.dump(wiemer_ds, open("cache/wiemer_ds.p", "wb"))
    wiemer_loader = {}
    for k in wiemer_ds.keys():
        wiemer_loader[k] = data.DataLoader(
            wiemer_ds[k], batch_size=batch_size, shuffle=False, num_workers=8
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
    model = NeuralRNNWiemer_HidddenSuperMAG(
        past_omni_length,
        future_length,
        train_ds.omni_features,
        train_ds.supermag_features,
        omni_resolution,
        nmax,
        targets_idx,learning_rate = learning_rate,
        l2reg=l2reg,
        dropout_prob=dropout_prob,
        n_hidden=n_hidden,
        loss=loss
    )
    model = model.double()

    # add wiemer data to the model to debug
    model.wiemer_data = wiemer_loader
    model.scaler = scaler

    # save the scaler to de-standarize prediction
    checkpoint_path = "tmp"
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    pickle.dump(scaler, open(f'{checkpoint_path}/scalers.p', "wb"))

    wandb_logger = WandbLogger(project="geoeffectivenet", log_model=True)
    wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path)
    if torch.cuda.is_available():
        trainer = pl.Trainer(
        gpus=-1,
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, EarlyStopping(monitor='val_MSE',patience = 100)]
    )
    else:
        trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, EarlyStopping(monitor='val_MSE',patience = 100)]
    )
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':

    print(f'Starting a run with {hyperparameter_defaults}')
    train(config)
