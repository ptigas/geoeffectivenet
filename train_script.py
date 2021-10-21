import os.path
import pickle
import pandas as pd
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
from utils.data_utils import get_iaga_data, get_omni_data, load_cached_data,get_wiemer_data,get_iaga_data_as_list
from utils.splitter import generate_indices
from dataloader import OMNIDataset, ShpericalHarmonicsDatasetBucketized,SuperMAGIAGADataset
# #-----------------------------------
# import argparse
# parser = argparse.ArgumentParser(description = 'GeoeffectiveNET hyperparameter tuning!!!')
# parser.add_argument('future_length',default = 1 ,type=int,help = 'future_length')
# parser.add_argument('past_omni_length',default = 600 ,type=int,help = 'past_omni_length')
# parser.add_argument('omni_resolution',default = 10,type=int,help = 'omni_resolution')
# parser.add_argument('nmax',default = 20,type=int,help = 'nmax modes')
# parser.add_argument('lag',default = 1,type=int,help = 'lag')
# parser.add_argument('learning_rate',default = 1e-4,type=int,help = 'learning_rate')
# parser.add_argument('batch_size',default = 256*5,type=int,help = 'batch_size')

# args = parser.parse_args()
# #-----------------------------------

torch.set_default_dtype(torch.float64)  # this is important else it will overflow

# hyperparameter_defaults = dict(future_length = 1, past_omni_length = 900,
#                                 omni_resolution = 1, nmax = 25,lag = 1,
#                                 learning_rate = 1e-04,batch_size = 256*8*2,
#                                 l2reg=3e-3,epochs = 10000, dropout_prob=0.71,n_hidden=64,
#                                 loss='MSE')


hyperparameter_best = dict(future_length = 1, past_omni_length = 120,
                                omni_resolution = 1, nmax = 20,lag = 1,
                                learning_rate = 5e-04,batch_size = 1024*8,
                                l2reg=1.6e-5,epochs = 1000, dropout_prob=0.1,n_hidden=64,
                                loss='MAE')
                                # learning_rate originally 1e-5

hyperparameter_defaults = hyperparameter_best

wandb.init(config=hyperparameter_defaults)
config = wandb.config

#----- Data loading also depends on the sweep parameters.
#----- Hence this process will be repeated per training cycle.
def train(config):
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

    wandb.run.name = f"FULL_{loss}_{past_omni_length}_{nmax}_{n_hidden}_{learning_rate*1e6}_{l2reg*1e6}"

    yearlist = list(np.arange(2010,2019).astype(int))
    supermag_data = SuperMAGIAGADataset(*get_iaga_data_as_list(base="full_data_panos/iaga/",year=yearlist))
    yearlist = list(np.arange(2010,2019).astype(str))
    omni_data = OMNIDataset(get_omni_data("data_local/omni/sw_data.h5", year=yearlist))

    yearlist = list(np.arange(2010,2019).astype(int))
    train_idx,test_idx,val_idx,wiemer_idx = generate_indices(base="full_data_panos/iaga/",year=yearlist,
                                                        LENGTH=past_omni_length,LAG=lag,
                                                        omni_path="data_local/omni/sw_data.h5",
                                                        weimer_path="data_local/weimer/")
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)
    wiemer_idx = np.asarray(wiemer_idx)

    train_ds = ShpericalHarmonicsDatasetBucketized(supermag_data,omni_data,train_idx,
            f107_dataset="data_local/f107.npz",targets=targets,past_omni_length=past_omni_length,
            past_supermag_length=1,future_length=future_length,lag=lag,zero_omni=False,
            zero_supermag=False,scaler=None,training_batch=True,nmax=nmax)
    print("Train dataloader defined....")
    val_ds = ShpericalHarmonicsDatasetBucketized(supermag_data,omni_data,val_idx,
            f107_dataset="data_local/f107.npz",targets=targets,past_omni_length=past_omni_length,
            past_supermag_length=1,future_length=future_length,lag=lag,zero_omni=False,
            zero_supermag=False,scaler=train_ds.scaler,training_batch=False,nmax=nmax)
    print("Val dataloader defined....")
    test_ds = ShpericalHarmonicsDatasetBucketized(supermag_data,omni_data,test_idx,
            f107_dataset="data_local/f107.npz",targets=targets,past_omni_length=past_omni_length,
            past_supermag_length=1,future_length=future_length,lag=lag,zero_omni=False,
            zero_supermag=False,scaler=train_ds.scaler,training_batch=False,nmax=nmax)
    print("Test dataloader defined....")
    wiemer_ds = ShpericalHarmonicsDatasetBucketized(supermag_data,omni_data,wiemer_idx,
            f107_dataset="data_local/f107.npz",targets=targets,past_omni_length=past_omni_length,
            past_supermag_length=1,future_length=future_length,lag=lag,zero_omni=False,
            zero_supermag=False,scaler=train_ds.scaler,training_batch=False,nmax=nmax)
    print("Weimer dataloader defined....")

    #Save the scaler
    scaler = train_ds.scaler

    wiemer_loader = data.DataLoader(
        wiemer_ds, batch_size=batch_size, shuffle=False, num_workers=12
    )
    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=12
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=12
    )
    test_loader = data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=12
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
        targets_idx,learning_rate = learning_rate,
        l2reg=l2reg,
        dropout_prob=dropout_prob,
        n_hidden=n_hidden,
        loss=loss
    )
    model = model.double()

    # add wiemer data to the model to debug
    model.wiemer_data = wiemer_loader
    model.test_data = test_loader
    model.scaler = scaler

    # save the scaler to de-standarize prediction
    # checkpoint_path = f"checkpoints_{int(learning_rate*1e5)}_{int(batch_size)}_{int(l2reg*1e6)}_{nmax}_{loss}"
    checkpoint_path = wandb.run.name
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

    print(f'Starting a run with {config}')
    train(config)
