import re
from datetime import datetime
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import tqdm
from scipy.special import sph_harm
from torch.utils import data
import h5py
from astropy.time import Time
from dataloader import (OMNIDataset, ShpericalHarmonicsDataset,
                        SuperMAGIAGADataset)


def persist_to_file(file_name):
    def decorator(original_func):
        try:
            cache = pickle.load(open(file_name, 'rb'))
        except (IOError, ValueError):
            cache = {}

        def new_func(param):
            if param not in cache:
                cache[param] = original_func(param)
                pickle.dump(cache, open(file_name, 'wb'))
            return cache[param]

        return new_func

    return decorator


def get_omni_data(path=None, year="2016"):
    import pandas as pd
    return pd.read_hdf(path, key=year)


def get_iaga_data(path, tiny=False):
    import glob

    import tqdm

    if tiny:
        files = sorted(
            [f for f in glob.glob(path + "supermag_iaga_tiny*.npz")],
            key=lambda f: int(re.sub("\D", "", f)),
        )
    else:
        files = sorted(
            [f for f in glob.glob(path + "supermag_iaga_[!tiny]*.npz")],
            key=lambda f: int(re.sub("\D", "", f)),
        )
    assert len(files) > 0

    data = []
    dates = []
    stations = []
    idx = []

    print("loading supermag iaga data...")
    for i, f in enumerate(tqdm.tqdm(files)):
        x = np.load(f, allow_pickle=True)
        data.append(x["data"])
        dates.append(x["dates"])
        print(np.datetime64(datetime.utcfromtimestamp(dates[-1][0])))
        idx.extend(data[-1].shape[0] * [i])
        features = x["features"]
        stations.append(x["stations"])

    max_stations = max([len(s) for s in stations])
    for i, d in enumerate(data):
        data[i] = np.concatenate(
            [d, np.zeros([d.shape[0], max_stations - d.shape[1], d.shape[2]]) * np.nan],
            axis=1,
        )
    dates = np.concatenate(dates)
    data = np.concatenate(data)
    return dates, data, features



def get_wiemer_data(targets, scaler, lag, past_omni_length, future_length):
    weimer = {}
    with h5py.File("data_local/TimeStepGeomagnetic_20150317_1min.h5", "r") as f:
        for k in f.keys():
            weimer[k] = f.get(k)[:]

    sg_data = SuperMAGIAGADataset(*get_iaga_data("data_local/iaga/2015/"))
    omni_data = OMNIDataset(get_omni_data("data_local/omni/sw_data.h5", year="2015"))
    weimer_times_unix = Time(weimer["JDTIMES"], format="jd").to_value("unix")
    wstart = np.argmin(np.abs(weimer_times_unix[0] - sg_data.dates)) - past_omni_length
    wend = (
        np.argmin(np.abs(weimer_times_unix[-1] - sg_data.dates)) + lag + future_length
    )
    weimerinds = np.arange(wstart, wend).astype(int)
    return ShpericalHarmonicsDataset(
        sg_data,
        omni_data,
        weimerinds,
        scaler=scaler,
        targets=targets,
        past_omni_length=past_omni_length,
        future_length=future_length,
        f107_dataset="data_local/f107.npz"
    )


def load_cached_data(filename, idx, scaler, supermag_data, omni_data, targets, past_omni_length, future_length):
    if os.path.exists(filename):
        data = pickle.load(open(filename, "rb"))
        return data, data.scaler
    else:
        data = ShpericalHarmonicsDataset(
            supermag_data,
            omni_data,
            idx,
            scaler=scaler,
            targets=targets,
            past_omni_length=past_omni_length,
            future_length=future_length,
            f107_dataset="data_local/f107.npz",
        )
        pickle.dump(data, open(filename, "wb"))
        return data, data.scaler