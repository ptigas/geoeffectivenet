import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import tqdm
from scipy.special import sph_harm
from torch.utils import data


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
