from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from scipy.special import sph_harm
from sklearn.preprocessing import StandardScaler
from torch.utils import data

from utils.helpers import dipole_tilt


class NamedAccess:
    def __init__(self, data, columns):
        self.columns = columns
        self.data = data

    def __getitem__(self, key):
        key = np.where(self.columns == key)[0][0]
        return self.data[..., key]


def columns_to_indices(all_columns, columns):
    return [np.where(all_columns == col)[0][0] for col in columns]


def stack_from_array(data, columns, col_anm_shp=None, dt_loc=None):
    if col_anm_shp is None:
        return data[:, columns]
    else:
        for col in columns:
            tmp = np.stack(data[:, col])
            if tmp.shape[-1] != col_anm_shp:
                tmp = tmp.reshape([-1, 1])
            try:
                final = np.concatenate((final, tmp), axis=-1)
            except:
                final = tmp
        return final.reshape([final.shape[0], -1])


def _float(tensor):
    return torch.from_numpy(tensor.astype(np.float64)).double()


class SuperMAGIAGADataset:
    def __init__(self, dates, data, features):
        self.dates = dates
        self.data = data
        self.features = features


class OMNIDataset:
    def __init__(self, data):
        self.data = data


def basis_matrix(nmax, theta, phi):
    from scipy.special import sph_harm

    assert len(theta) == len(phi)
    basis = []
    for n in range(nmax + 1):
        # for m in range(-n, n+1):
        for m in range(-n, n + 1):  # for m in range(0, n+1):
            y_mn = sph_harm(m, n, theta, phi)
            basis.append(y_mn.real.ravel())
            basis.append(y_mn.imag.ravel())
    basis = np.array(basis)
    return (
        basis.reshape(-1, theta.shape[0], theta.shape[1]).swapaxes(0, 1).swapaxes(2, 1)
    )


class ShpericalHarmonicsDataset(data.Dataset):
    def __init__(
        self,
        supermag_data,
        omni_data,
        idx,
        f107_dataset,
        targets="dbn_nez",
        past_omni_length=120,
        past_supermag_length=10,
        future_length=10,
        lag=0,
        zero_omni=False,
        zero_supermag=False,
        scaler=None,
        training_batch=True,
        nmax=20
    ):

        self.dates = supermag_data.dates[idx]

        self.supermag_data = supermag_data.data[idx]
        self.supermag_features = supermag_data.features

        self.target_idx = []
        for target in targets:
            self.target_idx.append(np.where(self.supermag_features == target)[0][0])

        self.omni = omni_data.data.iloc[idx].to_numpy()

        print("extracting f107")
        f107_data = np.load(f107_dataset)
        f107 = []
        for date in tqdm.tqdm(self.dates):
            match = np.argmin(np.abs(f107_data["dates"]- np.datetime64(
                datetime.utcfromtimestamp(date).replace(hour=0, minute=0)
            )))
            # match = np.where(mask)[0][0]
            val = f107_data["f107"][match]
            f107.append(val)
        f107 = np.array(f107)

        # add dipole
        self.omni = np.concatenate(
            [self.omni, dipole_tilt(self.dates).reshape(-1, 1)], 1
        )
        self.omni = np.concatenate([self.omni, f107.reshape(-1, 1)], 1)
        omni_columns = np.array(omni_data.data.columns.tolist() + ["dipole", "f107"])

        self.omni_features = omni_columns

        assert len(self.dates) == len(self.omni)
        assert len(self.dates) == len(self.supermag_data)

        self.targets = targets

        self.past_omni_length = past_omni_length
        self.past_supermag_length = past_supermag_length
        self.future_length = future_length
        self.lag = lag
        self.zero_omni = zero_omni
        self.zero_supermag = zero_supermag
        self.training_batch = training_batch

        if scaler is not None:
            print("using existing scaler")
            self.omni = scaler["omni"].transform(self.omni)
            target = self.supermag_data[..., self.target_idx]
            target_mean, target_std = scaler["supermag"]
            self.supermag_data[..., self.target_idx] = (target-target_mean)/target_std
            self.scaler = scaler
        else:
            self.scaler = {}
            print("learning scaler")
            self.scaler["omni"] = StandardScaler()
            target = self.supermag_data[...,self.target_idx]
            target_mean = np.nanmean(np.nanmean(target, 0), 0)
            target_std = np.nanstd(np.nanstd(target, 0), 0)
            self.scaler["supermag"] = [target_mean, target_std]
            self.omni = self.scaler["omni"].fit_transform(self.omni)
            self.supermag_data[...,self.target_idx] = (target-target_mean)/target_std

        self._nbasis = nmax

    def __len__(self):
        return (
            len(self.dates)
            - max(self.past_omni_length, self.past_supermag_length)
            - self.future_length
            - self.lag
            + 1
        )

    def __getitem__(self, index):
        index = index + max(self.past_omni_length, self.past_supermag_length)

        # average N minutes of the future
        if self.future_length > 1:
            future_supermag = torch.Tensor(
                self.supermag_data[
                    index + self.lag : index + self.future_length + self.lag
                ]
            )
            future_supermag = (
                torch.nn.AvgPool1d(self.future_length)(
                    future_supermag.permute([1, 2, 0])
                )[..., 0]
                .unsqueeze(0)
                .numpy()
            )
        else:
            future_supermag = self.supermag_data[
                index + self.lag : index + self.future_length + self.lag
            ]

        sm_future = NamedAccess(future_supermag, self.supermag_features)
        sm_future["MLT"]
        sm_future["MAGLAT"]

        _mlt = 90.0 - sm_future["MLT"] / 24.0 * 360.0
        _mcolat = 90.0 - sm_future["MAGLAT"]

        return (
            self.omni[index - self.past_omni_length : index],
            self.supermag_data[index - self.past_supermag_length : index],
            future_supermag,
            self.dates[index - self.past_omni_length : index],
            self.dates[index + self.lag : index + self.future_length + self.lag],
            (np.deg2rad(_mlt), np.deg2rad(_mcolat)),
        )
