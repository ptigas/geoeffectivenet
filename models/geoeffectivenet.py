
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from dataloader import basis_matrix
from models.base import BaseModel
from models.spherical_harmonics import SphericalHarmonics
from models.tcn import TemporalConvNet


class NamedAccess:
    def __init__(self, data, columns):
        self.columns = columns
        self.data = data

    def __getitem__(self, key):
        key = np.where(self.columns == key)[0][0]
        return self.data[..., key]


class NeuralRNNWiemer(BaseModel):
    def __init__(
        self,
        past_omni_length,
        future_length,
        omni_features,
        supermag_features,
        omni_resolution,
        nmax,
        targets_idx,**kwargs
    ):
        super(NeuralRNNWiemer, self).__init__(**kwargs)

        # idx of targets in dataset
        self.targets_idx = targets_idx

        self.omni_resolution = omni_resolution

        hidden = kwargs.pop('n_hidden',16)
        dropout_prob = kwargs.pop('dropout',0.5)
        levels = 2
        kernel_size = 24
        levels = levels
        kernel_size = kernel_size
        [hidden] * levels

        self.omni_past_encoder = nn.GRU(
            25, hidden, num_layers=1, bidirectional=False, batch_first=True
        )

        # self.omni_past_encoder = TemporalConvNet(25, num_channels, kernel_size, dropout=0.5)

        self.nmax = nmax
        self.sph = SphericalHarmonics(nmax)
        n_coeffs = len(self.sph.ylm) * 2

        self.encoder_mlp = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(16, n_coeffs*len(targets_idx), bias=False),  # 882
        )

        self.omni_features = omni_features
        self.supermag_features = supermag_features
        self.future_length = future_length

    def forward(
        self, past_omni, past_supermag, mlt, mcolat, dates, future_dates, **kargs
    ):
        # 10 mins average
        past_omni = nn.AvgPool1d(self.omni_resolution)(past_omni.permute([0, 2, 1])).permute([0, 2, 1])

        past_omni = NamedAccess(past_omni, self.omni_features)

        features = []
        # add the wiemer2013 features
        bt = (past_omni["by"] ** 2 + past_omni["bz"] ** 2) ** 0.5
        v = (past_omni["vx"] ** 2 + past_omni["vy"] ** 2 + past_omni["vz"] ** 2) ** 0.5

        features.append(past_omni["bx"])
        features.append(past_omni["by"])
        features.append(past_omni["bz"])
        features.append(bt)
        features.append(v)
        features.append(past_omni["dipole"])
        features.append(torch.sqrt(past_omni["f107"]))

        features.append(bt * torch.cos(past_omni["clock_angle"]))
        features.append(v * torch.cos(past_omni["clock_angle"]))
        features.append(past_omni["dipole"] * torch.cos(past_omni["clock_angle"]))
        features.append(
            torch.sqrt(past_omni["f107"]) * torch.cos(past_omni["clock_angle"])
        )

        features.append(bt * torch.sin(past_omni["clock_angle"]))
        features.append(v * torch.sin(past_omni["clock_angle"]))
        features.append(past_omni["dipole"] * torch.sin(past_omni["clock_angle"]))
        features.append(
            torch.sqrt(past_omni["f107"]) * torch.sin(past_omni["clock_angle"])
        )

        features.append(bt * torch.cos(2 * past_omni["clock_angle"]))
        features.append(v * torch.cos(2 * past_omni["clock_angle"]))
        features.append(past_omni["dipole"] * torch.cos(2 * past_omni["clock_angle"]))
        features.append(
            torch.sqrt(past_omni["f107"]) * torch.cos(2 * past_omni["clock_angle"])
        )

        features.append(bt * torch.sin(2 * past_omni["clock_angle"]))
        features.append(v * torch.sin(2 * past_omni["clock_angle"]))
        features.append(past_omni["dipole"] * torch.sin(2 * past_omni["clock_angle"]))
        features.append(
            torch.sqrt(past_omni["f107"]) * torch.sin(2 * past_omni["clock_angle"])
        )

        features.append(past_omni["clock_angle"])
        features.append(past_omni["temperature"])

        # PI = 22.0/7.0
        # offset = (dt.datetime(2013,1,1) - dt.datetime(1970,1,1)).total_seconds()/(365*24*60*60)
        # sin_year = torch.sin(2*PI*dates - offset)
        # cos_year = torch.cos(2*PI*dates - offset)

        # features.append(sin_year)
        # features.append(cos_year)

        # hours =  (pd.to_datetime(past_omni['timestamp'], unit='s').hour*60 + pd.to_datetime(data[:, self.date_col], unit='s').minute).values
        # sin_day = np.sin(2*torch.pi*hours/(24*60))
        # cos_day = np.cos(2*torch.pi*hours/(24*60))

        features = torch.stack(features, -1)

        # zero fill
        features[features.isnan()] = 0.0

        # fix the zero gradients error
        # future_supermag[future_supermag.isnan()] = 0.0

        assert not (torch.isnan(features).any() or torch.isinf(features).any())

        # import pdb; pdb.set_trace()
        # encoded = self.omni_past_encoder(features)[1][0]

        # encoded = self.omni_past_encoder(features.transpose(2, 1))[..., -1]
        encoded = self.omni_past_encoder(features)[1][0]

        coeffs = self.encoder_mlp(encoded).reshape(encoded.shape[0], -1, len(self.targets_idx))

        with torch.no_grad():
            basis = self.sph(mlt.squeeze(1), mcolat.squeeze(1))
            # basis_cpu = torch.Tensor(basis_matrix(self.nmax, theta.squeeze(1).detach().cpu().numpy(), phi.squeeze(1).detach().cpu().numpy())).cuda().double()

            # fix the zero gradients error
            basis[basis.isnan()] = 0.0
            # basis_cpu[basis_cpu.isnan()] = 0.0

        predictions = torch.einsum("bij,bjk->bik", basis.squeeze(1), coeffs)

        if torch.isnan(coeffs).all():
            import pdb
            pdb.set_trace()

        return basis, coeffs, predictions

 
class NeuralRNNWiemer_HidddenSuperMAG(BaseModel):
    def __init__(
        self,
        past_omni_length,
        future_length,
        omni_features,
        supermag_features,
        omni_resolution,
        nmax,
        targets_idx,**kwargs
    ):
        super(NeuralRNNWiemer_HidddenSuperMAG, self).__init__(**kwargs)

        # idx of targets in dataset
        self.targets_idx = targets_idx

        self.omni_resolution = omni_resolution

        hidden = 12 #kwargs.pop('n_hidden',16)
        dropout_prob = kwargs.pop('dropout',0.5)
        levels = 2
        kernel_size = 24
        levels = levels
        kernel_size = kernel_size
        [hidden] * levels

        self.omni_past_encoder = nn.GRU(
            25, hidden, num_layers=1, bidirectional=False, batch_first=True
        )

        # self.omni_past_encoder = TemporalConvNet(25, num_channels, kernel_size, dropout=0.5)

        self.nmax = nmax
        self.sph = SphericalHarmonics(nmax)
        n_coeffs = len(self.sph.ylm) * 2

        self.encoder_mlp = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(16, n_coeffs*len(targets_idx), bias=False),  # 882
        )

        self.omni_features = omni_features
        self.supermag_features = supermag_features
        self.future_length = future_length

    def forward(
        self, past_omni, past_supermag, mlt, mcolat, dates, future_dates, **kargs
    ):
        # 10 mins average
        past_omni = nn.AvgPool1d(self.omni_resolution)(past_omni.permute([0, 2, 1])).permute([0, 2, 1])

        past_omni = NamedAccess(past_omni, self.omni_features)

        features = []
        # add the wiemer2013 features
        bt = (past_omni["by"] ** 2 + past_omni["bz"] ** 2) ** 0.5
        v = (past_omni["vx"] ** 2 + past_omni["vy"] ** 2 + past_omni["vz"] ** 2) ** 0.5

        features.append(past_omni["bx"])
        features.append(past_omni["by"])
        features.append(past_omni["bz"])
        features.append(bt)
        features.append(v)
        features.append(past_omni["dipole"])
        features.append(torch.sqrt(past_omni["f107"]))

        features.append(bt * torch.cos(past_omni["clock_angle"]))
        features.append(v * torch.cos(past_omni["clock_angle"]))
        features.append(past_omni["dipole"] * torch.cos(past_omni["clock_angle"]))
        features.append(
            torch.sqrt(past_omni["f107"]) * torch.cos(past_omni["clock_angle"])
        )

        features.append(bt * torch.sin(past_omni["clock_angle"]))
        features.append(v * torch.sin(past_omni["clock_angle"]))
        features.append(past_omni["dipole"] * torch.sin(past_omni["clock_angle"]))
        features.append(
            torch.sqrt(past_omni["f107"]) * torch.sin(past_omni["clock_angle"])
        )

        features.append(bt * torch.cos(2 * past_omni["clock_angle"]))
        features.append(v * torch.cos(2 * past_omni["clock_angle"]))
        features.append(past_omni["dipole"] * torch.cos(2 * past_omni["clock_angle"]))
        features.append(
            torch.sqrt(past_omni["f107"]) * torch.cos(2 * past_omni["clock_angle"])
        )

        features.append(bt * torch.sin(2 * past_omni["clock_angle"]))
        features.append(v * torch.sin(2 * past_omni["clock_angle"]))
        features.append(past_omni["dipole"] * torch.sin(2 * past_omni["clock_angle"]))
        features.append(
            torch.sqrt(past_omni["f107"]) * torch.sin(2 * past_omni["clock_angle"])
        )

        features.append(past_omni["clock_angle"])
        features.append(past_omni["temperature"])

        # PI = 22.0/7.0
        # offset = (dt.datetime(2013,1,1) - dt.datetime(1970,1,1)).total_seconds()/(365*24*60*60)
        # sin_year = torch.sin(2*PI*dates - offset)
        # cos_year = torch.cos(2*PI*dates - offset)

        # features.append(sin_year)
        # features.append(cos_year)

        # hours =  (pd.to_datetime(past_omni['timestamp'], unit='s').hour*60 + pd.to_datetime(data[:, self.date_col], unit='s').minute).values
        # sin_day = np.sin(2*torch.pi*hours/(24*60))
        # cos_day = np.cos(2*torch.pi*hours/(24*60))

        features = torch.stack(features, -1)

        # zero fill
        features[features.isnan()] = 0.0
        init_state  = past_supermag[:,:,2:4]

        #Leverage nan!=nan
        inds = ~init_state.isnan()
        mean = torch.nansum(init_state,dim=-2)/torch.sum(inds,dim=-2)
        var = torch.nansum(torch.square(init_state - mean.reshape([-1,1,2])),dim=-2)/torch.sum(inds,dim=-2)

        init_summary = [mean.reshape(-1,2)]
        init_summary.append(var.reshape(-1,2))
        init_summary.append(torch.nanquantile(init_state,1.0,dim=-2).reshape(-1,2))
        init_summary.append(torch.nanquantile(init_state,0.0,dim=-2).reshape(-1,2))
        init_summary.append(torch.nanquantile(init_state,0.25,dim=-2).reshape(-1,2))
        init_summary.append(torch.nanquantile(init_state,0.75,dim=-2).reshape(-1,2))
        init_summary = torch.stack(init_summary, -1).reshape([1,-1,12])

        # fix the zero gradients error
        # future_supermag[future_supermag.isnan()] = 0.0

        assert not (torch.isnan(features).any() or torch.isinf(features).any())

        # import pdb; pdb.set_trace()
        # encoded = self.omni_past_encoder(features)[1][0]

        # encoded = self.omni_past_encoder(features.transpose(2, 1))[..., -1]
        encoded = self.omni_past_encoder(features,init_summary)[1][0]

        coeffs = self.encoder_mlp(encoded).reshape(encoded.shape[0], -1, len(self.targets_idx))

        with torch.no_grad():
            basis = self.sph(mlt.squeeze(1), mcolat.squeeze(1))
            # basis_cpu = torch.Tensor(basis_matrix(self.nmax, theta.squeeze(1).detach().cpu().numpy(), phi.squeeze(1).detach().cpu().numpy())).cuda().double()

            # fix the zero gradients error
            basis[basis.isnan()] = 0.0
            # basis_cpu[basis_cpu.isnan()] = 0.0

        predictions = torch.einsum("bij,bjk->bik", basis.squeeze(1), coeffs)

        if torch.isnan(coeffs).all():
            import pdb
            pdb.set_trace()

        return basis, coeffs, predictions       
