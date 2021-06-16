import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import datetime as dt
from models.spherical_harmonics import SphericalHarmonics
import utils
from models.base import BaseModel
from dataloader import basis_matrix
from models.tcn import TemporalConvNet


class NamedAccess():
    def __init__(self, data, columns):
        self.columns = columns
        self.data = data

    def __getitem__(self, key):
        key = np.where(self.columns== key)[0][0]
        return self.data[..., key]


# class Persistent(nn.Module):
#     def __init__(self, past_omni_length, future_length, omni_features, supermag_features, nmax, targets):
#         super(Persistent, self).__init__()

#         n_coeffs = 0
#         for n in range(nmax+1):
#             for m in range(0, n+1):
#                 n_coeffs += 1
#         n_coeffs *= 2

#         self.omni_features = omni_features
#         self.supermag_features = supermag_features

#         self.targets = targets
#         self.future_length = future_length

#         self.decoder = nn.GRUCell(input_size=targets, hidden_size=64) #Was 512

#         self.g = nn.Linear(16, n_coeffs//2)
#         self.h = nn.Linear(16, n_coeffs//2)

#     def forward(self,
#                 past_omni,
#                 past_supermag,
#                 future_basis,
#                 dates,
#                 future_dates,
#                 **kargs):

#         return None, past_supermag[:, [-1], :, :].repeat([1, future_basis.shape[1], 1, 1]), None


class NeuralRNNWiemer(BaseModel):
    def __init__(self, past_omni_length, future_length, omni_features, supermag_features, nmax, targets_idx):
        super(NeuralRNNWiemer, self).__init__()

        # idx of targets in dataset
        self.targets_idx = targets_idx

        hidden = 16
        levels = 2
        kernel_size = 24
        levels = levels
        kernel_size = kernel_size
        num_channels = [hidden]*levels

        self.omni_past_encoder = nn.GRU(25,
                                        hidden,
                                        num_layers=1,
                                        bidirectional=False,
                                        batch_first=True,
                                        dropout=0.5)



        #self.omni_past_encoder = TemporalConvNet(25, num_channels, kernel_size, dropout=0.5)

        self.nmax = nmax
        self.sph = SphericalHarmonics(nmax)
        n_coeffs = len(self.sph.ylm)*2

        self.encoder_mlp = nn.Sequential(
            nn.Linear(hidden, 16),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(16, n_coeffs, bias=False) # 882
        )

        self.omni_features = omni_features
        self.supermag_features = supermag_features
        self.future_length = future_length



    def forward(self,
                past_omni,
                past_supermag,
                mlt, mcolat,
                dates,
                future_dates,
                **kargs):

        past_omni = NamedAccess(past_omni, self.omni_features)

        features = []
        # add the wiemer2013 features
        bt = (past_omni['by']**2 + past_omni['bz']**2)**.5
        v = (past_omni['vx']**2 + past_omni['vy']**2 + past_omni['vz']**2)**.5

        features.append(past_omni['bx'])
        features.append(past_omni['by'])
        features.append(past_omni['bz'])
        features.append(bt)
        features.append(v)
        features.append(past_omni['dipole'])
        features.append(torch.sqrt(past_omni['f107']))

        features.append(bt*torch.cos(past_omni['clock_angle']))
        features.append(v*torch.cos(past_omni['clock_angle']))
        features.append(past_omni['dipole']*torch.cos(past_omni['clock_angle']))
        features.append(torch.sqrt(past_omni['f107'])*torch.cos(past_omni['clock_angle']))

        features.append(bt*torch.sin(past_omni['clock_angle']))
        features.append(v*torch.sin(past_omni['clock_angle']))
        features.append(past_omni['dipole']*torch.sin(past_omni['clock_angle']))
        features.append(torch.sqrt(past_omni['f107'])*torch.sin(past_omni['clock_angle']))

        features.append(bt*torch.cos(2*past_omni['clock_angle']))
        features.append(v*torch.cos(2*past_omni['clock_angle']))
        features.append(past_omni['dipole']*torch.cos(2*past_omni['clock_angle']))
        features.append(torch.sqrt(past_omni['f107'])*torch.cos(2*past_omni['clock_angle']))

        features.append(bt*torch.sin(2*past_omni['clock_angle']))
        features.append(v*torch.sin(2*past_omni['clock_angle']))
        features.append(past_omni['dipole']*torch.sin(2*past_omni['clock_angle']))
        features.append(torch.sqrt(past_omni['f107'])*torch.sin(2*past_omni['clock_angle']))

        features.append(past_omni['clock_angle'])
        features.append(past_omni['temperature'])

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
        #future_supermag[future_supermag.isnan()] = 0.0

        assert not (torch.isnan(features).any() or torch.isinf(features).any())

        #import pdb; pdb.set_trace()
        #encoded = self.omni_past_encoder(features)[1][0]

        #encoded = self.omni_past_encoder(features.transpose(2, 1))[..., -1]
        encoded = self.omni_past_encoder(features)[1][0]

        coeffs = self.encoder_mlp(encoded)

        with torch.no_grad():
            basis = self.sph(mlt.squeeze(1), mcolat.squeeze(1))
            #basis_cpu = torch.Tensor(basis_matrix(self.nmax, theta.squeeze(1).detach().cpu().numpy(), phi.squeeze(1).detach().cpu().numpy())).cuda().double()

            # fix the zero gradients error
            basis[basis.isnan()] = 0.0
            #basis_cpu[basis_cpu.isnan()] = 0.0

        predictions = torch.einsum('bij,bj->bi', basis.squeeze(1), coeffs)


        if torch.isnan(coeffs).all():
            import pdb; pdb.set_trace()

        #coeffs_error = (basis-basis_cpu).abs()

        return basis, coeffs, predictions