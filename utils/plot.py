import io

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.metrics import r2_score

import PIL.Image
import torch
from matplotlib import cycler
from torchvision.transforms import ToTensor
import sys
from utils.helpers import basis_matrix

def spherical_plot_forecasting(nmax, coeffs, predictions, target):
    plt.style.use("default")
    plt.rcParams.update({
        "lines.linewidth":
        1.0,
        "axes.grid":
        False,
        "grid.linestyle":
        ":",
        "axes.grid.axis":
        "both",
        "axes.prop_cycle":
        cycler('color', [
            '0071bc', 'd85218', 'ecb01f', '7d2e8d', '76ab2f', '4cbded',
            'a1132e'
        ]),
        "xtick.top":
        True,
        "xtick.minor.size":
        0,
        "xtick.direction":
        "in",
        "xtick.minor.visible":
        True,
        "ytick.right":
        True,
        "ytick.minor.size":
        0,
        "ytick.direction":
        "in",
        "ytick.minor.visible":
        True,
        "legend.framealpha":
        1.0,
        "legend.edgecolor":
        "white",
        "legend.fancybox":
        False,
        "figure.figsize": (12, 12),
        "figure.autolayout":
        False,
        "savefig.dpi":
        300,
        "savefig.pad_inches":
        0.01,
        "savefig.transparent":
        True
    })

    shape_spherical= (45, 360)
    grid_phi_spherical   = (np.arange(shape_spherical[0])+0.5)/shape_spherical[0]*np.pi/4     #colat
    grid_phi_spherical   = grid_phi_spherical.reshape(shape_spherical[0],1)@np.ones((1,shape_spherical[1]),dtype=np.float)
    grid_theta_spherical = (np.arange(shape_spherical[1])+0.5)/shape_spherical[1]*2.0*np.pi #longitude
    grid_theta_spherical = np.ones((shape_spherical[0],1),dtype=np.float)@grid_theta_spherical.reshape(1,shape_spherical[1])

    basis_grid = basis_matrix(nmax, grid_theta_spherical.ravel().reshape(-1), grid_phi_spherical.ravel().reshape(-1))
    basis_grid = torch.Tensor(basis_grid).double().squeeze(0).cuda()

    COLORS = cm.get_cmap('viridis')

    grid_predictions = torch.einsum('bj,ij->bi', coeffs, basis_grid).detach().cpu().numpy()
    grid_predictions = grid_predictions.reshape(-1, * grid_theta_spherical.shape)

    i = 0

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.pcolormesh(
        grid_theta_spherical,
        grid_phi_spherical,
        grid_predictions[i], cmap='coolwarm', shading='auto')
    ax.set_title("Prediction: ")
    ax.set_theta_offset(-np.pi/2)

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    return ToTensor()(image)
