import io

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

import wandb
from utils.helpers import R2
from utils.plot import spherical_plot_forecasting


# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = PIL.Image.open(buf)
    return ToTensor()(image)


class BaseModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, training_batch, batch_idx):
        (
            past_omni,
            past_supermag,
            future_supermag,
            past_dates,
            future_dates,
            (phi, theta),
        ) = training_batch

        _, coeffs, predictions = self(
            past_omni, past_supermag, phi, theta, past_dates, future_dates
        )

        predictions[torch.isnan(predictions)] = 0
        future_supermag[torch.isnan(future_supermag)] = 0
        target_col = self.targets_idx
        future_supermag = future_supermag[..., target_col].squeeze(1)
        loss = ((future_supermag - predictions) ** 2).mean()

        # sparsity L1
        loss += 1e-3 * torch.norm(coeffs, p=1)

        self.log("train_MSE", loss, on_step=False, on_epoch=True)
        self.log(
            "train_r2",
            R2(future_supermag, predictions).mean(),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        (
            past_omni,
            past_supermag,
            future_supermag,
            past_dates,
            future_dates,
            (mlt, mcolat),
        ) = val_batch

        _, coeffs, predictions = self(
            past_omni, past_supermag, mlt, mcolat, past_dates, future_dates
        )

        predictions[torch.isnan(predictions)] = 0
        future_supermag[torch.isnan(future_supermag)] = 0
        target_col = self.targets_idx

        future_supermag = future_supermag[..., target_col].squeeze(1)
        loss = ((future_supermag - predictions) ** 2).mean()

        self.log(
            "val_R2",
            R2(future_supermag, predictions).mean(),
            on_step=False,
            on_epoch=True,
        )
        self.log("val_MSE", loss, on_step=False, on_epoch=True)

        if batch_idx == 0:
            # hack: need to find a callback way
            predictions = []
            coeffs = []
            targets = []
            for (
                past_omni,
                past_supermag,
                future_supermag,
                past_dates,
                future_dates,
                (mlt, mcolat),
            ) in self.wiemer_data:
                past_omni = past_omni.cuda()
                past_supermag = past_supermag.cuda()
                mlt = mlt.cuda()
                mcolat = mcolat.cuda()
                past_dates = past_dates.cuda()
                future_dates = future_dates.cuda()
                _, _coeffs, pred = self(
                    past_omni, past_supermag, mlt, mcolat, past_dates, future_dates
                )
                predictions.append(pred.cuda())
                coeffs.append(_coeffs.cuda())
                targets.append(future_supermag[..., target_col].cuda())
            predictions = torch.cat(predictions).detach()
            coeffs = torch.cat(coeffs).detach()
            targets = torch.cat(targets).detach().squeeze(1)

            predictions[torch.isnan(predictions)] = 0
            targets[torch.isnan(targets)] = 0

            self.log(
                "wiemer_R2",
                R2(targets, predictions).mean(),
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "wiemer_MSE",
                ((targets - predictions) ** 2).mean(),
                on_step=False,
                on_epoch=True,
            )

            fig, ax = plt.subplots()

            ax.scatter(targets.cpu().numpy().ravel(), predictions.cpu().numpy().ravel())
            self.logger.experiment.log(
                {
                    "wiemer_scatter": [
                        wandb.Image(get_img_from_fig(fig), caption="val_scatter")
                    ]
                }
            )
            nice_idx = [500]
            pred_sphere = spherical_plot_forecasting(
                self.nmax, coeffs[nice_idx], predictions[nice_idx], targets[nice_idx]
            )
            self.logger.experiment.log(
                {"pred_sphere": [wandb.Image(pred_sphere, caption="pred_sphere")]}
            )

            plt.figure().clear()
            plt.close()
            plt.cla()
            plt.clf()
