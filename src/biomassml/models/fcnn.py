from turtle import back
import pytorch_lightning as pl
import torch.nn as nn
from .utils import _linear_dropout_bn, patch_module
import torch
from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

import matplotlib.pyplot as plt
from mpml.utils import plot_map
import pandas as pd
from typing import Union


class FCNN(pl.LightningModule):
    def __init__(
        self,
        continuous_dim: int,
        backbone_layers: str = "264-264",
        head_layers: str = "264-264",
        dropout_p: float = 0.1,
        initialization: str = "kaiming",
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropconnect: bool = False,
        output_dim: int = 1,
        target_names: list = None,
        pretrained_backbone: Union[torch.nn.Sequential, None] = None,
        bb_output_dim: int = None,
    ) -> None:
        super().__init__()
        self.continuous_dim = continuous_dim
        self.initialization = initialization
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p
        self.dropconnect = dropconnect
        self.output_dim = output_dim
        self.backbone_layers = backbone_layers
        self.head_layers = head_layers
        self.target_names = target_names
        self.pretrained_backbone = pretrained_backbone
        self.bb_output_dim = bb_output_dim
        self.save_hyperparameters()

        self._build_network()

        self.lr = 1e-3
        self.loss_fn = nn.MSELoss(reduction="sum")

    def _build_sequential(self, input_dim, in_layers, outdim=None):
        layers = []
        _curr_units = input_dim

        in_layers = in_layers.split("-")
        if outdim is not None:
            in_layers.append(outdim)
        for i, units in enumerate(in_layers):
            layers.extend(
                _linear_dropout_bn(
                    self.activation,
                    self.initialization,
                    self.batch_norm,
                    _curr_units,
                    int(units),
                    self.dropout_p,
                    self.dropconnect,
                    False,  # self.hparams.bayesian if i == in_layers else None
                )
            )
            _curr_units = int(units)
        model = nn.Sequential(*layers)

        return model, _curr_units

    def _build_network(self):
        # Linear Layers

        if self.pretrained_backbone is None:
            self.backbone, backbone_out_dim = self._build_sequential(
                self.continuous_dim, self.backbone_layers
            )
            self.hparams.bb_output_dim = backbone_out_dim
        else:
            self.backbone = self.pretrained_backbone
            assert isinstance(self.backbone, nn.Sequential)
            assert isinstance(
                self.bb_output_dim, int
            ), "If you use a pretrained backbone, you must specify the output dimension of the backbone as int"
            backbone_out_dim = self.bb_output_dim

        if self.head_layers is not None:
            self.head, head_out_dim = self._build_sequential(backbone_out_dim, self.head_layers)
        else:
            self.head = nn.Sequential()
            head_out_dim = backbone_out_dim
        self.output_layer = nn.Linear(head_out_dim, self.output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

    def get_metrics(self, yhat, y, tag):
        mae = mean_absolute_error(yhat, y)
        mse = mean_squared_error(yhat, y)
        mape = mean_absolute_percentage_error(yhat, y)
        r2 = r2_score(yhat, y)
        self.log(f"{tag}_mae", mae)
        self.log(f"{tag}_mse", mse)
        self.log(f"{tag}_mape", mape)
        self.log(f"{tag}_r2", r2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        # check if this reshape makes sense
        loss = (
            self.loss_fn(y_hat, y)
        )
        self.log("train_loss", loss)
        self.get_metrics(y_hat, y, "train")
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        self.get_metrics(y_hat, y, "valid")
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)

        for i in range(len(y)):
            fig, ax = plt.subplots(2, 1)
            plot_map(pd.Series(y[i, :], self.target_names), self.target_names, ax[0])

            plot_map(
                pd.Series(y_hat[i, :].detach().numpy(), self.target_names), self.target_names, ax[1]
            )

            fig.savefig(f"prediction_{i}_{batch_idx}.png")
        self.get_metrics(y_hat, y, "test")
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
