from tkinter import Y
import pytorch_lightning as pl
import torch.nn as nn
from torchmetrics import R2Score
from .utils import _linear_dropout_bn, patch_module
import torch
import shap
from torchmetrics.functional import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
import numpy as np


def enable_dropout(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


class FCNN(pl.LightningModule):
    def __init__(
        self,
        continuous_dim: int,
        layers: str = "264-264",
        dropout_p: float = 0.1,
        initialization: str = "kaiming",
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropconnect: bool = False,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.continuous_dim = continuous_dim
        self.initialization = initialization
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p
        self.dropconnect = dropconnect
        self.output_dim = output_dim
        self.layers = layers

        self._build_network()

    def _build_network(self):
        # Linear Layers
        layers = []
        _curr_units = self.continuous_dim

        in_layers = self.layers.split("-")
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
        self.backbone = nn.Sequential(*layers)
        if self.dropconnect:
            patch_module(
                self.backbone,
                ["Linear"],
                weight_dropout=self.dropout_p,
                inplace=True,
            )
        self.bb_output_dim = _curr_units
        self.output_layer = nn.Linear(self.bb_output_dim, self.output_dim)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = self.backbone(x)
        x = self.output_layer(x)
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
        loss = self.loss_fn(y_hat, y)
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
        self.get_metrics(y_hat, y, "test")
        self.log("test_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
