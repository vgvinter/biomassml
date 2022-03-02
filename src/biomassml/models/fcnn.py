from tkinter import SE
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

from typing import Sequence, Union
from biomassml.utils.conservation_law import conservation_loss


class FCNN(pl.LightningModule):
    def __init__(
        self,
        chemistry_dim: int,
        process_dim: int,
        chemistry_backbone_layers: str = "264-264",
        process_backbone_layers: str = "264-264",
        head_layers: str = "264-264",
        dropout_p: float = 0.1,
        initialization: str = "kaiming",
        activation: str = "ReLU",
        batch_norm: bool = False,
        dropconnect: bool = False,
        output_dim: int = 1,
        conservation_loss_weight: float = 0.1,
        target_names: Sequence[str] = None,
        pretrained_chemistry_backbone: Union[torch.nn.Sequential, None] = None,
        pretrained_process_backbone: Union[torch.nn.Sequential, None] = None,
        chemistry_bb_output_dim: int = None,
        process_bb_output_dim: int = None,
    ) -> None:
        super().__init__()
        self.chemistry_dim = chemistry_dim
        self.process_dim = process_dim
        self.initialization = initialization
        self.activation = activation
        self.batch_norm = batch_norm
        self.dropout_p = dropout_p
        self.dropconnect = dropconnect
        self.output_dim = output_dim
        self.chemistry_backbone_layers = chemistry_backbone_layers
        self.process_backbone_layers = process_backbone_layers
        self.head_layers = head_layers
        self.target_names = target_names
        self.pretrained_chemistry_backbone = pretrained_chemistry_backbone
        self.pretrained_process_backbone = pretrained_process_backbone
        self.chemistry_bb_output_dim = chemistry_bb_output_dim
        self.process_bb_output_dim = process_bb_output_dim
        self.conservation_loss_weight = conservation_loss_weight
        self.save_hyperparameters()

        self._build_network()

        self.lr = 1e-3
        self.loss_fn = nn.MSELoss()

    def _build_sequential(self, input_dim, in_layers, outdim=None):
        layers = []
        _curr_units = input_dim

        in_layers = in_layers.split("-")
        in_layers = [int(x) for x in in_layers if x != ""]
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

        # Todo(kjappelbaum): Remove duplicated code
        if self.pretrained_chemistry_backbone is None:
            self.chemistry_backbone, backbone_out_dim = self._build_sequential(
                self.chemistry_dim, self.chemistry_backbone_layers
            )
            self.chemistry_bb_output_dim = backbone_out_dim
        else:
            self.chemistry_backbone = self.pretrained_chemistry_backbone
            assert isinstance(self.chemistry_backbone, nn.Sequential)
            assert isinstance(
                self.chemistry_bb_output_dim, int
            ), "If you use a pretrained backbone, you must specify the output dimension of the backbone as int"
            chemistry_bb_output_dim = self.chemistry_bb_output_dim

        if self.pretrained_process_backbone is None:
            self.process_backbone, backbone_out_dim = self._build_sequential(
                self.process_dim, self.process_backbone_layers
            )
            self.process_bb_output_dim = backbone_out_dim
        else:
            self.process_backbone = self.pretrained_process_backbone
            assert isinstance(self.process_backbone, nn.Sequential)
            assert isinstance(
                self.process_bb_output_dim, int
            ), "If you use a pretrained backbone, you must specify the output dimension of the backbone as int"
            process_bb_output_dim = self.process_bb_output_dim

        if self.head_layers is not None:
            self.head, head_out_dim = self._build_sequential(
                self.process_bb_output_dim + self.chemistry_bb_output_dim, self.head_layers
            )
        else:
            self.head = nn.Sequential()
            head_out_dim = self.process_bb_output_dim + self.chemistry_bb_output_dim
        self.output_layer = nn.Linear(head_out_dim, self.output_dim)
        self.hparams.chemistry_bb_output_dim = self.chemistry_bb_output_dim
        self.hparams.process_bb_output_dim = self.process_bb_output_dim

    def compute_loss(self, y_pred, y_true):
        loss = self.loss_fn(y_pred, y_true)
        conservation_loss = 0
        if self.conservation_loss_weight:
            conservation_loss += conservation_loss(y_pred, y_true, self.target_names)
        total_loss = loss + self.conservation_loss_weight * conservation_loss

        return {"total_loss": total_loss, "loss": loss, "conservation_loss": conservation_loss}

    def forward(self, x_chemistry, x_process):
        x_chem = self.chemistry_backbone(x_chemistry)
        x_proc = self.process_backbone(x_process)
        x = torch.cat((x_chem, x_proc), dim=1)
        x = self.head(x)
        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

    def get_metrics(self, yhat, y, tag):
        mae = mean_absolute_error(yhat, y)
        mse = mean_squared_error(yhat, y)
        mape = mean_absolute_percentage_error(yhat, y)
        # Skipping R2 for now as it is not good metric for LOOCV
        # r2 = r2_score(yhat, y)
        self.log(f"{tag}_mae", mae)
        self.log(f"{tag}_mse", mse)
        self.log(f"{tag}_mape", mape)
        # self.log(f"{tag}_r2", r2)

    def process_batch(self, batch):
        x_chemistry, x_process, y = batch
        if self.trainer.datamodule.label_scaler is not None:
            y = self.trainer.datamodule.label_scaler.transform(y)

        if self.trainer.datamodule.chemistry_scaler is not None:
            x_chemistry = self.trainer.datamodule.chemistry_scaler.transform(x_chemistry)

        if self.trainer.datamodule.process_scaler is not None:
            x_process = self.trainer.datamodule.process_scaler.transform(x_process)

        return x_chemistry, x_process, y

    def training_step(self, batch, batch_idx):
        x_chemistry, x_process, y = self.process_batch(batch)
        y_hat = self.forward(x_chemistry, x_process)
        # check if this reshape makes sense
        losses = self.compute_loss(y_hat, y)

        for k, v in losses.items():
            self.log(f"train_{k}", v)
        self.get_metrics(y_hat, y, "train")
        return {"loss": losses["total_loss"]}

    def validation_step(self, batch, batch_idx):
        x_chemistry, x_process, y = self.process_batch(batch)
        y_hat = self.forward(x_chemistry, x_process)
        # check if this reshape makes sense
        losses = self.compute_loss(y_hat, y)

        for k, v in losses.items():
            self.log(f"valid_{k}", v)
        self.get_metrics(y_hat, y, "valid")

    def test_step(self, batch, batch_idx):
        x_chemistry, x_process, y = self.process_batch(batch)
        y_hat = self.forward(x_chemistry, x_process)
        # check if this reshape makes sense
        losses = self.compute_loss(y_hat, y)

        for k, v in losses.items():
            self.log(f"test_{k}", v)
        self.get_metrics(y_hat, y, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), self.lr)
        return optimizer
