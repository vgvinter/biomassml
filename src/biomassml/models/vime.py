import pytorch_lightning as pl
import torch.nn as nn
import torch
from .utils import _linear_dropout_bn, patch_module
from torch import Tensor
import random


def generate_mask(*size: int, mask_prob: float) -> Tensor:
    """
    Args:
        *size (int): Shape of the expected values.
        mask_prob (float): The probability of masking.
    Returns:
        Tensor: If True, these features should be masked.
            Otherwise, these features should be not changed.
    Examples:
        >>> generate_mask(4, mask_prob=0.5)
        tensor([0, 1, 1, 0])
    """
    mask = (torch.rand(*size) > mask_prob).long()
    return mask

def masking(x: Tensor, x_mask: Tensor, mask: Tensor) -> Tensor:
    """
    Args:
        x (Tensor):
        x_mask (Tensor):
        mask (Tensor):
    Returns:
        Tensor: Masked deatures.
        .. math:: x * mask + x_mask * (1 - mask).
    Examples:
        >>> import torch
        >>> x = torch.tensor([[1, 2, 3]])
        >>> x_mask = torch.tensor([[0, 0, 0]])
        >>> mask = torch.tensor([[0.5, 0.5, 0.5]])
        >>> masking(x, x_mask, mask)
        tensor([[0.5000, 1.0000, 1.5000]])
    """
    if x.dim() == 3:
        mask = mask.unsqueeze(-1)
    elif x.dim() > 3:
        raise ValueError(f"{x.dim()}D tensor is invalid for masking")
    return x * mask + x_mask * (1 - mask)

class VIMEModel(pl.LightningModule):
    def __init__(
        self,
        continuous_dim,
        initialization,
        activation,
        batch_norm,
        dropout_p,
        dropconnect,
        output_dim,
        layers,
        head_1_layers,
        head_2_layers,
        prob: float =0.3,
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
        self.head_1_layers = head_1_layers
        self.head_2_layers = head_2_layers

        self.scale_crossentropy_loss = False
        self.lr = 1e-4
        self.prob = prob

        self._build_network()

        self.save_hyperparameters()

        self.mse_loss = nn.MSELoss()
        self.crossentropy_loss = nn.BCEWithLogitsLoss()

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
        self.backbone, backbone_out_dim = self._build_sequential(self.continuous_dim, self.layers)
        self.hparams.bb_output_dim = backbone_out_dim
        self.task1_head, _ = self._build_sequential(
            backbone_out_dim, self.head_1_layers, self.output_dim
        )
        self.task2_head, _ = self._build_sequential(
            backbone_out_dim, self.head_2_layers, self.output_dim
        )

        if self.dropconnect:
            for module in [self.backbone, self.task1_head, self.task2_head]:
                patch_module(
                    module,
                    ["Linear"],
                    weight_dropout=self.dropout_p,
                    inplace=True,
                )

    def get_predictions(self, x):
        encoding = self.backbone(x)
        task1_preds = self.task1_head(encoding)
        task2_preds = self.task2_head(encoding)

        return encoding, task1_preds, task2_preds

    def compute_loss(self, task1_preds, task2_preds, rows, mask):
        mse_loss = self.mse_loss(task1_preds, rows)
        cross_entropy_loss = self.crossentropy_loss(task2_preds, mask)

        if self.scale_crossentropy_loss:
            cross_entropy_loss = cross_entropy_loss / mask.shape[1]
        total_loss = mse_loss + cross_entropy_loss

        return mse_loss, cross_entropy_loss, total_loss

    def random_idx(self, size: int):
        return torch.randint(0, size, (size,))

    def augment_and_get_mask(self, y):
        x_shuffled = []
        for i in range(self.continuous_dim):
            x_shuffled.append(y[self.random_idx(len(y)), i])
        x_shuffled = torch.stack(x_shuffled, 1)


        mask = generate_mask(len(y), self.continuous_dim, mask_prob=self.prob)
        x = masking(y, x_shuffled, mask)

        return x, mask.float()

    def training_step(self, batch, batch_idx):
        y = batch
        x, mask = self.augment_and_get_mask(y)
        encoding, task1_preds, task2_preds = self.get_predictions(x)
        mse_loss, cross_entropy_loss, total_loss = self.compute_loss(
            task1_preds, task2_preds, y, mask
        )

        self.log("train_mse_loss", mse_loss, prog_bar=True)
        self.log("train_cross_entropy_loss", cross_entropy_loss, prog_bar=True)
        self.log("train_total_loss", total_loss, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        y = batch
        x, mask = self.augment_and_get_mask(y)
        encoding, task1_preds, task2_preds = self.get_predictions(x)
        mse_loss, cross_entropy_loss, total_loss = self.compute_loss(
            task1_preds, task2_preds, y, mask
        )

        self.log("valid_mse_loss", mse_loss, prog_bar=True)
        self.log("valid_cross_entropy_loss", cross_entropy_loss, prog_bar=True)
        self.log("valid_total_loss", total_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        y = batch
        x, mask = self.augment_and_get_mask(y)
        encoding, task1_preds, task2_preds = self.get_predictions(x)
        mse_loss, cross_entropy_loss, total_loss = self.compute_loss(
            task1_preds, task2_preds, y, mask
        )

        self.log("test_mse_loss", mse_loss, prog_bar=True)
        self.log("test_cross_entropy_loss", cross_entropy_loss, prog_bar=True)
        self.log("test_total_loss", total_loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
