from pytorch_lightning.callbacks.finetuning import BaseFinetuning, mulplicative
from typing import Callable, Optional, Dict, Any, List
import logging 
import torch
from torch.nn import Module, 
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl

from pytorch_lightning.utilities.exceptions import MisconfigurationException


log = logging.getLogger(__name__)

class BackboneFinetuning(BaseFinetuning):
    r"""Finetune a backbone model based on a learning rate user-defined scheduling.
    When the backbone learning rate reaches the current model learning rate
    and ``should_align`` is set to True, it will align with it for the rest of the training.
    Args:
        unfreeze_backbone_at_epoch: Epoch at which the backbone will be unfreezed.
        lambda_func: Scheduling function for increasing backbone learning rate.
        backbone_initial_ratio_lr:
            Used to scale down the backbone learning rate compared to rest of model
        backbone_initial_lr: Optional, Initial learning rate for the backbone.
            By default, we will use ``current_learning /  backbone_initial_ratio_lr``
        should_align: Whether to align with current learning rate when backbone learning
            reaches it.
        initial_denom_lr: When unfreezing the backbone, the initial learning rate will
            ``current_learning_rate /  initial_denom_lr``.
        train_bn: Whether to make Batch Normalization trainable.
        verbose: Display current learning rate for model and backbone
        rounding: Precision for displaying learning rate
    Example::
        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import BackboneFinetuning
        >>> multiplicative = lambda epoch: 1.5
        >>> backbone_finetuning = BackboneFinetuning(200, multiplicative)
        >>> trainer = Trainer(callbacks=[backbone_finetuning])
    """

    def __init__(
        self,
        unfreeze_backbone_at_epoch: int = 10,
        lambda_func: Callable = multiplicative,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: Optional[float] = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        rounding: int = 12,
        backbones: List[str] = None,
    ) -> None:
        super().__init__()

        self.unfreeze_backbone_at_epoch: int = unfreeze_backbone_at_epoch
        self.lambda_func: Callable = lambda_func
        self.backbone_initial_ratio_lr: float = backbone_initial_ratio_lr
        self.backbone_initial_lr: Optional[float] = backbone_initial_lr
        self.should_align: bool = should_align
        self.initial_denom_lr: float = initial_denom_lr
        self.train_bn: bool = train_bn
        self.verbose: bool = verbose
        self.rounding: int = rounding
        self.previous_backbone_lr: Optional[float] = None
        self.backbones = backbones

    def on_save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_backbone_lr": self.previous_backbone_lr,
        }

    def on_load_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", callback_state: Dict[int, List[Dict[str, Any]]]
    ) -> None:
        self.previous_backbone_lr = callback_state["previous_backbone_lr"]
        super().on_load_checkpoint(trainer, pl_module, callback_state["internal_optimizer_metadata"])

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Raises:
            MisconfigurationException:
                If LightningModule has no nn.Module `backbone` attribute.
        """
        check_ok = True
        for bb_name in self.backbones:
            if not hasattr(pl_module, bb_name) or not isinstance(pl_module.getattr(bb_name), Module):
                check_ok = False
        if check_ok:
            return super().on_fit_start(trainer, pl_module)
        raise MisconfigurationException("The LightningModule should have a nn.Module `backbone` attribute")

    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        for bb_name in self.backbones:
            bb = pl_module.getattr(bb_name)
            self.freeze(bb)

    def finetune_function(
        self, pl_module: "pl.LightningModule", epoch: int, optimizer: Optimizer, opt_idx: int
    ) -> None:
        """Called when the epoch begins."""
        if epoch == self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            initial_backbone_lr = (
                self.backbone_initial_lr
                if self.backbone_initial_lr is not None
                else current_lr * self.backbone_initial_ratio_lr
            )
            self.previous_backbone_lr = initial_backbone_lr
            for bb_name in self.backbones:
                bb = pl_module.getattr(bb_name)
                self.unfreeze_and_add_param_group(
                    bb,
                    optimizer,
                    initial_backbone_lr,
                    train_bn=self.train_bn,
                    initial_denom_lr=self.initial_denom_lr,
                )
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(initial_backbone_lr, self.rounding)}"
                )

        elif epoch > self.unfreeze_backbone_at_epoch:
            current_lr = optimizer.param_groups[0]["lr"]
            next_current_backbone_lr = self.lambda_func(epoch + 1) * self.previous_backbone_lr
            next_current_backbone_lr = (
                current_lr
                if (self.should_align and next_current_backbone_lr > current_lr)
                else next_current_backbone_lr
            )
            optimizer.param_groups[-1]["lr"] = next_current_backbone_lr
            self.previous_backbone_lr = next_current_backbone_lr
            if self.verbose:
                log.info(
                    f"Current lr: {round(current_lr, self.rounding)}, "
                    f"Backbone lr: {round(next_current_backbone_lr, self.rounding)}"
                )