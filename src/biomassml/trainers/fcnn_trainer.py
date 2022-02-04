"""Trainer for crystal graph convolutional neural networks."""
from subprocess import call
import time

import wandb

from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    StochasticWeightAveraging,
    ModelSummary,
    EarlyStopping,
)
from pytorch_lightning.loggers import WandbLogger
from torchsummary import summary
from biomassml.data.datamodule import SupervisedDatamodule
from biomassml.models.fcnn import FCNN

from .utils import log_hyperparameters
import numpy as np


def train(config: DictConfig):
    if config.get("seed") is not None:
        seed_everything(config.seed, workers=True)
    else:
        seed_everything(np.random.randint(0, 1000), workers=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    logger = WandbLogger(
        project=config.project_name,
        entity=config.entity,
        tags=config.tags,
        log_model=config.log_model,
    )

    datamodule: SupervisedDatamodule = instantiate(config.data)

    outname = f"{timestr}_emgl"

    model: FCNN = instantiate(
        config.model,
        _convert_="partial",
        continuous_dim=len(config.data.features),
    )

    # summary(model, input_size=(1,len(config.data.features)))

    callbacks = []
    checkpointer = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        monitor="valid_loss",
        verbose=True,
        dirpath=outname,
        every_n_val_epochs=1,
    )
    callbacks.append(checkpointer)
    callbacks.append(ModelSummary(max_depth=-1))
    callbacks.append(CustomWriter("predictions", write_interval=5))
    if config.swa:
        callbacks.append(StochasticWeightAveraging())

    if config.patience:
        if config.patience > 0:
            callbacks.append(EarlyStopping(monitor="valid_loss", patience=config.patience))

    # Initialize a trainer

    trainer: Trainer = instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    logger.watch(model)

    log_hyperparameters(config, model, trainer)
    if config.trainer.auto_lr_find:
        trainer.tune(model, datamodule)

    # Train the model ⚡
    trainer.fit(model, datamodule=datamodule)

    # Test
    trainer.test(model, datamodule=datamodule)

    wandb.finish()
