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
from biomassml.data.datamodule import SupervisedDatamodule
from biomassml.models.fcnn import FCNN
from biomassml.models.vime import VIMEModel

from .utils import log_hyperparameters
import numpy as np
import torch
from pytorch_lightning.callbacks import BackboneFinetuning


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
        offline=False,
    )

    datamodule: SupervisedDatamodule = instantiate(config.data)

    outname = f"{timestr}_emgl"

    chemistry_backbone = None
    process_backbone = None
    chemistry_bb_output_dim = None
    process_bb_output_dim = None 

    if config.model.pretrained_chemistry_backbone is not None:
        d = torch.load(config.model.pretrained_chemistry_backbone)
        chemistry_backbone = VIMEModel.load_from_checkpoint(config.model.pretrained_chemistry_backbone).backbone
        chemistry_bb_output_dim = d["hyper_parameters"]["bb_output_dim"]

    if config.model.pretrained_process_backbone is not None:
        d = torch.load(config.model.pretrained_process_backbone)
        process_backbone = VIMEModel.load_from_checkpoint(config.model.pretrained_process_backbone).backbone
        process_bb_output_dim = d["hyper_parameters"]["bb_output_dim"]

    model: FCNN = instantiate(
        config.model,
        _convert_="partial",
        chemistry_dim=len(config.data.chemistry_features),
        process_dim=len(config.data.process_features),
        output_dim=len(config.data.labels),
        target_names=config.data.labels,
        pretrained_chemistry_backbone=chemistry_backbone,
        pretrained_process_backbone=process_backbone,
        chemistry_bb_output_dim=chemistry_bb_output_dim,
        process_bb_output_dim=process_bb_output_dim
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
    if config.swa:
        callbacks.append(StochasticWeightAveraging())

    if config.patience:
        if config.patience > 0:
            callbacks.append(EarlyStopping(monitor="valid_loss", patience=config.patience))

    if "backbone_finetuning" in config:
        bb_finetuning = BackboneFinetuning(**config.backbone_finetuning)
        callbacks.append(bb_finetuning)
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
