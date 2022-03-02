from abc import ABC, abstractmethod
from dataclasses import dataclass
from re import I
from typing import List, Optional, Union, Sequence
from pathlib import Path

import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from loguru import logger
import pandas as pd

from biomassml.utils import TorchStandardScaler
from .utils import fit_scalers
from .dataset import SupervisedDataset


class BaseKFoldDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_folds(self) -> None:
        """Implement how folds should be initialized"""
        pass

    @abstractmethod
    def setup_fold_index(self, fold_index: int) -> None:
        """
        Given a fold index, implement how the train and validation
        dataset/dataloader should look for the current fold
        """
        pass


@dataclass
class KFoldDataModule(BaseKFoldDataModule):
    train_fold: Optional[Dataset] = None
    test_fold: Optional[Dataset] = None

    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        labels: Sequence[str],
        chemistry_features: Sequence[str],
        process_features: Sequence[str],
        stratify_column: str = None,
        num_folds: int = 5,
        loocv: bool = False,
        batch_size: int = 64,
        num_workers: int = 4,
        label_scale: bool = True,
        feature_scale: bool = True,
    ) -> None:
        super().__init__()
        self.num_folds = num_folds

        self.stratified = True if stratify_column is not None else False
        self.stratify_column = stratify_column
        self.loocv = loocv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_scale = label_scale
        self.feature_scale = feature_scale

        self.train_df = train_df
        self.valid_df = valid_df

        if loocv:
            logger.info(f"Overriding num_folds to {len(train_df)} as loocv is enabled")
            self.num_folds = len(train_df)

        features = list(chemistry_features + process_features)
        labels = list(labels)

        self.labels = labels
        self.features = features
        self.chemistry_features = chemistry_features
        self.process_features = process_features

        if label_scale:
            self.label_scaler = TorchStandardScaler()
        else:
            self.label_scaler = None
        if feature_scale:
            self.feature_scaler = TorchStandardScaler()
        else:
            self.feature_scaler = None

        if label_scale:
            self.label_scaler = TorchStandardScaler()
        else:
            self.label_scaler = None
        if feature_scale:
            self.process_scaler = TorchStandardScaler()
            self.chemistry_scaler = TorchStandardScaler()
        else:
            self.process_scaler = None
            self.chemistry_scaler = TorchStandardScaler()

    def fit_transformers(self, train_dl):
        fit_scalers(train_dl, self.chemistry_scaler, self.process_scaler, self.label_scaler)

    def setup_folds(self) -> None:
        if self.stratified:
            labels = self.df[self.stratify_column]
            if labels is None:
                raise ValueError(
                    "Tried to extract labels for stratified K folds but failed."
                    " Make sure that the dataset of your train dataloader either"
                    " has an attribute `labels` or that `label_extractor` attribute"
                    " is initialized correctly"
                )
            splitter = StratifiedKFold(self.num_folds, shuffle=self.shuffle)
        else:
            labels = None
            splitter = KFold(self.num_folds, shuffle=self.shuffle)

        self.splits = [split for split in splitter.split(range(len(self.df)), y=labels)]

    def get_dataset(self, df):
        return SupervisedDataset(df, self.chemistry_features, self.process_features, self.labels)

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, test_indices = self.splits[fold_index]
        ds = self.get_dataset(self.df)
        self.train_fold = Subset(ds, train_indices)
        self.test_fold = Subset(ds, test_indices)

    def fit_transformers(self, train_dl):
        logger.info('Fitting transformers')
        fit_scalers(train_dl, self.feature_scaler, self.label_scaler)

    def train_dataloader(self) -> DataLoader:
        train_dl = DataLoader(
            self.train_fold,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.fit_transformers(train_dl)
        return train_dl

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.get_dataset(self.valid_df),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_fold,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
