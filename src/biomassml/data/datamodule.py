from re import T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Union, Sequence
from pathlib import Path
from sklearn.model_selection import train_test_split
from .dataset import SupervisedDataset
import pandas as pd
from biomassml.utils.scaler import TorchStandardScaler
from .utils import fit_scalers
from loguru import logger

class SupervisedDatamodule(LightningDataModule):
    def __init__(
        self,
        df: Union[Path, str],
        labels: Sequence[str],
        chemistry_features: Sequence[str],
        process_features: Sequence[str],
        train_size: float = 0.8,
        test_size: float = 0.5,
        stratify_column: str = None,
        batch_size: int = 64,
        num_workers: int = 4,
        label_scale: bool = True,
        feature_scale: bool = True,
    ) -> None:
        super().__init__()
        df = pd.read_csv(df)
        strat = df[stratify_column] if stratify_column is not None else None
        train_df, test = train_test_split(df, train_size=train_size, stratify=strat)
        strat = test[stratify_column] if stratify_column is not None else None
        valid_df, test_df = train_test_split(test, train_size=test_size, stratify=strat)
        features = list(chemistry_features + process_features)
        labels = list(labels)

        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        self.labels = labels
        self.features = features
        self.chemistry_features = chemistry_features
        self.process_features = process_features
        self.batch_size = batch_size
        self.num_workers = num_workers

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
        logger.info('Fitting transformers')
        fit_scalers(train_dl, self.feature_scaler, self.label_scaler)

    def train_dataloader(self) -> DataLoader:
        self.fit_transformers()
        train_dl = DataLoader(
            dataset=SupervisedDataset(
                self.train_df, self.chemistry_features, self.process_features, self.labels
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.fit_transformers(train_dl)
        return train_dl

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=SupervisedDataset(
                self.valid_df, self.chemistry_features, self.process_features, self.labels
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=SupervisedDataset(
                self.test_df, self.chemistry_features, self.process_features, self.labels
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
