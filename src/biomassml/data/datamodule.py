from re import T
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Union, Sequence
from pathlib import Path
from sklearn.model_selection import train_test_split
from .dataset import SupervisedDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

class SupervisedDatamodule(LightningDataModule):
    def __init__(
        self,
        df: Union[Path, str],
        labels: Sequence[str],
        features: Sequence[str],
        train_size: float = 0.8,
        test_size: float = 0.5,
        stratify_column: str = None,
        batch_size: int = 64,
        num_workers: int = 4,
        scale: bool = True,
    ) -> None:
        super().__init__()
        df = pd.read_csv(df)
        strat = df[stratify_column] if stratify_column is not None else None
        train_df, test = train_test_split(
            df, train_size=train_size, stratify=strat
        )
        strat = test[stratify_column] if stratify_column is not None else None
        valid_df, test_df = train_test_split(test, train_size=test_size, stratify=strat)
        features = list(features)
        labels = list(labels)
        if scale: 
            self.scaler = StandardScaler()
      
            self.scaler.fit(train_df[features])

       
            train_df[features] = self.scaler.fit_transform(train_df[features])
            valid_df[features] = self.scaler.transform(valid_df[features])
            test_df[features] = self.scaler.transform(test_df[features])
        else: 
            self.scaler = None
        
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df

        self.labels = labels
        self.features = features
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=SupervisedDataset(self.train_df, self.features, self.labels),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def monitor_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=SupervisedDataset(self.train_df, self.features, self.labels),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=SupervisedDataset(self.valid_df, self.features, self.labels),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=SupervisedDataset(self.test_df, self.features, self.labels),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
