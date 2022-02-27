from typing import Union, List
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from .vime_dataset import VIMEDataset
from sklearn.model_selection import train_test_split
from pathlib import Path

class VIMEDataModule(pl.LightningDataModule):
    def __init__(
        self,
      df: Union[Path, str],
        features: List[str],
        train_size: float = 0.8,
        test_size: float = 0.5,
        batch_size: int = 64,
        prefetch_factor: int = 2,
        num_workers: int = 6,
    ):
        super().__init__()
        df = pd.read_csv(df)

        self.train_df, test = train_test_split(
            df,
            train_size=train_size,
        )
        self.valid_df, self.test_df = train_test_split(test, train_size=test_size)

        self.feature_names = features
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers



    def get_loader(self, df, batch_size, num_workers, prefetch_factor, shuffle=True):

        ds = VIMEDataset(
            df,
            feature_names=self.feature_names,
        )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )

    def train_dataloader(self):
        return self.get_loader(
            pd.concat([self.train_df, self.test_df]),
            self.batch_size,
            self.num_workers,
            self.prefetch_factor,
        )

    def val_dataloader(self):
        return self.get_loader(
            self.valid_df,
            self.batch_size,
            self.num_workers,
            self.prefetch_factor,
            shuffle=False,
        )

    def test_dataloader(self):
        return self.get_loader(
            self.test_df,
            self.batch_size,
            self.num_workers,
            self.prefetch_factor,
            shuffle=False,
        )
