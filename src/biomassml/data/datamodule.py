from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Union, Sequence
from pathlib import Path
from .dataset import SupervisedDataset


class SupervisedDatamodule(LightningDataModule):
    def __init__(
        self,
        train_df: Union[Path, str],
        valid_df: Union[Path, str],
        test_df: Union[Path, str],
        labels: Sequence[str],
        features: Sequence[str],
        batch_size: int = 32,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
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
            dataset=SupervisedDataset(self.train_df, self.features, self.labels),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=SupervisedDataset(self.test_df, self.features, self.labels),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

