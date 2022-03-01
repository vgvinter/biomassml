from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from typing import List, Union
from .utils import as_tt


class SupervisedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        chemistry_features: List[str],
        process_features: List[str],
        label_names: List[str],
    ):
        self.df = df.dropna(subset=chemistry_features + process_features + label_names)
        self.label_names = label_names
        self.chemistry_features = chemistry_features
        self.process_features = process_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        return (
            as_tt(row[self.chemistry_features]),
            as_tt(row[self.process_features]),
            as_tt(row[self.label_names]),
        )
