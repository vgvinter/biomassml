from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from typing import List, Union
from .utils import as_tt


class SupervisedDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, feature_names: List[str], label_names: List[str]):
        self.df = pd.read_csv(dataframe)
        self.df.dropna(subset=feature_names+label_names, inplace=True)
        self.label_names = label_names
        self.feature_names = feature_names

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        return as_tt(row[self.feature_names]), as_tt(row[self.label_names])
