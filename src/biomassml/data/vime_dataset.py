from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Union
from .utils import as_tt
import torch


class VIMEDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
    ):
        self.df = df.dropna(subset=feature_names)
        self.feature_names = feature_names

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        row = row[self.feature_names]

        row = torch.tensor(row.astype(np.float32))

        return row


