import torch
import numpy as np


def as_tt(frame):
    return torch.tensor(frame.values.astype(np.float32))


def fit_scalers(dl, chemistry_scaler, process_scaler, label_scaler):
    process_features = []
    chemistry_features = []
    ys = []

    for chemistry_x, process_x, y in dl:
        chemistry_features.append(chemistry_x)
        process_features.append(process_x)
        ys.append(y)

    process_features = torch.stack(process_features, dim=0)
    chemistry_features = torch.stack(chemistry_features, dim=0)

    ys = torch.stack(ys, dim=0)

    if process_scaler is not None:
        process_scaler.fit(process_features)

    if chemistry_scaler is not None:
        chemistry_scaler.fit(chemistry_features)

    if label_scaler is not None:
        label_scaler.fit(ys)
