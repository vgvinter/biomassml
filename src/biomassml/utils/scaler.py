import torch
class TorchStandardScaler:
    def __init__(self) -> None:
        self.fitted = False 

    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)
        self.fitted = True

    def transform(self, x):
        if not self.fitted: 
            raise ValueError("Scaler has not been fitted yet")
        x -= self.mean
        x /= self.std + 1e-7
        return x

    def inverse_transform(self, x):
        if not self.fitted: 
            raise ValueError("Scaler has not been fitted yet")
        x *= self.std
        x += self.mean
        return x


class TorchMinMaxScaler:
    def __init__(self) -> None:
        self.fitted = False 

    def fit(self, x):
        self.min_val = torch.min(x, 0, keepdim=True).values
        self.max_val = torch.max(x, 0, keepdim=True).values
        self.range = (self.max_val - self.min_val) + 1e-7
        self.fitted = True

    def transform(self, x):
        if not self.fitted: 
            raise ValueError("Scaler has not been fitted yet")
        x -= self.min_val
        x /= self.range
        return x

    def inverse_transform(self, x):
        if not self.fitted: 
            raise ValueError("Scaler has not been fitted yet")
        x *= self.range
        x += self.min_val
        return x