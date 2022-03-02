class TorchStandardScaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean
        x /= self.std + 1e-7
        return x

    def inverse_transform(self, x):
        x *= self.std
        x += self.mean
        return x
