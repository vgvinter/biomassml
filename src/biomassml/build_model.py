import GPy
import numpy as np
from .coregionalized_regressor import GPCoregionalizedRegression

__all__ = [
    "get_matern_32_kernel",
    "get_matern_52_kernel",
    "get_ratquad_kernel",
    "get_linear_kernel",
    "build_coregionalized_model",
    "build_model",
]


def get_matern_32_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.Matern32:
    """Matern-3/2 kernel without ARD"""
    return GPy.kern.Matern32(NFEAT, ARD=ARD, **kwargs)


def get_matern_52_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.Matern52:
    """Matern-5/2 kernel without ARD"""
    return GPy.kern.Matern52(NFEAT, ARD=ARD, **kwargs)


def get_ratquad_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.RatQuad:
    """Rational quadratic kernel without ARD"""
    return GPy.kern.RatQuad(NFEAT, ARD=ARD, **kwargs)


def get_linear_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.Linear:
    """Rational quadratic kernel without ARD"""
    return GPy.kern.Linear(NFEAT, ARD=ARD, **kwargs)


def build_coregionalized_model(
    X_train: np.array, y_train: np.array, kernel=None, w_rank: int = 1, **kwargs
) -> GPy.models.GPCoregionalizedRegression:
    """Wrapper for building a coregionalized GPR, it will have as many
    outputs as y_train.shape[1].
    Each output will have its own noise term"""
    NFEAT = X_train.shape[1]
    num_targets = y_train.shape[1]
    if isinstance(kernel, GPy.kern.src.kern.Kern):
        K = kernel
    else:
        K = get_matern_52_kernel(NFEAT)
    icm = GPy.util.multioutput.ICM(
        input_dim=NFEAT, num_outputs=num_targets, kernel=K, W_rank=w_rank
    )

    target_list = [y_train[:, i].reshape(-1, 1) for i in range(num_targets)]
    m = GPCoregionalizedRegression(
        [X_train] * num_targets, target_list, kernel=icm, normalizer=True, **kwargs
    )
    # We constrain the variance of the RBF/Matern ..
    # as the variance is now encoded in the kappa B of the ICM
    # Not constraining it would lead to a degeneracy
    m[".*ICM.*.variance"].constrain_fixed(1.0)
    # initialize the noise model
    m[".*Gaussian_noise_*"] = 1.0
    return m


def build_model(
    X_train: np.array, y_train: np.array, index: int = 0, kernel=None, **kwargs
) -> GPy.models.GPRegression:
    """Build a single-output GPR model"""
    NFEAT = X_train.shape[1]
    if isinstance(kernel, GPy.kern.src.kern.Kern):
        K = kernel
    else:
        K = get_matern_52_kernel(NFEAT)
    m = GPy.models.GPRegression(
        X_train, y_train[:, index].reshape(-1, 1), kernel=K, normalizer=True, **kwargs
    )
    m[".*Gaussian_noise_*"] = 0.1
    return m
