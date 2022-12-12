import GPy
import numpy as np
from .coregionalized_regressor import GPCoregionalizedRegression

__all__ = [
    "get_rbf_kernel",
    "get_matern_32_kernel",
    "get_matern_52_kernel",
    "get_ratquad_kernel",
    "get_linear_kernel",
    "build_coregionalized_model",
    "build_model",
    "set_xy_coregionalized",
]


ARD_WRAPPERS = {
    "matern_32": lambda X: get_matern_32_kernel(X),
    "matern_52": lambda X: get_matern_52_kernel(X),
    "ratquad": lambda X: get_ratquad_kernel(X),
    "linear": lambda X: get_linear_kernel(X),
    "rbf": lambda X: get_rbf_kernel(X),
    "rbf_plus_linear": lambda X: get_rbf_kernel(X) + get_linear_kernel(X),
    "matern_32_plus_linear": lambda X: get_matern_32_kernel(X) + get_linear_kernel(X),
    "matern_52_plus_linear": lambda X: get_matern_52_kernel(X) + get_linear_kernel(X),
    "ratquad_plus_linear": lambda X: get_ratquad_kernel(X) + get_linear_kernel(X),
    "rbf_linear": lambda X: get_rbf_kernel(X) * get_linear_kernel(X),
    "matern_32_linear": lambda X: get_matern_32_kernel(X) * get_linear_kernel(X),
    "matern_52_linear": lambda X: get_matern_52_kernel(X) * get_linear_kernel(X),
    "ratquad_linear": lambda X: get_ratquad_kernel(X) * get_linear_kernel(X),
}

NO_ARD_WRAPPERS = {
    "matern_32": lambda X: get_matern_32_kernel(X, ARD=False),
    "matern_52": lambda X: get_matern_52_kernel(X, ARD=False),
    "ratquad": lambda X: get_ratquad_kernel(X, ARD=False),
    "linear": lambda X: get_linear_kernel(X, ARD=False),
    "rbf": lambda X: get_rbf_kernel(X, ARD=False),
    "rbf_plus_linear": lambda X: get_rbf_kernel(X, ARD=False) + get_linear_kernel(X, ARD=False),
    "matern_32_plus_linear": lambda X: get_matern_32_kernel(X, ARD=False)
    + get_linear_kernel(X, ARD=False),
    "matern_52_plus_linear": lambda X: get_matern_52_kernel(X, ARD=False)
    + get_linear_kernel(X, ARD=False),
    "ratquad_plus_linear": lambda X: get_ratquad_kernel(X, ARD=False)
    + get_linear_kernel(X, ARD=False),
    "rbf_linear": lambda X: get_rbf_kernel(X, ARD=False) * get_linear_kernel(X, ARD=False),
    "matern_32_linear": lambda X: get_matern_32_kernel(X, ARD=False)
    * get_linear_kernel(X, ARD=False),
    "matern_52_linear": lambda X: get_matern_52_kernel(X, ARD=False)
    * get_linear_kernel(X, ARD=False),
    "ratquad_linear": lambda X: get_ratquad_kernel(X, ARD=False) * get_linear_kernel(X, ARD=False),
}


def get_rbf_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.RBF:
    """Radial basis function kernel"""
    return GPy.kern.RBF(NFEAT, ARD=ARD, **kwargs)


def get_matern_32_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.Matern32:
    """Matern-3/2 kernel"""
    return GPy.kern.Matern32(NFEAT, ARD=ARD, **kwargs)


def get_matern_52_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.Matern52:
    """Matern-5/2 kernel"""
    return GPy.kern.Matern52(NFEAT, ARD=ARD, **kwargs)


def get_ratquad_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.RatQuad:
    """Rational quadratic kernel"""
    return GPy.kern.RatQuad(NFEAT, ARD=ARD, **kwargs)


def get_linear_kernel(NFEAT: int, ARD=True, **kwargs) -> GPy.kern.Linear:
    """Linear kernel"""
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


def set_xy_coregionalized(model, X, y, mask=None):
    """Wrapper to update a coregionalized model with new data"""
    num_target = y.shape[1]
    if mask is None:
        X_array = [X] * num_target
        y_array = [y[:, i].reshape(-1, 1) for i in range(num_target)]

    else:
        X_array = [X[mask[:, i]] for i in range(num_target)]
        y_array = [y[mask[:, i], i].reshape(-1, 1) for i in range(num_target)]

    model.set_XY(X_array, y_array)

    return model
