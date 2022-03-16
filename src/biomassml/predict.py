import GPy
import numpy as np
from .coregionalized_regressor import GPCoregionalizedRegression


def predict_coregionalized(
    model: GPy.models.GPCoregionalizedRegression, X: np.array, index: int = 0
):
    """Wrapper function for the prediction method of a coregionalized
    GPy regression model.
    It return the standard deviation instead of the variance"""
    assert isinstance(
        model, (GPy.models.GPCoregionalizedRegression, GPCoregionalizedRegression)
    ), "This wrapper function is written for GPy.models.GPCoregionalizedRegression"
    newX = np.hstack([X, index * np.ones_like(X)])
    mu_c0, var_c0 = model.predict(
        newX,
        Y_metadata={"output_index": index * np.ones((newX.shape[0], 1)).astype(int)},
    )

    return mu_c0, np.sqrt(var_c0)


def predict(model: GPy.models.GPRegression, X: np.array):
    """Wrapper function for the prediction method of a GPy regression model.
    It return the standard deviation instead of the variance"""
    assert isinstance(
        model, GPy.models.GPRegression
    ), "This wrapper function is written for GPy.models.GPRegression"
    mu, var = model.predict(X)
    return mu, np.sqrt(var)
