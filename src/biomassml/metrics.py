from tabnanny import verbose
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error, mean_absolute_percentage_error
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy import sqrt
from scipy.stats import norm
from loguru import logger
from .predict import predict, predict_coregionalized
from .build_model import set_xy_coregionalized

__all__ = [
    "get_regression_metrics",
    "loocv_train_test",
    "picp",
    "mpiw",
    "negative_log_likelihood_Gaussian",
]


def get_regression_metrics(y_true, y_pred) -> dict:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred)
    }


def picp(y_true, y_mean, y_err):
    """
    Based on UQ360 implementation
    Prediction Interval Coverage Probability (PICP). Computes the fraction of samples for which the grounds truth lies
    within predicted interval. Measures the prediction interval calibration for regression.
    Args:
        y_true: Ground truth
        y_mean: predicted mean
        y_err: predicted uncertainty
    Returns:
        float: the fraction of samples for which the grounds truth lies within predicted interval
    """
    y_upper = y_mean.squeeze() + y_err.squeeze()
    y_lower = y_mean.squeeze() - y_err.squeeze()
    satisfies_upper_bound = y_true.squeeze() <= y_upper
    satisfies_lower_bound = y_true.squeeze() >= y_lower
    return np.mean(satisfies_upper_bound * satisfies_lower_bound)


def mpiw(y_err):
    """
    Based on UQ360 implementation
    Mean Prediction Interval Width (MPIW). Computes the average width of the prediction intervals. Measures the
    sharpness of intervals.
    Args:
        y_err: predicted uncertainty
    Returns:
        float: the average width of the prediction interval across samples
    """
    return np.mean(np.abs(2*y_err.squeeze()))


def negative_log_likelihood_Gaussian(y_true, y_mean, y_err):
    """
    Based on UQ360 implementation
    Computes Gaussian negative_log_likelihood assuming symmetric band around the mean.
    Args:
        y_true: Ground truth
        y_mean: predicted mean
        y_err: predicted uncertainty
    Returns:
        float: nll
    """
    y_std = y_err
    nll = np.mean(-norm.logpdf(y_true.squeeze(), loc=y_mean.squeeze(), scale=y_std.squeeze()))
    return nll


def loocv_train_test(gp_model, X, y, coregionalized: bool = False, n_restarts=20):
    """
    Perform Leave-One-Out cross-Validation on the model.
    """
    prediction_collection_train = []
    prediction_collection_test = []
    for i, (train_indices, test_indices) in enumerate(LeaveOneOut().split(X)):
        logger.info(f"LOOCV iteration {i}")
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_train = x_scaler.fit_transform(X_train)
        X_test = x_scaler.transform(X_test)

        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)

        predictions_train = {}
        predictions_test = {}
        if coregionalized:
            set_xy_coregionalized(gp_model, X_train, y_train)
            gp_model.optimize_restarts(n_restarts, verbose=False)

            for objective in range(y.shape[1]):
                y_pred_test_mu, y_pred_test_std = predict_coregionalized(
                    gp_model, X_test, index=objective
                )
                y_pred_train_mu, y_pred_train_std = predict_coregionalized(
                    gp_model, X_train, index=objective
                )
                predictions_train[objective] = {
                    "mu": y_pred_train_mu,
                    "std": y_pred_train_std,
                    "true": y_train[:, objective],
                }
                predictions_test[objective] = {
                    "mu": y_pred_test_mu,
                    "std": y_pred_test_std,
                    "true": y_test[:, objective],
                }

        else:
            for objective in range(y.shape[1]):
                gp_model.set_XY(X_train, y_train[:, objective].reshape(-1, 1))
                gp_model.optimize_restarts(n_restarts, verbose=False)
                y_pred_test_mu, y_pred_test_std = predict(gp_model, X_test)
                y_pred_train_mu, y_pred_train_std = predict(gp_model, X_train)
                predictions_train[objective] = {
                    "mu": y_pred_train_mu,
                    "std": y_pred_train_std,
                    "true": y_train[:, objective],
                }
                predictions_test[objective] = {
                    "mu": y_pred_test_mu,
                    "std": y_pred_test_std,
                    "true": y_test[:, objective],
                }
        prediction_collection_train.append(predictions_train)
        prediction_collection_test.append(predictions_test)

    test_metrics = {}

    for objective in range(y.shape[1]):
        y_true = np.concatenate([pred[objective]["true"] for pred in prediction_collection_test])
        y_pred_mu = np.concatenate([pred[objective]["mu"] for pred in prediction_collection_test])
        y_pred_std = np.concatenate([pred[objective]["std"] for pred in prediction_collection_test])
        y_err = y_pred_std
        test_metrics[objective] = {
            "nll": negative_log_likelihood_Gaussian(y_true, y_pred_mu, y_err),
            "picp": picp(y_true, y_pred_mu, y_err),
            "mpiw": mpiw(y_err),
            **get_regression_metrics(y_true, y_pred_mu),
        }
    return test_metrics, prediction_collection_train, prediction_collection_test
