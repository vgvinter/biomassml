from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
from sklearn.model_selection import LOOCV
import numpy as np
from scipy.stats import norm
from .predict import predict, predict_coregionalized

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
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
    }


def picp(y_true, y_err):
    """
    Based on UQ360 implementation
    Prediction Interval Coverage Probability (PICP). Computes the fraction of samples for which the grounds truth lies
    within predicted interval. Measures the prediction interval calibration for regression.
    Args:
        y_true: Ground truth
        y_err: predicted uncertainty
    Returns:
        float: the fraction of samples for which the grounds truth lies within predicted interval.
    """
    y_upper = y_true + y_err
    y_lower = y_true - y_err
    satisfies_upper_bound = y_true <= y_upper
    satisfies_lower_bound = y_true >= y_lower
    return np.mean(satisfies_upper_bound * satisfies_lower_bound)


def mpiw(y_err):
    """
    Based on UQ360 implementation
    Mean Prediction Interval Width (MPIW). Computes the average width of the the prediction intervals. Measures the
    sharpness of intervals.
    Args:
        y_err: predicted uncertainty
    Returns:
        float: the average width the prediction interval across samples.
    """
    return np.mean(np.abs(y_err))


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


def loocv_train_test(gp_model, X, y, coregionalized: bool = False):
    """
    Perform Leave-One-Out cross-Validation on the model.
    """
    prediction_collection_train = []
    prediction_collection_test = []
    for train_indices, test_indices in LOOCV().split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        predictions_train = {}
        predictions_test = {}
        if coregionalized:
            gp_model.optimize_restarts(n_restarts=20)

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
            gp_model.optimize_restarts(n_restarts=20)
            y_pred_test_mu, y_pred_test_std = predict(gp_model, X_test)
            y_pred_train_mu, y_pred_train_std = predict(gp_model, X_train)
            for objective in range(y.shape[1]):
                predictions_train[objective] = {
                    "mu": y_pred_train_mu,
                    "std": y_pred_train_std,
                    "true": y_train[:, objective],
                }
                predictions_test[objective] = {
                    "mu": y_pred_test_mu,
                    "std": y_pred_test_std,
                    "true": y_train[:, objective],
                }
        prediction_collection_train.append(predictions_train)
        prediction_collection_test.append(predictions_test)

    test_metrics = {}

    for objective in range(y.shape[1]):
        y_true = np.concatenate([pred["true"] for pred in prediction_collection_test])
        y_pred_mu = np.concatenate([pred["mu"] for pred in prediction_collection_test])
        y_pred_std = np.concatenate([pred["std"] for pred in prediction_collection_test])
        y_err = y_pred_std
        test_metrics[objective] = {
            "nll": negative_log_likelihood_Gaussian(y_true, y_pred_mu, y_err),
            "picp": picp(y_true, y_err),
            "mpiw": mpiw(y_err),
            **get_regression_metrics(y_true, y_pred_mu),
        }
    return test_metrics, prediction_collection_train, prediction_collection_test
