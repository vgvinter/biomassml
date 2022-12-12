import shap
from numpy import sqrt
from sklearn.inspection import PartialDependenceDisplay
from .predict import predict_coregionalized, predict


def predict_wrapper(model, X, index: int = 0, coregionalized: bool = False):
    if coregionalized:
        mu, _ = predict_coregionalized(model, X, index)
    else:
        mu, _ = predict(model, X)
    return mu.flatten()


def get_shap_values(
    model, X, x_scaler, y_scaler, index: int = 0, custom_wrapper=None, coregionalized: bool = False
):
    """
    Get SHAP values for a given model and data
    X: unscaled X
    y_scaler: y scaler for TARGETS_GASIF = CO, H2, COMB, GAS
    """
    X = x_scaler.transform(X)
    if custom_wrapper is None:
        explainer = shap.KernelExplainer(
            lambda X: predict_wrapper(model, X, index=index, coregionalized=coregionalized), X
        )
    else:
        explainer = shap.KernelExplainer(lambda X: custom_wrapper(model, X, y_scaler), X)
    shap_values = explainer.shap_values(X)
    return shap_values


def custom_wrapper_CH4(model, X, y_scaler):
    """
    y_scaler: y scaler for TARGETS_GASIF = CO, H2, COMB, GAS
    """
    y_pred_mu_CO = (
        predict_coregionalized(model, X, 0)[0] * sqrt(y_scaler.var_[0]) + y_scaler.mean_[0]
    )
    y_pred_mu_H2 = (
        predict_coregionalized(model, X, 1)[0] * sqrt(y_scaler.var_[1]) + y_scaler.mean_[1]
    )
    y_pred_mu_COMB = (
        predict_coregionalized(model, X, 2)[0] * sqrt(y_scaler.var_[2]) + y_scaler.mean_[2]
    )
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    return y_pred_mu_CH4.flatten()


class GPYEstimator:
    """Helper to mimic the interface of sklearn.base.BaseEstimator for partial dependency plots
    X: unscaled X
    """

    def __init__(self, model, i, x_scaler, y_scaler):
        self._fitted = True
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.model = model
        self.i = i
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def fit(self, X, y):
        X = self.x_scaler.transform(X)  # not sure if this is needed also here
        ...

    def predict(self, X):
        X = self.x_scaler.transform(X)
        y_pred_mu = (
            predict_coregionalized(self.model, X, self.i)[0] * sqrt(self.y_scaler.var_[self.i])
            + self.y_scaler.mean_[self.i]
        )
        return y_pred_mu.flatten()


class GPY_CH4_Estimator:
    """Helper to mimic the interface of sklearn.base.BaseEstimator for partial dependency plots
    X: unscaled X
    y_scaler: y scaler for TARGETS_GASIF = CO, H2, COMB, GAS
    """

    def __init__(self, model, x_scaler, y_scaler):
        self._fitted = True
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def fit(self, X, y):
        X = self.x_scaler.transform(X)  # not sure if this is needed also here
        ...

    def predict(self, X):
        X = self.x_scaler.transform(X)
        y_pred_mu_CO = (
            predict_coregionalized(self.model, X, 0)[0] * sqrt(self.y_scaler.var_[0])
            + self.y_scaler.mean_[0]
        )
        y_pred_mu_H2 = (
            predict_coregionalized(self.model, X, 1)[0] * sqrt(self.y_scaler.var_[1])
            + self.y_scaler.mean_[1]
        )
        y_pred_mu_COMB = (
            predict_coregionalized(self.model, X, 2)[0] * sqrt(self.y_scaler.var_[2])
            + self.y_scaler.mean_[2]
        )
        y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
        return y_pred_mu_CH4.flatten()
