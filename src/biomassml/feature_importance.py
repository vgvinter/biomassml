import shap
from .predict import predict_coregionalized, predict
from sklearn.inspection import permutation_importance


def predict_wrapper(model, X, index: int = 0, coregionalized: bool = False):
    if coregionalized:
        mu, _ = predict_coregionalized(model, X, index)
        return mu.flatten()
    mu, _ = predict(model, X, index)
    return mu.flatten()


def get_shap_values(model, X, index: int = 0, custom_wrapper=None, coregionalized: bool = False):
    """
    Get SHAP values for a given model and data.
    """
    if custom_wrapper is None:
        explainer = shap.KernelExplainer(
            lambda X: predict_wrapper(model, X, index=index, coregionalized=coregionalized), X
        )
    else:
        explainer = shap.KernelExplainer(lambda X: custom_wrapper(model, X), X)
    shap_values = explainer.shap_values(X)
    return shap_values

'''
class GPYEstimator:
    """Helper to mimic the interface of sklearn.base.BaseEstimator for partial dependency plots
    X: X_scaled
    """

    def __init__(self, model, i):
        self._fitted = True
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.model = model
        self.i = i

    def fit(self, X, y):
        ...

    def predict(self, X):
        return predict_coregionalized(self.model, X, self.i)[0].flatten()


class GPYMethaneEstimator:
    """Helper to mimic the interface of sklearn.base.BaseEstimator for partial dependency plots"""

    def __init__(self, model):
        self._fitted = True
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.model = model

    def fit(self, X, y):
        ...

    def predict(self, X):
        return (
            predict_coregionalized(self.model, X, 2)[0].flatten()
            - predict_coregionalized(self.model, X, 0)[0].flatten()
            - predict_coregionalized(self.model, X, 1)[0].flatten()
        )
'''


class GPYEstimator:
    """Helper to mimic the interface of sklearn.base.BaseEstimator for partial dependency plots
    X: unscaled X
    """

    def __init__(self, model, i, x_scaler, y_scaler_4):
        self._fitted = True
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.model = model
        self.i = i
        self.x_scaler = x_scaler
        self.y_scaler_4 = y_scaler_4

    def fit(self, X, y):
        X = scale_X(X, self.x_scaler)  # not sure if this is needed
        ...

    def predict(self, X):
        X = scale_X(X, self.x_scaler)
        y_pred_mu = predict_coregionalized(
            self.model, X, self.i)[0]*sqrt(self.y_scaler_4.var_[self.i]) + self.y_scaler_4.mean_[self.i]
        return y_pred_mu.flatten()





