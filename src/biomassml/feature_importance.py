import shap
from .predict import predict_coregionalized, predict


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
