import shap
from numpy import sqrt
from .predict import predict_coregionalized, predict
from sklearn.inspection import permutation_importance
from .predict_outputs import scale_X


def predict_wrapper(model, X, index: int = 0, coregionalized: bool = False):
    if coregionalized:
        mu, _ = predict_coregionalized(model, X, index)
        return mu.flatten()
    mu, _ = predict(model, X, index)
    return mu.flatten()


def get_shap_values(model, X, x_scaler, y_scaler_11,
                    index: int = 0, custom_wrapper=None, coregionalized: bool = False):
    """
    Get SHAP values for a given model and data.
    """
    if custom_wrapper is None:
        explainer = shap.KernelExplainer(
            lambda X: predict_wrapper(model, X, index=index, coregionalized=coregionalized), X
        )
    else:
        explainer = shap.KernelExplainer(lambda X: custom_wrapper(model, X, x_scaler, y_scaler_11), X)
    shap_values = explainer.shap_values(X)
    return shap_values


def custom_wrapper_CH4(model, X, x_scaler, y_scaler_11):
    X = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_mu_H2 = predict_coregionalized(model, X, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_mu_COMB = predict_coregionalized(model, X, 2)[0]*sqrt(y_scaler_11.var_[4]) + y_scaler_11.mean_[4]
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    y_pred_mu_CH4_scaled = ((y_pred_mu_CH4 - y_scaler_11.mean_[2])/sqrt(y_scaler_11.var_[2])).flatten() 
    return y_pred_mu_CH4_scaled


def custom_wrapper_H2CO(model, X, x_scaler, y_scaler_11):
    X = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_mu_H2 = predict_coregionalized(model, X, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_mu_H2CO = y_pred_mu_H2 / y_pred_mu_CO
    y_pred_mu_H2CO_scaled = ((y_pred_mu_H2CO - y_scaler_11.mean_[6])/sqrt(y_scaler_11.var_[6])).flatten() 
    return y_pred_mu_H2CO_scaled


def custom_wrapper_HHV(model, X, x_scaler, y_scaler_11):
    X = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_mu_H2 = predict_coregionalized(model, X, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_mu_COMB = predict_coregionalized(model, X, 2)[0]*sqrt(y_scaler_11.var_[4]) + y_scaler_11.mean_[4]
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    y_pred_mu_HHV = (11.76*y_pred_mu_CO + 11.882*y_pred_mu_H2 + 37.024*y_pred_mu_CH4)/100
    y_pred_mu_HHV_scaled = ((y_pred_mu_HHV - y_scaler_11.mean_[7])/sqrt(y_scaler_11.var_[7])).flatten()
    return y_pred_mu_HHV_scaled


def custom_wrapper_GAS(model, X, x_scaler, y_scaler_11):
    X = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_mu_H2 = predict_coregionalized(model, X, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_mu_COMB = predict_coregionalized(model, X, 2)[0]*sqrt(y_scaler_11.var_[4]) + y_scaler_11.mean_[4]
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    y_pred_mu_HHV = (11.76*y_pred_mu_CO + 11.882*y_pred_mu_H2 + 37.024*y_pred_mu_CH4)/100
    y_pred_mu_Edens = predict_coregionalized(model, X, 3)[0]*sqrt(y_scaler_11.var_[9]) + y_scaler_11.mean_[9] 
    y_pred_mu_GAS = y_pred_mu_Edens / y_pred_mu_HHV
    y_pred_mu_GAS_scaled = ((y_pred_mu_GAS - y_scaler_11.mean_[10])/sqrt(y_scaler_11.var_[10])).flatten()
    return y_pred_mu_GAS_scaled


def custom_wrapper_Edensity(model, X, x_scaler, y_scaler_11):
    X = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_mu_H2 = predict_coregionalized(model, X, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_mu_COMB = predict_coregionalized(model, X, 2)[0]*sqrt(y_scaler_11.var_[4]) + y_scaler_11.mean_[4]
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    y_pred_mu_HHV = (11.76*y_pred_mu_CO + 11.882*y_pred_mu_H2 + 37.024*y_pred_mu_CH4)/100
    y_pred_mu_GAS = predict_coregionalized(model, X, 3)[0]*sqrt(y_scaler_11.var_[10]) + y_scaler_11.mean_[10] 
    y_pred_mu_Edens = y_pred_mu_HHV * y_pred_mu_GAS
    y_pred_mu_Edens_scaled = ((y_pred_mu_Edens - y_scaler_11.mean_[9])/sqrt(y_scaler_11.var_[9])).flatten()
    return y_pred_mu_Edens_scaled


def custom_wrapper_CGE(model, X, x_scaler, y_scaler_11, HHVbiom):
    """HHVbiom = FEATURES_GASIF_12['HHVbiom (MJ/kg)']"""
    X = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_mu_H2 = predict_coregionalized(model, X, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_mu_COMB = predict_coregionalized(model, X, 2)[0]*sqrt(y_scaler_11.var_[4]) + y_scaler_11.mean_[4]
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    y_pred_mu_HHV = (11.76*y_pred_mu_CO + 11.882*y_pred_mu_H2 + 37.024*y_pred_mu_CH4)/100
    y_pred_mu_Edens = predict_coregionalized(model, X, 3)[0]*sqrt(y_scaler_11.var_[9]) + y_scaler_11.mean_[9] 
    y_pred_mu_GAS = y_pred_mu_Edens / y_pred_mu_HHV
    y_pred_mu_CGE = y_pred_mu_GAS * y_pred_mu_HHV / np.reshape(HHVbiom.values, (-1,1))*100
    y_pred_mu_CGE_scaled = ((y_pred_mu_CGE - y_scaler_11.mean_[8])/sqrt(y_scaler_11.var_[8])).flatten()
    return y_pred_mu_CGE_scaled


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


class GPY_CH4_Estimator:
    
    def __init__(self, model, x_scaler, y_scaler_11):
        self._fitted = True
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler_11 = y_scaler_11
        
    def fit(self, X, y):
        X = scale_X(X, self.x_scaler)  # not sure if this is needed
        ...
        
    def predict(self, X):
        X = scale_X(X, self.x_scaler)
        y_pred_mu_CO = predict_coregionalized(self.model, X, 0)[0]*sqrt(self.y_scaler_11.var_[1]) + self.y_scaler_11.mean_[1]
        y_pred_mu_H2 = predict_coregionalized(self.model, X, 1)[0]*sqrt(self.y_scaler_11.var_[3]) + self.y_scaler_11.mean_[3]
        y_pred_mu_COMB = predict_coregionalized(self.model, X, 2)[0]*sqrt(self.y_scaler_11.var_[4]) + self.y_scaler_11.mean_[4]
        y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
        return y_pred_mu_CH4.flatten()


class GPY_H2CO_Estimator:
    
    def __init__(self, model, x_scaler, y_scaler_11):
        self._fitted = True
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler_11 = y_scaler_11
        
    def fit(self, X, y):
        X = scale_X(X, self.x_scaler)  # not sure if this is needed
        ...
        
    def predict(self, X):
        X = scale_X(X, self.x_scaler)
        y_pred_mu_CO = predict_coregionalized(self.model, X, 0)[0]*sqrt(self.y_scaler_11.var_[1]) + self.y_scaler_11.mean_[1]
        y_pred_mu_H2 = predict_coregionalized(self.model, X, 1)[0]*sqrt(self.y_scaler_11.var_[3]) + self.y_scaler_11.mean_[3]
        y_pred_mu_H2CO = y_pred_mu_H2 / y_pred_mu_CO
        return y_pred_mu_H2CO.flatten()


class GPY_HHV_Estimator:
    
    def __init__(self, model, x_scaler, y_scaler_11):
        self._fitted = True
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler_11 = y_scaler_11
        
    def fit(self, X, y):
        X = scale_X(X, self.x_scaler)  # not sure if this is needed
        ...
        
    def predict(self, X):
        X = scale_X(X, self.x_scaler)
        y_pred_mu_CO = predict_coregionalized(self.model, X, 0)[0]*sqrt(self.y_scaler_11.var_[1]) + self.y_scaler_11.mean_[1]
        y_pred_mu_H2 = predict_coregionalized(self.model, X, 1)[0]*sqrt(self.y_scaler_11.var_[3]) + self.y_scaler_11.mean_[3]
        y_pred_mu_COMB = predict_coregionalized(self.model, X, 2)[0]*sqrt(self.y_scaler_11.var_[4]) + self.y_scaler_11.mean_[4]
        y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
        y_pred_mu_HHV = (11.76*y_pred_mu_CO + 11.882*y_pred_mu_H2 + 37.024*y_pred_mu_CH4)/100
        return y_pred_mu_HHV.flatten()


class GPY_GAS_Estimator:
    
    def __init__(self, model, x_scaler, y_scaler_11):
        self._fitted = True
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler_11 = y_scaler_11
        
    def fit(self, X, y):
        X = scale_X(X, self.x_scaler)  # not sure if this is needed
        ...
        
    def predict(self, X):
        X = scale_X(X, self.x_scaler)
        y_pred_mu_CO = predict_coregionalized(self.model, X, 0)[0]*sqrt(self.y_scaler_11.var_[1]) + self.y_scaler_11.mean_[1]
        y_pred_mu_H2 = predict_coregionalized(self.model, X, 1)[0]*sqrt(self.y_scaler_11.var_[3]) + self.y_scaler_11.mean_[3]
        y_pred_mu_COMB = predict_coregionalized(self.model, X, 2)[0]*sqrt(self.y_scaler_11.var_[4]) + self.y_scaler_11.mean_[4]
        y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
        y_pred_mu_HHV = (11.76*y_pred_mu_CO + 11.882*y_pred_mu_H2 + 37.024*y_pred_mu_CH4)/100
        y_pred_mu_Edens = predict_coregionalized(
            self.model, X, 3)[0]*sqrt(self.y_scaler_11.var_[9]) + self.y_scaler_11.mean_[9] 
        y_pred_mu_GAS = y_pred_mu_Edens / y_pred_mu_HHV
        return y_pred_mu_GAS.flatten()


class GPY_Edensity_Estimator:
    
    def __init__(self, model, x_scaler, y_scaler_11):
        self._fitted = True
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler_11 = y_scaler_11
        
    def fit(self, X, y):
        X = scale_X(X, self.x_scaler)  # not sure if this is needed
        ...
        
    def predict(self, X):
        X = scale_X(X, self.x_scaler)
        y_pred_mu_CO = predict_coregionalized(self.model, X, 0)[0]*sqrt(self.y_scaler_11.var_[1]) + self.y_scaler_11.mean_[1]
        y_pred_mu_H2 = predict_coregionalized(self.model, X, 1)[0]*sqrt(self.y_scaler_11.var_[3]) + self.y_scaler_11.mean_[3]
        y_pred_mu_COMB = predict_coregionalized(self.model, X, 2)[0]*sqrt(self.y_scaler_11.var_[4]) + self.y_scaler_11.mean_[4]
        y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
        y_pred_mu_HHV = (11.76*y_pred_mu_CO + 11.882*y_pred_mu_H2 + 37.024*y_pred_mu_CH4)/100
        y_pred_mu_GAS = predict_coregionalized(
            self.model, X, 3)[0]*sqrt(self.y_scaler_11.var_[10]) + self.y_scaler_11.mean_[10] 
        y_pred_mu_Edens = y_pred_mu_HHV * y_pred_mu_GAS
        return y_pred_mu_Edens.flatten()


class GPY_CGE_Estimator:
    
    def __init__(self, model, x_scaler, y_scaler_11, HHVbiom):
        """HHVbiom = FEATURES_GASIF_12['HHVbiom (MJ/kg)']"""
        self._fitted = True
        self.fitted_ = True
        self._estimator_type = "regressor"
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler_11 = y_scaler_11
        self.HHVbiom = HHVbiom
        
    def fit(self, X, y):
        X = scale_X(X, self.x_scaler)  # not sure if this is needed
        ...
        
    def predict(self, X):
        X = scale_X(X, self.x_scaler)
        y_pred_mu_CO = predict_coregionalized(self.model, X, 0)[0]*sqrt(self.y_scaler_11.var_[1]) + self.y_scaler_11.mean_[1]
        y_pred_mu_H2 = predict_coregionalized(self.model, X, 1)[0]*sqrt(self.y_scaler_11.var_[3]) + self.y_scaler_11.mean_[3]
        y_pred_mu_COMB = predict_coregionalized(self.model, X, 2)[0]*sqrt(self.y_scaler_11.var_[4]) + self.y_scaler_11.mean_[4]
        y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
        y_pred_mu_HHV = (11.76*y_pred_mu_CO + 11.882*y_pred_mu_H2 + 37.024*y_pred_mu_CH4)/100
        y_pred_mu_Edens = predict_coregionalized(
            self.model, X, 3)[0]*sqrt(self.y_scaler_11.var_[9]) + self.y_scaler_11.mean_[9] 
        y_pred_mu_GAS = y_pred_mu_Edens / y_pred_mu_HHV
        y_pred_mu_CGE = y_pred_mu_GAS * y_pred_mu_HHV / np.reshape(self.HHVbiom.values, (-1,1))*100
        return y_pred_mu_CGE.flatten()

