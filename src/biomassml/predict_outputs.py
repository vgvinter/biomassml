import numpy as np
from numpy import sqrt
from sklearn.preprocessing import StandardScaler
from .predict import predict_coregionalized
from .utils import additive_errorprop


def get_scalers(X, y):
    """
    X: FEATURES_GASIF
    y: TARGETS_GASIF
    """
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    return x_scaler, y_scaler


def scale_X(X, x_scaler):
    X_scaled = x_scaler.transform(X)
    return X_scaled


def scale_output(y_pred_mu, y_pred_std, y_scaler, n):
    """It returns scaled predictions
    y_pred_mu: unscaled predicted mean
    y_pred_std: unscaled predicted uncertainty
    y_scaler: y scaler
    n: output index in y_scaler
    """
    y_pred_mu_scaled = ((y_pred_mu - y_scaler.mean_[n])/sqrt(y_scaler.var_[n])).flatten() 
    y_pred_std_scaled = (y_pred_std/sqrt(y_scaler.var_[n])).flatten()
    return y_pred_mu_scaled, y_pred_std_scaled


def predict_CO(X, model, x_scaler, y_scaler):
    """It returns unscaled predictions
    y_scaler: y scaler for TARGETS_GASIF = CO, H2, COMB, GAS, CH4
    """
    X_scaled = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X_scaled, 0)[0]*sqrt(y_scaler.var_[0]) + y_scaler.mean_[0]
    y_pred_std_CO = predict_coregionalized(model, X_scaled, 0)[1]*sqrt(y_scaler.var_[0])
    return y_pred_mu_CO, y_pred_std_CO 


def predict_H2(X, model, x_scaler, y_scaler):
    """It returns unscaled predictions
    y_scaler: y scaler for TARGETS_GASIF = CO, H2, COMB, GAS, CH4
    """
    X_scaled = x_scaler.transform(X)
    y_pred_mu_H2 = predict_coregionalized(model, X_scaled, 1)[0]*sqrt(y_scaler.var_[1]) + y_scaler.mean_[1]
    y_pred_std_H2 = predict_coregionalized(model, X_scaled, 1)[1]*sqrt(y_scaler.var_[1])
    return y_pred_mu_H2, y_pred_std_H2


def predict_COMB(X, model, x_scaler, y_scaler):
    """It returns unscaled predictions
    y_scaler: y scaler for TARGETS_GASIF = CO, H2, COMB, GAS, CH4
    """
    X_scaled = x_scaler.transform(X)
    y_pred_mu_COMB = predict_coregionalized(model, X_scaled, 2)[0]*sqrt(y_scaler.var_[2]) + y_scaler.mean_[2]
    y_pred_std_COMB = predict_coregionalized(model, X_scaled, 2)[1]*sqrt(y_scaler.var_[2])
    return y_pred_mu_COMB, y_pred_std_COMB   


def predict_GAS(X, model, x_scaler, y_scaler):
    """It returns unscaled predictions
    y_scaler: y scaler for TARGETS_GASIF = CO, H2, COMB, GAS, CH4
    """
    X_scaled = x_scaler.transform(X)
    y_pred_mu_GAS = predict_coregionalized(model, X_scaled, 3)[0]*sqrt(y_scaler.var_[3]) + y_scaler.mean_[3]
    y_pred_std_GAS = predict_coregionalized(model, X_scaled, 3)[1]*sqrt(y_scaler.var_[3])
    return y_pred_mu_GAS, y_pred_std_GAS


def predict_CH4(X, model, x_scaler, y_scaler):
    """It returns unscaled predictions
    y_scaler: y scaler for TARGETS_GASIF = CO, H2, COMB, GAS, CH4
    """
    X_scaled = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X_scaled, 0)[0]*sqrt(y_scaler.var_[0]) + y_scaler.mean_[0]
    y_pred_std_CO = predict_coregionalized(model, X_scaled, 0)[1]*sqrt(y_scaler.var_[0])
    y_pred_mu_H2 = predict_coregionalized(model, X_scaled, 1)[0]*sqrt(y_scaler.var_[1]) + y_scaler.mean_[1]
    y_pred_std_H2 = predict_coregionalized(model, X_scaled, 1)[1]*sqrt(y_scaler.var_[1])
    y_pred_mu_COMB = predict_coregionalized(model, X_scaled, 2)[0]*sqrt(y_scaler.var_[2]) + y_scaler.mean_[2]
    y_pred_std_COMB = predict_coregionalized(model, X_scaled, 2)[1]*sqrt(y_scaler.var_[2])
    
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    y_pred_std_CH4 = additive_errorprop([y_pred_std_COMB, y_pred_std_CO, y_pred_std_H2])

    return y_pred_mu_CH4, y_pred_std_CH4


def predict_CH4_covar(X, y, model, x_scaler, y_scaler):
    """It returns unscaled predictions considering covariance in the calculation of the error propagation
    y_scaler: y scaler for TARGETS_GASIF = CO, H2, COMB, GAS, CH4
    """
    X_scaled = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X_scaled, 0)[0]*sqrt(y_scaler.var_[0]) + y_scaler.mean_[0]
    y_pred_std_CO = predict_coregionalized(model, X_scaled, 0)[1]*sqrt(y_scaler.var_[0])
    y_pred_mu_H2 = predict_coregionalized(model, X_scaled, 1)[0]*sqrt(y_scaler.var_[1]) + y_scaler.mean_[1]
    y_pred_std_H2 = predict_coregionalized(model, X_scaled, 1)[1]*sqrt(y_scaler.var_[1])
    y_pred_mu_COMB = predict_coregionalized(model, X_scaled, 2)[0]*sqrt(y_scaler.var_[2]) + y_scaler.mean_[2]
    y_pred_std_COMB = predict_coregionalized(model, X_scaled, 2)[1]*sqrt(y_scaler.var_[2])
    
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    
    cov1 = np.corrcoef(y['volCOMB (%)'], y['volCO (%)'])[0,1]*y_pred_std_COMB.flatten()*y_pred_std_CO.flatten()
    cov2 = np.corrcoef(y['volCOMB (%)'], y['volH2 (%)'])[0,1]*y_pred_std_COMB.flatten()*y_pred_std_H2.flatten()
    cov3 = np.corrcoef(y['volCO (%)'], y['volH2 (%)'])[0,1]*y_pred_std_CO.flatten()*y_pred_std_H2.flatten()
    y_pred_std_CH4_covar = sqrt(np.sum(np.square([y_pred_std_COMB, y_pred_std_CO, y_pred_std_H2]),
                                       axis=0).flatten() - 2*cov1 - 2*cov2 + 2*cov3)

    return y_pred_mu_CH4, y_pred_std_CH4_covar

