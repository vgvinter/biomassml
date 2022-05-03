import numpy as np
from numpy import sqrt
from sklearn.preprocessing import StandardScaler
from .predict import predict_coregionalized
from .utils import additive_errorprop


def get_scalers(X, y):
    x_scaler = StandardScaler()
    y_scaler_11 = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler_11.fit_transform(y)
    return x_scaler, y_scaler_11


def scale_output(y_pred_mu, y_pred_std, y_scaler_11, n):
    y_pred_mu_scaled = ((y_pred_mu - y_scaler_11.mean_[n])/sqrt(y_scaler_11.var_[n])).flatten() 
    y_pred_std_scaled = (y_pred_std/sqrt(y_scaler_11.var_[n])).flatten()
    return y_pred_mu_scaled, y_pred_std_scaled


def predict_CO(X, model, x_scaler, y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X_scaled, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_std_CO = predict_coregionalized(model, X_scaled, 0)[1]*sqrt(y_scaler_11.var_[1])
    return y_pred_mu_CO, y_pred_std_CO 


def predict_H2(X, model, x_scaler, y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_H2 = predict_coregionalized(model, X_scaled, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_std_H2 = predict_coregionalized(model, X_scaled, 1)[1]*sqrt(y_scaler_11.var_[3])
    return y_pred_mu_H2, y_pred_std_H2


def predict_COMB(X, model, x_scaler, y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_COMB = predict_coregionalized(model, X_scaled, 2)[0]*sqrt(y_scaler_11.var_[4]) + y_scaler_11.mean_[4]
    y_pred_std_COMB = predict_coregionalized(model, X_scaled, 2)[1]*sqrt(y_scaler_11.var_[4])
    return y_pred_mu_COMB, y_pred_std_COMB   


def predict_Edensity(X, model, x_scaler, y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_Edens = predict_coregionalized(model, X_scaled, 3)[0]*sqrt(y_scaler_11.var_[9]) + y_scaler_11.mean_[9]
    y_pred_std_Edens = predict_coregionalized(model, X_scaled, 3)[1]*sqrt(y_scaler_11.var_[9])
    return y_pred_mu_Edens, y_pred_std_Edens


def predict_CH4(X, model, x_scaler, y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X_scaled, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_std_CO = predict_coregionalized(model, X_scaled, 0)[1]*sqrt(y_scaler_11.var_[1])
    y_pred_mu_H2 = predict_coregionalized(model, X_scaled, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_std_H2 = predict_coregionalized(model, X_scaled, 1)[1]*sqrt(y_scaler_11.var_[3])
    y_pred_mu_COMB = predict_coregionalized(model, X_scaled, 2)[0]*sqrt(y_scaler_11.var_[4]) + y_scaler_11.mean_[4]
    y_pred_std_COMB = predict_coregionalized(model, X_scaled, 2)[1]*sqrt(y_scaler_11.var_[4])
    
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    y_pred_std_CH4 = additive_errorprop([y_pred_std_COMB, y_pred_std_CO, y_pred_std_H2])

    return y_pred_mu_CH4, y_pred_std_CH4


def predict_H2CO(X, model, x_scaler, y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X_scaled, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_std_CO = predict_coregionalized(model, X_scaled, 0)[1]*sqrt(y_scaler_11.var_[1])
    y_pred_mu_H2 = predict_coregionalized(model, X_scaled, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_std_H2 = predict_coregionalized(model, X_scaled, 1)[1]*sqrt(y_scaler_11.var_[3])
    
    y_pred_mu_H2CO = y_pred_mu_H2 / y_pred_mu_CO
    y_pred_std_H2CO = additive_errorprop([y_pred_std_H2/y_pred_mu_H2, y_pred_std_CO/y_pred_mu_CO])*y_pred_mu_H2CO

    return y_pred_mu_H2CO, y_pred_std_H2CO


def predict_HHV(X, model, x_scaler, y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X_scaled, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_std_CO = predict_coregionalized(model, X_scaled, 0)[1]*sqrt(y_scaler_11.var_[1])
    y_pred_mu_H2 = predict_coregionalized(model, X_scaled, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_std_H2 = predict_coregionalized(model, X_scaled, 1)[1]*sqrt(y_scaler_11.var_[3])
    y_pred_mu_COMB = predict_coregionalized(model, X_scaled, 2)[0]*sqrt(y_scaler_11.var_[4]) + y_scaler_11.mean_[4]
    y_pred_std_COMB = predict_coregionalized(model, X_scaled, 2)[1]*sqrt(y_scaler_11.var_[4])
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    y_pred_std_CH4 = additive_errorprop([y_pred_std_COMB, y_pred_std_CO, y_pred_std_H2])
    
    y_pred_mu_HHV = (11.76*y_pred_mu_CO + 11.882*y_pred_mu_H2 + 37.024*y_pred_mu_CH4)/100
    y_pred_std_HHV = additive_errorprop([11.76/100*y_pred_std_CO, 11.882/100*y_pred_std_H2, 37.024/100*y_pred_std_CH4])

    return y_pred_mu_HHV, y_pred_std_HHV


def predict_GAS(X, model, x_scaler, y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X_scaled, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_std_CO = predict_coregionalized(model, X_scaled, 0)[1]*sqrt(y_scaler_11.var_[1])
    y_pred_mu_H2 = predict_coregionalized(model, X_scaled, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_std_H2 = predict_coregionalized(model, X_scaled, 1)[1]*sqrt(y_scaler_11.var_[3])
    y_pred_mu_COMB = predict_coregionalized(model, X_scaled, 2)[0]*sqrt(y_scaler_11.var_[4]) + y_scaler_11.mean_[4]
    y_pred_std_COMB = predict_coregionalized(model, X_scaled, 2)[1]*sqrt(y_scaler_11.var_[4])
    y_pred_mu_Edens = predict_coregionalized(model, X_scaled, 3)[0]*sqrt(y_scaler_11.var_[9]) + y_scaler_11.mean_[9]
    y_pred_std_Edens = predict_coregionalized(model, X_scaled, 3)[1]*sqrt(y_scaler_11.var_[9])
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    y_pred_std_CH4 = additive_errorprop([y_pred_std_COMB, y_pred_std_CO, y_pred_std_H2])
    y_pred_mu_HHV = (11.76*y_pred_mu_CO + 11.882*y_pred_mu_H2 + 37.024*y_pred_mu_CH4)/100
    y_pred_std_HHV = additive_errorprop([11.76/100*y_pred_std_CO, 11.882/100*y_pred_std_H2, 37.024/100*y_pred_std_CH4])
    
    y_pred_mu_GAS = y_pred_mu_Edens / y_pred_mu_HHV
    y_pred_std_GAS = additive_errorprop([y_pred_std_Edens/y_pred_mu_Edens,
                                         y_pred_std_HHV/y_pred_mu_HHV])*y_pred_mu_GAS
    
    return y_pred_mu_GAS, y_pred_std_GAS


def predict_CGE(X, model, x_scaler, y_scaler_11, HHVbiom):
    '''HHVbiom=FEATURES_GASIF_12['HHVbiom (MJ/kg)']'''
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X_scaled, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_std_CO = predict_coregionalized(model, X_scaled, 0)[1]*sqrt(y_scaler_11.var_[1])
    y_pred_mu_H2 = predict_coregionalized(model, X_scaled, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_std_H2 = predict_coregionalized(model, X_scaled, 1)[1]*sqrt(y_scaler_11.var_[3])
    y_pred_mu_COMB = predict_coregionalized(model, X_scaled, 2)[0]*sqrt(y_scaler_11.var_[4]) + y_scaler_11.mean_[4]
    y_pred_std_COMB = predict_coregionalized(model, X_scaled, 2)[1]*sqrt(y_scaler_11.var_[4])
    y_pred_mu_Edens = predict_coregionalized(model, X_scaled, 3)[0]*sqrt(y_scaler_11.var_[9]) + y_scaler_11.mean_[9]
    y_pred_std_Edens = predict_coregionalized(model, X_scaled, 3)[1]*sqrt(y_scaler_11.var_[9])
    y_pred_mu_CH4 = y_pred_mu_COMB - y_pred_mu_CO - y_pred_mu_H2
    y_pred_std_CH4 = additive_errorprop([y_pred_std_COMB, y_pred_std_CO, y_pred_std_H2])
    y_pred_mu_HHV = (11.76*y_pred_mu_CO + 11.882*y_pred_mu_H2 + 37.024*y_pred_mu_CH4)/100
    y_pred_std_HHV = additive_errorprop([11.76/100*y_pred_std_CO, 11.882/100*y_pred_std_H2, 37.024/100*y_pred_std_CH4])
    y_pred_mu_GAS = y_pred_mu_Edens / y_pred_mu_HHV
    y_pred_std_GAS = additive_errorprop([y_pred_std_Edens/y_pred_mu_Edens,
                                        y_pred_std_HHV/y_pred_mu_HHV])*y_pred_mu_GAS
    
    y_pred_mu_CGE = y_pred_mu_GAS * y_pred_mu_HHV / np.reshape(HHVbiom.values, (-1,1))*100
    y_pred_std_CGE = additive_errorprop([y_pred_std_GAS/y_pred_mu_GAS, 
                                        y_pred_std_HHV/y_pred_mu_HHV]) * y_pred_mu_GAS * y_pred_mu_HHV / np.reshape(
        HHVbiom.values, (-1,1))*100
    
    return y_pred_mu_CGE, y_pred_std_CGE




