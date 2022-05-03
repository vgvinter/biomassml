from sklearn.preprocessing import StandardScaler
from .predict import predict_coregionalized


def get_scalers(X, y):
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    return x_scaler, y_scaler


def scale_output(y_pred_mu, y_pred_std, y_scaler=y_scaler_11, n=0):
    y_pred_mu_scaled = ((y_pred_mu - y_scaler.mean_[n])/sqrt(y_scaler.var_[n])).flatten() 
    y_pred_std_scaled = (y_pred_std/sqrt(y_scaler.var_[n])).flatten()
    return y_pred_mu_scaled, y_pred_std_scaled


def predict_CO(X, model=model, x_scaler=x_scaler_12, y_scaler=y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_CO = predict_coregionalized(model, X_scaled, 0)[0]*sqrt(y_scaler_11.var_[1]) + y_scaler_11.mean_[1]
    y_pred_std_CO = predict_coregionalized(model, X_scaled, 0)[1]*sqrt(y_scaler_11.var_[1])
    return y_pred_mu_CO, y_pred_std_CO 


def predict_H2(X, model=model, x_scaler=x_scaler_12, y_scaler=y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_H2 = predict_coregionalized(model, X_scaled, 1)[0]*sqrt(y_scaler_11.var_[3]) + y_scaler_11.mean_[3]
    y_pred_std_H2 = predict_coregionalized(model, X_scaled, 1)[1]*sqrt(y_scaler_11.var_[3])
    return y_pred_mu_H2, y_pred_std_H2  .


def predict_COMB(X, model=model, x_scaler=x_scaler_12, y_scaler=y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_COMB = predict_coregionalized(model, X_scaled, 2)[0]*sqrt(y_scaler_11.var_[4]) + y_scaler_11.mean_[4]
    y_pred_std_COMB = predict_coregionalized(model, X_scaled, 2)[1]*sqrt(y_scaler_11.var_[4])
    return y_pred_mu_COMB, y_pred_std_COMB   


def predict_Edensity(X, model=model, x_scaler=x_scaler_12, y_scaler=y_scaler_11):
    '''It returns unscaled predictions'''
    X_scaled = x_scaler.transform(X)
    y_pred_mu_Edens = predict_coregionalized(model, X_scaled, 3)[0]*sqrt(y_scaler_11.var_[9]) + y_scaler_11.mean_[9]
    y_pred_std_Edens = predict_coregionalized(model, X_scaled, 3)[1]*sqrt(y_scaler_11.var_[9])
    return y_pred_mu_Edens, y_pred_std_Edens


def predict_CH4(X, model=model, x_scaler=x_scaler_12, y_scaler=y_scaler_11):
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


