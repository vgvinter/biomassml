from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error


def get_regression_metrics(y_true, y_pred) -> dict:
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MaxError": max_error(y_true, y_pred),
    }
