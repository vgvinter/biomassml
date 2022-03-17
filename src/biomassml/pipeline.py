from .build_model import ARD_WRAPPERS, NO_ARD_WRAPPERS, build_coregionalized_model, build_model
from .metrics import loocv_train_test
from .utils import get_timestring
from .io import dump_pickle
import pandas as pd


def loocv_pipe(kernel, X, y, coregionalized: bool = False, ard: bool = False):
    """
    Perform leave-one-out cross-validation on the given kernel.
    """
    time = get_timestring()
    if ard:
        if coregionalized:
            model = build_coregionalized_model(X, y, kernel=ARD_WRAPPERS[kernel](X))
        else:
            model = build_model(X, y, kernel=ARD_WRAPPERS[kernel](X))
    else:
        if coregionalized:
            model = build_coregionalized_model(X, y, kernel=NO_ARD_WRAPPERS[kernel](X))
        else:
            model = build_model(X, y, kernel=NO_ARD_WRAPPERS[kernel](X))

    result = loocv_train_test(model, X, y, coregionalized=coregionalized)

    dump_pickle(result, f"loocv_{time}_{kernel}_{coregionalized}_{ard}.pkl")


def run_loocv_from_file(file, features, labels, kernel, coregionalized, ard):
    df = pd.read_csv(file)
    X = df[features].values
    y = df[labels].values
    loocv_pipe(kernel, X, y, coregionalized, ard)
