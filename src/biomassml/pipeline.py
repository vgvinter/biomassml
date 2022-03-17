from .build_model import ARD_WRAPPERS, NO_ARD_WRAPPERS, build_coregionalized_model, build_model
from .metrics import loocv_train_test


def loocv_pipe(kernel, X, y, coregionalized: bool = False, ard: bool = False):
    """
    Perform leave-one-out cross-validation on the given kernel.
    """
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
