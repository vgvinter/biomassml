from .build_model import ARD_WRAPPERS, NO_ARD_WRAPPERS, build_coregionalized_model, build_model


def loocv_pipe(kernel, coregionalized: bool = False, ard: bool = False)