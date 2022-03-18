from .build_model import ARD_WRAPPERS, NO_ARD_WRAPPERS, build_coregionalized_model, build_model
from .metrics import loocv_train_test
from .utils import get_timestring
from .io import dump_pickle
import pandas as pd
from loguru import logger
import wandb


def loocv_pipe(
    kernel,
    X,
    y,
    features,
    coregionalized: bool = False,
    ard: bool = False,
    y_scramble: bool = False,
    tags: list = None,
):
    """
    Perform leave-one-out cross-validation on the given kernel.
    """
    run = wandb.init(project="biomassml", tags=tags)
    run.config.update(
        {"kernel": kernel, "coregionalized": coregionalized, "ard": ard, "y_scramble": y_scramble}
    )
    time = get_timestring()
    if ard:
        if coregionalized:
            model = build_coregionalized_model(X, y, kernel=ARD_WRAPPERS[kernel](X.shape[1]))
        else:
            model = build_model(X, y, kernel=ARD_WRAPPERS[kernel](X.shape[1]))
    else:
        if coregionalized:
            model = build_coregionalized_model(X, y, kernel=NO_ARD_WRAPPERS[kernel](X.shape[1]))
        else:
            model = build_model(X, y, kernel=NO_ARD_WRAPPERS[kernel](X.shape[1]))

    result = loocv_train_test(model, X, y, coregionalized=coregionalized)

    metrics = result[0]
    flat_metrics = {}

    for objectve, metric_dict in metrics.items():
        for metric, value in metric_dict.items():
            flat_metrics[f"{metric}_{objectve}"] = value

    wandb.log(flat_metrics)
    dump_pickle(result, f"loocv_{time}_{kernel}_{coregionalized}_{ard}.pkl")


def run_loocv_from_file(file, features, labels, kernel, coregionalized, ard, y_scramble):
    logger.info(f"Reading {file}")
    df = pd.read_csv(file)
    X = df[features].values
    y = df[labels].values
    if y_scramble:
        y = df[labels].sample(len(df)).values
    logger.info(f"Feature shape: {X.shape}, label shape {y.shape}")
    loocv_pipe(kernel, X, y, features, coregionalized, ard, y_scramble)
