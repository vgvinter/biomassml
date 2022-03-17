import hydra
from omegaconf import DictConfig, OmegaConf
from biomassml.pipeline import run_loocv_from_file


@hydra.main(config_path="conf", config_name="default")
def run_loocv(cfg: DictConfig):
    run_loocv_from_file(
        file=cfg.datafile,
        features=cfg.features,
        labels=cfg.labels,
        kernel=cfg.kernel,
        coregionalized=cfg.coregionalized,
        ard=cfg.ard,
        y_scramble=cfg.y_scramble,
    )
