import hydra
from omegaconf import DictConfig, OmegaConf
from biomassml.pipeline import loocv_pipe

@hydra.main(config_path="conf", config_name="default"):
def run_loocv(cfg: DictConfig):
    loocv_pipe(**cfg)
