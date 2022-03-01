import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf/vime", config_name="config.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from biomassml.trainers.vime_trainer import train

    return train(config)

if __name__ == "__main__":
    main()