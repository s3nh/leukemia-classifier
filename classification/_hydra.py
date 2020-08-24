import logging
import hydra

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from train import ClassificationTask

#Took from simple example
@hydra.main(config_path : str = 'config/config.yaml')
def hydraNT(cfg : DictConfig) -> Trainer:
    """
    Describe it more properly
    TODO make a concept of config.yaml file, dependent of different archs and
    TODO more complex dataset
    :param cfg:
    :return:
    """
    pass
    logger.info(f"Training with : {cfg.pretty}")


    module = ClassificationTask(cfg.network, cfg.train)
    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = Trainer(**cfg.pl_trainer, logger = trainer_logger)
    trainer.fit(
        module,
        train_dataloader = DataLoader(

        )

    )











