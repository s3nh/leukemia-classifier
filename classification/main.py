import os 
from data import ClassificationTaskDataModule
from train import ClassificationTask
import pytorch_lightning as pl
import torch 
import torchvision
import typing 
from typing import Dict
import yaml 

from pytorch_lightning.callbacks import ModelCheckpoint
from utils import customized_callbacks

def read_config(path : str =  'config/config.yaml') -> Dict:
    with open(path, 'r') as confile:
        config = yaml.safe_load(confile)
    return config

def main():
    path = 'config/config.yaml'
    config = read_config()
    data = ClassificationTaskDataModule(config_path = path)
    model = ClassificationTask(config_path = path)
    callbacks = customized_callbacks()
    trainer = pl.Trainer(
        #weights_summary = None, 
        progress_bar_refresh_rate=10, 
        num_sanity_val_steps= 0, 
        gpus = config.get('gpus'), 
        min_epochs = 0, #config.get('np_epochs'), 
        max_epochs = config.get('nb_epochs'),
        checkpoint_callback=  callbacks
    ) 
    
    trainer.fit(model, data) 


if __name__ == "__main__":
    main()