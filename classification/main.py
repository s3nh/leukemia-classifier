import os 
from data import ClassificationTaskDataModule
from train import ClassificationTask
import pytorch_lightning as pl
import torch 
import torchvision
import typing 
from typing import Dict
import yaml 

def read_config(path : str =  'config/config.yaml'):
    with open(path, 'r') as confile:
        config = yaml.safe_load(confile)
    return config

def main():
    """
    Call for config,
    Call for Classification task 
    then set pytorch lightning trainer 
    """
    path = 'config/config.yaml'
    #assert os.path.exists(path), 'Path does not exist!' 
    config = read_config()
    
    #Based on config info
    data = ClassificationTaskDataModule(config_path = path)
    print(data.batch_size)
    model = ClassificationTask(config_path = path)
    
    trainer = pl.Trainer(
        weights_summary = None, 
        progress_bar_refresh_rate=1, 
        num_sanity_val_steps= 0, 
        gpus = config.get('gpus'), 
        min_epochs = config.get('np_epochs'), 
        max_epochs = config.get('nb_epochs'),
    ) 
    
    trainer.fit(model, data) 

if __name__ == "__main__":
    main()