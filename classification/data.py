import pytorch_lightning as pl
from utils import predefined_transform
from typing import Dict
import yaml

# What I dont like is that config has to be repeated twice in that configuration
# It is no longer possible to 
class ClassificationTaskDataModule(pl.LightningDataModule):
    def __init__(self, config_path = 'config/config.yaml'):
        self.config = self.read_config(path = config_path)    
        self.transform = predefined_transform()
        self.batch_size = self.config.get('batch_size')
    
    def read_config(self, path : str) -> Dict:
        with open(path, 'r') as confile:
            config = yaml.safe_load(confile)
        return config
    
    def prepare_data(self):
         train_dataset = ImageFolder(root = self.config.get('train'), 
                                     )
         valid_dataset = ImageFolder(root = self.config.get('validation'), 
                                     )
         

    def setup(self, stage: str):
        train_dataset = ImageFolder(root = self.config.get('train'),
                                transform = self.transform
                                )
        valid_dataset = ImageFolder(root = self.config.get('validation'),
                                transform = self.transform
                                )
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        
    def __dataloader(self, train : bool) -> None:
        dataset = self.train_dataset if train else self.valid_dataset
        dataloader = DataLoader(dataset = dataset,
                                batch_size = self.batch_size,
                                shuffle = True if train else False)
        return dataloader

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)
    