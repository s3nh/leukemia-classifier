import albumentations as A
import argparse
from collections import OrderedDict 
from pathlib import Path 
from typing import Optional, Generator, Union, Dict

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import optim 
from torch.nn import Module 

from torch.optim.lr_scheduler import MultiStepLR 
from torch.optim.optimizer import Optimizer 
from torch.utils.data import DataLoader

from torchvision import models 
from torchvision import transforms 
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from pytorch_lightning import _logger as log 

from utils import _make_trainable, freeze
from utils import predefined_transform

import yaml

class ClassificationTask(pl.LightningModule):
    def __init__(self,
                 transform = predefined_transform(),
                 config_path : str = 'config/config.yaml',
                 **kwargs) -> None:
        super().__init__()
        self.config = self.read_config(path=config_path)
        self.batch_size = self.config.get('batch_size');
        self.backbone = self.config.get('backbone');
        self.train_bn = self.config.get('train_bn');
        self.lr = self.config.get('lr');
        self.lr_scheduler_gamma = self.config.get('lr_scheduler_gamma');
        self.num_workers = self.config.get('num_workers');
        self.n_classes = self.config.get('n_classes');
        self.transform = transform
        self.__build_model()

    def __build_model(self):
        model_func = getattr(models, self.backbone);
        backbone = model_func(pretrained=True)
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*_layers)
        freeze(module = self.feature_extractor, train_bn = self.train_bn)

        _fc_layers = [torch.nn.Linear(self.config.get('FC1'), self.config.get('FC2')),
                      torch.nn.Linear(self.config.get('FC2'), self.config.get('FC3')),
                      torch.nn.Linear(self.config.get('FC3'), self.n_classes)]

        self.fc = torch.nn.Sequential(*_fc_layers)
        self.loss_func = nn.CrossEntropyLoss

    def setup(self, stage: str):
        train_dataset = ImageFolder(root = self.config.get('train'),
                                transform = self.transform
                                )

        valid_dataset = ImageFolder(root = self.config.get('validation'),
                                transform = self.transform
                                )
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        
    def forward(self, x):
        """
        :param x: input data
        :return: logits
        """
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x

    def loss(self, logits, labels):
        """
        :param logits: output from forward pass
        :param labels:  predefined labels
        :return: loss function value
        """
        return self.loss_func(input = logits, target = labels)

    def train(self, mode=True):
        super().train(mode=mode)

    def training_step(self, batch, batch_idx):
        x,y = batch
        y_logits = self.forward(x)
        y_pred = torch.argmax( nn.Softmax(y_logits))
        y_true = y.view((-1, 1)).type_as(x)

        train_loss = self.loss(y_logits, y_true)
        num_correct = torch.eq(y_pred.view(-1), y_true.view(-1)).sum()

        tqdm_dict = {'train_loss' : train_loss}
        output = OrderedDict({'loss' : train_loss,
                              'num_correct' : num_correct,
                              'log' : tqdm_dict,
                              'progress_bar' : tqdm_dict})
        return output

    def validation_step(self, batch, batch_idx):
        """
        Input processing, loss formating 
        and forward process could be also involved 
        in other functions, 
        to set validation/training parts more readable 
        
        """
        x, y = batch
        val_logits = self.forward(x)
        val_pred = torch.argmax(nn.Softmax(y_logits)) 
        val_true = y.view((-1, 1)).type_as(x)
        
        val_loss = self.loss(val_logits, val_true)  
        num_val_correct = torch.eq(val_pred.view(-1), val_true(-1)).sum() 
        tqdm_dict = {'val_loss' : val_loss}
        output = OrderedDict({'val_loss' : val_loss, 
                              'num_correct' : num_val_correct, 
                               'log' : tqdm_dict, 
                               'progress_bar' : tqdm_dict }) 

        return output
        
    def configure_optimizers(self):
        """
        #TODO 
        What if we want to change optimizers? 
        """
        optimizer = optim.Adam(filter(lambda p : p.requires_grad,
                                      self.parameters()),
                                     lr = self.lr)
        scheduler = MultiStepLR(optimizer,
                                milestones = self.config.get('milestones'), 
                                gamma = self.lr_scheduler_gamma)
        return [optimizer], [scheduler]

    """
    Comment, as long as data.py is not tested  

    def __dataloader(self, train : bool) -> None:
        dataset = self.train_dataset if train else self.valid_dataset
        dataloader = DataLoader(dataset = dataset,
                                batch_size = self.batch_size,
                                shuffle = True if train else False)
        return dataloader

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False);
    """
    
    def read_config(self, path : str) -> Dict:
        with open(path, 'r') as confile:
            config = yaml.safe_load(confile)
        return config

    @staticmethod
    def add_model_specific_args(config):
        """
        self.config should make things done
        :param config:
        :return:
        """
        pass

def main() -> None:
    """
    Model training process
    """
    with open('config/config.yaml', 'r') as confile:
        config = yaml.safe_load(confile)

    model = ClassificationTask() 
    trainer = pl.Trainer(
            weights_summary = None, 
            progress_bar_refresh_rate = 1, 
            num_sanity_val_steps = 0, 
            gpus = config.get('gpus'), 
            min_epochs = config.get('nb_epochs'), 
            max_epochs = config.get('nb_epochs'))

    trainer.fit(model)  

if __name__ == "__main__":
    main()