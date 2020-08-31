import albumentations as A
import argparse
from collection import OrderedDict 
from pathlib import Path 
from typing import Optional, Generator, Union

import torch 
import torch.nn.functional as F
from torch import optim 
from torch.nn import Module 

# Optim section 

from torch.optim.lr_scheduler import MultiStepLR 
from torch.optim.optimizer import Optimizer 
from torch.utils.data import DataLoader

# Torchvision 

from torchvision import models 
from torchvision import transforms 

import pytorch_lightning as pl
from pytorch_lightning import _logger as log 

from utils import make_traineble, freeze
from utils import predefined_transform

"""
Define an classification module 
for pretrained structures, based on 
configuration files
"""

class ClassificationTask(pl.LightningModul):
    def __init__(self,
                 backbone: str = 'resnet50',
                 train_bn: bool = True,
                 batchsize: int = 8,
                 lr: float = 1e-4,
                 lr_scheduler_gamma: float = 1e-1,
                 num_workers: int  = -1,
                 n_classes : int  = 10,
                 transform = predefined_transform(),
                 config_path : str = 'config/config.yaml',
                 #TODO Kwargs are not necessary imo
                 **kwargs) -> None:
        super().__init__()
        self.backbone = backbone;
        self.train_bn = train_bn;
        self.lr = lr;
        self.lr_scheduler_gamma = lr_scheduler_gamma;
        self.num_workers = num_workers;
        self.config = self.read_config(path=config_path)
        #Change naming, transform seems to be very error creating
        self.transform = transform
        self.__build_model()

    def __build_model(self):
        # Sneaky
        model_func = getattr(models, self.backbone);
        backbone = model_func(pretrained=True)
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = torch.nn.Sequential(*_layers)
        freeze(module = self.feature_extractor, train_bn = self.train_bn)

        _fc_layers = [torch.nn.Linear(2048, 256),
                      torch.nn.Linear(256, 32),
                      torch.nn.Linear(32, self.n_classes)]

        self.fc = torch.nn.Sequential(*_fc_layers)
        self.loss_func = nn.CrossEntropyLoss

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

# Add valid steps
    def configure_optimizers(self):
        optimizer = optim.Adam(filter(lambda p : p.requires_grad,
                                      self.parameters()),
                                     lr = self.lr)

        scheduler = MultiStepLR(optimizer,
                                gamma = self.lr_scheduler_gamma)

        return [optimizer], [scheduler]

    """
    def transform(self):
        return predefined_transform()
    """
    # Assuming well prepared dataset
    # like imagefolder 

    def setup(self, stage: str):
        data_path = Path(self.dl_path)
        train_dataset = ImageFolder(root = data_path.join('train'),
                                   transform = self.transform
                                   )

        valid_dataset = ImageFolder(root = data_path.join('validation'),
                                    transform = self.transform
                                    )
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

    def __dataloader(self, train : bool) -> None:

        dataset = self.train_dataset if train else self.valid_dataset;
        dataloader = DataLoader(dataset = dataset,
                                batch_size = self.batch_size,
                                shuffle = True if train else False)

        return dataloader

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False);

    @staticmethod 
    def read_config(self, path : str) -> Dict:
        with open(config_path, 'r') as confile:
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
        # add args here
        model = ClassificationTask() 
        
        trainer = pl.Trainer(
                weights_summary = None, 
                progress_bar_refresh_rate = 1, 
                num_sanity_val_steps = 0, 
                gpu = config.gpus, 
                min_epochs = config.nb_epochs, 
                max_epochs = config.nv_epochs)

        trainer.fit(model)  
