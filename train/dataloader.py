import cv2
import numpy as np 
import os 
import torch
import torch.nn as nn

class Food101Data(DataLoader):

    def __init__(self, path : str, classes : Dict) -> None:
        self.path = path
        self.classes = classes
        self.files = os.listdir(self.path)

    def __getitem__(self, idx):
        
        img = self.files[idx]
        img = cv2.imread(img)
        _class =  self.classes.get(idx)
        return img, _class

    def __random__(self):
        raise NotImplemented


    def __len__(self):
        return len(self.files)


