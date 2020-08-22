import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torch.nn import Module 
import typing 

def _make_trainable(module: Module) -> None:
    """ Unfreezes a given module
    
    Arg:
        module: Module which you want to unfreeze 
    """
    for param in module.parameters():
        param.requires_grad = True
    module.train()

def _recursive_freeze(module : Module, 
        train_bn : bool =True) -> None:
    """ Layers freezer to make some layers nontrainable
        Args:
            module: Module which you want to freeze 
            train_bn: If you want Batch Norm to be trainable or not 
    """
    #Get list of model layers 
    children = list(module.children())
    if not children:
        if not (isinstance(module, BN_TYPES) and train_bn):
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        else:

            _make_trainable(module)
    else:
        for child in children:
            _recursive_freeze(module = child, train_bn = train_bn)


def freeze(module : Module, n: Optional[int] =  None, train_bn: bool = True) -> None:
    """

    :param module:  Module to freeze
    :param n:  max freeze depth
    :param train_bn:  if True, train on bn
    :return:
    """

    children = list(module.children())
    n_max = len(children) if n is None else n


    for child in children[:n_max]:
        _recursive_freeze(module = child, train_bn = train_bn)

    for child in children[nmax:]:
        _make_trainable(module=child)









