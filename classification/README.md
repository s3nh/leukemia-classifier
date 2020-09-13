#### Classification pipeline 


Contains elements of classification pipeline, 
assuming that we are using pretrained structures. 


- Pytorch-lightning 
- Pytorch 

##### Model training process

- 20200901 

to train the model, just pass 

``` 
python train.py 
``` 

with properly defined config file


```yaml

nb_epoch: Number of epohs
gpu: number of gpus used
backbones: backbone architecture (at the moment resnet50 is checked)
train_bn: if you want to train BatchNorm
train: train dataset path
validation: validation dataset path (assuming that we used torchvision.dataset.ImageFolder structure)
batch_size:
lr: Learning Rate
lr_scheduler_gamma: 
num_workers: 
n_classes: number of classes 
FC1: number of neurons in 1st trainable layer
FC2 number of neurons  in 2nd trainable layer
FC3: number of neurons in 3rd trainable layer

```


