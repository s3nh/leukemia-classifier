import torchvision.models 
import os 
import yaml

def get_config(path: str) -> None:
    assert os.path.exists(path), 'Path does not exist'
    with open(path, 'rb') as confile:
        config = yaml.safe_load(confile)
    return config 

def get_model(model_name, config ,**kwargs):
    """
    add require_grad

    """
    model =  config['model_name'](pretrained = kwagrs.get('pretrained'))
    model.fc.in_features 
    for param in model.parameters[-1]:
        param.requires_grad = False
    model.ft.fc = nn.Linear(num_ftrs, config['num_classes'])
    return model  

