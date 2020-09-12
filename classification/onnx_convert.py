import torch
import torchvision 
import typing 
from typing import Dict
from train import ClassificationTask
from utils import read_config
#TODO ClassificationTask should be included as param  and getattr to argument

def load_model_pt(path: str) ->None:
    model = ClassificationTask()
    model.load_state_dict(torch.load(path))
    return model

def onnx_conversion(model: None, config: Dict) -> None:
    """
    Args:
        model: trained model, in this case .pt extension with predefined Network Architecture
        config: needed for properly defined output path / input_size / batch_size parameters 
       
    Return: converted .onnx model in indicated path
    """
    model.eval()
    input_size = config.get('input_size')
    x = torch.randn(config.get('batch_size'), *config.get('input_size')[::-1], requires_grad=True)
    torch_out = model(x)
    print(torch_out)
    torch.onnx.export(
        model,
        x, 
        'leukemia_resnet50.onnx', 
        export_params=True, 
        opset_version = 10, 
        do_constant_folding= True,
        input_names = ['input'], 
        output_names = ['output'], 
        dynamic_axes={'input' : {0 : 'batch_size'}, 
                      'output' : {0 : 'batch_size'}}
    )

def main():
    config = read_config()
    model=load_model_pt(path = config.get('torch_model_path')) 
    onnx_conversion(model, path = '.', config = config)
    
if __name__ == "__main__":
    main()