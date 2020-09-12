import torch
import torchvision 
import typing 
from typing import Dict
from train import ClassificationTask
from utils import read_config

def load_model_pt(path: str) ->None:
    model = ClassificationTask()
    model.load_state_dict(torch.load(path))
    return model


def onnx_conversion(model: None, path : str, config: Dict) -> None:
    model.eval()
    print("Model eval succesfully")
    input_size = config.get('input_size')
    print(input_size)
    x = torch.randn(config.get('batch_size'), *config.get('input_size')[::-1], requires_grad=True)
    print(x.size)
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
    print(model)
    onnx_conversion(model, path = '.', config = config)
    
if __name__ == "__main__":
    main()