import torch
import torchvision 
from train import ClassificationTask
from utils import read_config

# TODO - standarize

def load_model_pt(path: str) ->None:
    model = ClassificationTask()
    return model.load_state_dict(torch.load(path))

    
def onnx_conversion(path : str) -> None:

    return  torch.onnx.export(
        #x is an input <- ffs
        x, 
        'leukemia_resnet50.onnx', 
        export_params=True, 
        # TODO 
        #input names
        #optimization part 
    )
    

def main():
    config = read_config()
    model=load_model_pt(path = config.get('torch_model_path')) 
        
if __name__ == "__main__":
    main()