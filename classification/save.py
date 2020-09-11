import torch 
import torchvision 
import pytorch_lightning as pl
from train import ClassificationTask 
from inference import load_model
from utils import read_config

def save_pt_model(path: str) -> None:
    """
    path: path to which .pt file will be exported 
    """
    model = load_model()
    torch.save(model.state_dict(), path) 
    
def main():
    config = read_config() 
    save_pt_model(path= config.get('torch_model_path'))
    
if __name__ == "__main__":
    main()
