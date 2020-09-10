from train import ClassificationTask
from utils import read_config

def load_model(**kwargs) -> None:
    config = read_config() 
    model = ClassificationTask.load_from_checkpoint(config.get('trained_path'))
    return model
           
def main():
    model =  load_model()
    print(model)

if __name__ == "__main__":
    main()
