import onnx 
from utils import read_config


def conversion_check(path:str) -> None:

    model = onnx.load(path)
    checker = onnx.checker.check_model(model) 
    graph = onnx.helper.printable_graph(model.graph) 
    return checker, graph

def main():
    config = read_config()
    checker, graph = conversion_check(path = config.get('onnx_path'))    
    print(checker)
    print(graph) 
    
if __name__ == "__main__":
    main()
