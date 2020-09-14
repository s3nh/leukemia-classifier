import numpy as np 
import onnxruntime
import typing 

from typing import Dict
from utils import read_config

def _session_run(config : Dict, output: None = None) -> None:
    ort_session = onnxruntime.InferenceSession(config.get('onnx_path'))
    if output is None:
        outputs = ort_session.run(None, {'input': np.random.randn(1, 3, 450, 450).astype(np.float32)})
    else:
        raise NotImplemented   
    print(outputs)
    
def main():
    config = read_config()
    _session_run(config = config, output = None)
    
if __name__ == "__main__":
    main()
    