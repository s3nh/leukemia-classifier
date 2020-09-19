import numpy as np 
import onnxruntime
import typing 

from typing import Dict
from utils import read_config
# Why not utils?


def _session_run(config : Dict) -> None:
    ort_session = onnxruntime.InferenceSession(config.get('onnx_path'))
    return ort_session

def _predict(_sess : None, config: Dict, image: None) -> None:
    input_name: str = _sess.get_inputs()[0].name
    output_name: str = _sess.get_outputs()[0].name
    proba: np.ndarray = _sess.run([output_name], {input_name: image})[0]
    return proba 

    