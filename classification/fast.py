from typing import Optional 

from fastapi import FastAPI 
from onnx_inference import _session_run, _predict
from utils import read_config
from utils import load_image

app = FastAPI()

config = read_config()
sess = _session_run(config = config)
print(sess)


@app.get('/predict/')    
async def get_predict():
    contents = await file.read()
    image = load_image(contents, width =450, height = 450) 
    pred = _predict(_sess = sess, config = config, image= image)
    return pred

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 6007)