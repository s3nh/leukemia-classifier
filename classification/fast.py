from typing import Optional 

from fastapi import FastAPI 
from onnx_inference import _session_run, _predict
from utils import read_config
from utils import load_image

app = FastAPI()

@app.get("/")
def read_root():
    config = read_config()
    sess = _session_run(config = config)
    
@app.get('/predict/')    
def get_predict(image: None):
    contents = await file.read()
    image = load_image(contents, width =450, height = 450) #Image.open(io.BytesIO(contents)).convert('RGB')
    pred = _predict(_sess = sess, image= image)
    return pred

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 6007)