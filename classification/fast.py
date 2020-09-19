import json
import numpy as np 
from typing import Optional 
from fastapi import FastAPI, File, UploadFile, HTTPException
from onnx_inference import _session_run, _predict
from utils import read_config
from utils import load_image, load_contents

app = FastAPI()

config = read_config()
sess = _session_run(config = config)

@app.post('/predict/')    
async def get_predict(file : UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail = f'File \'{file.filename}\' is not an image')
    contents = await file.read()
    image = load_contents(contents= contents)
    print(image.shape)
    #image = load_image(contents, width =450, height = 450) 
    pred = _predict(_sess = sess, config = config, image= image)
    print(pred[0][0])
    return {"predict" : str(1/(1+ np.exp(-pred[0][1])))}

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 6007)