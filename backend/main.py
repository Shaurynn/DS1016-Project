import uuid
import shutil
import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile, File
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import config
from backend.deeplearn import preprocess, predict


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post('/{model}')
def predict_image(model: str, file: UploadFile = File(...), ref_atlas: UploadFile = File(...)):
    model = config.MODELS[model]
    with open("./input.nii", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    with open("./ref_atlas.nii", "wb") as buffer:
        shutil.copyfileobj(ref_atlas.file, buffer)

    prediction = predict()

    return {'prediction':1}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
