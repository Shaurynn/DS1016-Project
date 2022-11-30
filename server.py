from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn
import shutil
import streamlit as st

import shutil
import tensorflow as tf
import os
import gzip
import tarfile

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping

from prediction import preprocess, model_loading, predict


app = FastAPI()

@app.get('/index')
def ItsWorking():
    return "It's Working!"

@app.post('/api/predict')
def predict_image(file: UploadFile = File(...), ref_atlas: UploadFile = File(...)):
    with open("./input.nii", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    with open("./ref_atlas.nii", "wb") as buffer:
        shutil.copyfileobj(ref_atlas.file, buffer)

    # preprocess("./input.nii", "./ref_atlas.nii")
    # saved_model = model_loading()
    prediction = predict()

    return {'prediction':1}


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
