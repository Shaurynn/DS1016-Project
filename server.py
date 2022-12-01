from fastapi import FastAPI
from fastapi import UploadFile, File
import uvicorn
import shutil
import streamlit as st
import tensorflow as tf
import os
import gzip
import tarfile
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.callbacks import EarlyStopping

from prediction import preprocess, predict

MODELS = {
    "Inception": "models/inception_model1.h5",
    "ResNet3D": "models/inception_model1.h5",
}

st.title("Alzheimer's Disease Detection")

image = st.file_uploader("Choose an image")
atlas = st.file_uploader("Choose an atlas")

# displays the select widget for the styles
model = st.selectbox("Choose the deep-learning model", [i for i in MODELS.keys()])

if st.button("Upload"):
    if image is not None and atlas is not None and model is not None:
        with open(os.path.join("./",image.name),"wb") as f:
            f.write(image.getbuffer())
        with open(os.path.join("./", atlas.name),"wb") as a:
            a.write(atlas.getbuffer())
    st.success("Files saved")

def detect_AD():
    preprocess(os.path.join("./",image.name),os.path.join("./", atlas.name))
    #saved_model = os.path.join("./",model)
    prediction = predict(f"./{image.name}_2d.npy")

trigger = st.button('Predict', on_click=detect_AD)
