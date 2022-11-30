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

def predict():
    preprocess(image, atlas)
    saved_model = model
    prediction = predict()

    return {'prediction':1}

trigger = st.button('Predict', on_click=predict)
