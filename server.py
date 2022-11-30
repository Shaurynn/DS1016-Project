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

st.markdown("""
# Welcome to Atulya's Project
## "How to give my team-mates aneurysms"
""")

app = FastAPI()

@app.get('/index')
def ItsWorking():
    return "It's Working!"

st.cache
def predict_image(uploaded_file,uploaded_atlas):
    with open("./input.nii", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    with open("./ref_atlas.nii", "wb") as buffer:
        shutil.copyfileobj(ref_atlas.file, buffer)

    # preprocess("./input.nii", "./ref_atlas.nii")
    # saved_model = model_loading()
    prediction = predict()

    return {'prediction':1}


uploaded_file = st.file_uploader("Please uploade your MRI nifti file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

uploaded_atlas = st.file_uploader("Please upload your atlas nifti file")

if uploaded_atlas is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)


pred_ = predict_image(uploaded_file, uploaded_atlas)


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host='0.0.0.0')
