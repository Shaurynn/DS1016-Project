import requests
import streamlit as st
from PIL import Image

MODELS = {
    "Inception": "inception_model1",
    "ResNet3D": "resnet_3d_model1",
}

st.title("Alzheimer's Disease Dectection")

image = st.file_uploader("Choose an image")
atlas = st.file_uploader("Choose an atlas")

# displays the select widget for the styles
model = st.selectbox("Choose the deep-learning model", [i for i in MODELS.keys()])


if st.button("Upload"):
    if image is not None and atlas is not None and model is not None:
        files = {"model": model.getvalue(), "file": image.getvalue(), "ref_atlas": atlas.getvalue()}
        res = requests.post(f"http://backend:8080/{model}", files=files)
        img_path = res.json()
        image = Image.open(img_path.get("name"))
        st.image(image, width=500)
