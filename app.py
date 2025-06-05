import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import json
import os

# --- 1. Page Config ---
st.set_page_config(
    page_title="Soybean Leaf Classifier",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="auto"
)


# --- 2. Load Model & Class Indices once ---
@st.cache_resource(show_spinner=True)
def load_trained_model(model_path: str):
    model = load_model(model_path)
    return model


@st.cache_data(show_spinner=True)
def load_class_indices(json_path: str):
    with open(json_path, "r") as f:
        class_indices = json.load(f)
    # invert mapping: {0: 'Healthy', 1: 'Unhealthy'}
    inv_map = {v: k for k, v in class_indices.items()}
    return class_indices, inv_map


# Paths (assume model & JSON are in same directory as this .py file)
MODEL_PATH = os.path.join(os.getcwd(), "soybean_model.h5")
CLASS_INDICES_PATH = os.path.join(os.getcwd(), "class_indices.json")

model = load_trained_model(MODEL_PATH)
class_indices, inv_map = load_class_indices(CLASS_INDICES_PATH)

# --- 3. Title & Description ---
st.title("🌱 Soybean Leaf Health Classifier")
st.write(
    """
    Upload an image of a soybean leaf, and this app will tell you whether 
    it appears **Healthy** or **Unhealthy**, based on a transfer‑learning model
    trained on Soybean Leaf images (MobileNetV2 + custom head).
    """
)

# --- 4. File Uploader ---
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess image
    IMG_SIZE = (224, 224)
    img = image.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # MobileNetV2 preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

    # Predict
    preds = model.predict(img_array)
    score = preds[0][0]

    # Because we used sigmoid activation, score ∈ [0,1]
    if score < 0.5:
        pred_label = [k for k, v in class_indices.items() if v == 0][0]  # 'Healthy'
        confidence = 1 - score
    else:
        pred_label = [k for k, v in class_indices.items() if v == 1][0]  # 'Unhealthy'
        confidence = score

    st.success(f"Prediction: **{pred_label}**")
    # st.write(f"Confidence: {confidence * 100:.2f}%")
