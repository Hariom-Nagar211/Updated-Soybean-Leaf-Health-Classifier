# app.py

import streamlit as st
import subprocess
import sys
import os
import numpy as np
from PIL import Image

st.set_page_config(page_title="Soybean Leaf Classifier", layout="centered")
st.title("🌱 Soybean Leaf: Healthy or Unhealthy?")

@st.cache_resource
def ensure_tf():
    """
    Ensure TensorFlow is installed. If not, pip-install it at runtime.
    We pin tensorflow-cpu==2.10.0 to match the trained model.
    """
    try:
        import tensorflow as tf
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "tensorflow-cpu==2.10.0"]
        )
        import tensorflow as tf
    return tf

# Obtain TensorFlow in the current session
tf = ensure_tf()

@st.cache_resource
def load_model_from_h5():
    """
    Load the trained Keras model from best_model.h5.
    """
    model_path = "best_model.h5"
    if not os.path.exists(model_path):
        st.error("Model file best_model.h5 not found in the repo.")
        st.stop()
    return tf.keras.models.load_model(model_path)

model = load_model_from_h5()
CLASS_NAMES = ["Healthy", "Unhealthy"]

uploaded_file = st.sidebar.file_uploader(
    "Upload a soybean leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds[0])
    confidence = preds[0][pred_idx]

    st.subheader(f"Prediction: **{CLASS_NAMES[pred_idx]}**")
    st.write(f"Confidence: {confidence * 100:.2f}%")

    if CLASS_NAMES[pred_idx] == "Unhealthy":
        st.warning("⚠️ The leaf appears Unhealthy.")
    else:
        st.success("✅ The leaf appears Healthy.")
else:
    st.write("Please upload a JPG/PNG image from the sidebar.")
