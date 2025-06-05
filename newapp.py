import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Soybean Leaf Classifier", layout="centered")
st.title("🌱 Soybean Leaf: Healthy or Unhealthy?")

@st.cache_resource
def load_model_from_h5():
    # Load the full Keras model (architecture + weights)
    return tf.keras.models.load_model("best_model.h5")

model = load_model_from_h5()
CLASS_NAMES = ["Healthy", "Unhealthy"]

uploaded_file = st.sidebar.file_uploader(
    "Upload a soybean leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Preprocess just like during training
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

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
