import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Potato Disease Classifier", layout="centered")

st.title("🥔 Potato Disease Classification")
st.write("Upload a potato leaf image to predict the disease.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potatoes.keras")

model = load_model()

class_names = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

IMG_SIZE = 256

uploaded_file = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🧪 Prediction: **{predicted_class}**")
    st.info(f"🔍 Confidence: **{confidence:.2f}%**")
