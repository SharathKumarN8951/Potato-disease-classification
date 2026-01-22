import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Potato Disease Classification")

st.title("🥔 Potato Disease Classification")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potatoes.keras")

model = load_model()

# ✅ MUST MATCH TRAINING DATA EXACTLY
class_names = [
    "Potato__Early_blight",
    "Potato__Late_blight",
    "Potato__healthy"
]

uploaded_file = st.file_uploader(
    "Upload a potato leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((256, 256))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    st.subheader("🔍 Prediction Probabilities")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {predictions[i]*100:.2f}%")

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    st.success(f"✅ Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
