import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Potato Disease Classification")

st.title("🥔 Potato Disease Classification")
st.write("Upload a potato leaf image to predict the disease")

# ---- Load model ----
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potatoes.keras")

model = load_model()

# ⚠️ MUST MATCH TRAINING CLASS ORDER
class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___Healthy"
]

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # ---- Load image ----
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---- Preprocess (VERY IMPORTANT) ----
    img = image.resize((256, 256))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0                 # normalization
    img_array = np.expand_dims(img_array, axis=0) # (1,256,256,3)

    # ---- Prediction ----
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100

    # ---- Debug probabilities ----
    st.subheader("🔍 Prediction Probabilities")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {predictions[0][i]*100:.2f}%")

    # ---- Final output ----
    st.success(f"✅ Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
