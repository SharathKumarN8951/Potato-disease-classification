import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- Page config ----------------
st.set_page_config(
    page_title="Potato Disease Classification",
    layout="centered"
)

st.title("🥔 Potato Disease Classification")
st.write("Upload a potato leaf image")

# ---------------- Thresholds ----------------
LOW_CONFIDENCE_THRESHOLD = 60.0
HIGH_CONFIDENCE_THRESHOLD = 95.0
ENTROPY_THRESHOLD = 1.0

# ---------------- Utility functions ----------------
def prediction_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))


# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potatoes.keras", compile=False)

model = load_model()

class_names = [
    "Potato__Early_blight",
    "Potato__Late_blight",
    "Potato__healthy"
]

# ---------------- File uploader ----------------
uploaded_file = st.file_uploader(
    "Choose an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------------- Prediction ----------------
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100
    entropy = prediction_entropy(predictions)

    # ---------------- NON-LEAF DECISION ----------------
    is_non_leaf = (
        confidence > HIGH_CONFIDENCE_THRESHOLD and entropy < 0.6
    ) or (
        confidence < LOW_CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD
    )

    if is_non_leaf:
        st.error("❌ **Prediction: NON-LEAF IMAGE**")
        st.info(
            "This image does not appear to be a potato leaf.\n\n"
            "👉 Please upload a clear potato leaf image."
        )

        st.subheader("🔍 Model Output (for debugging)")
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {predictions[i] * 100:.2f}%")

    else:
        st.subheader("🔍 Prediction Probabilities")
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {predictions[i] * 100:.2f}%")

        st.success(f"✅ Prediction: **{predicted_class}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")
