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
st.write("Upload a potato leaf image to predict the disease")

# ---------------- Thresholds ----------------
CONFIDENCE_THRESHOLD = 70.0   # % (tune between 60–80)
ENTROPY_THRESHOLD = 0.8       # lower = confident, higher = uncertain

# ---------------- Utility function ----------------
def prediction_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potatoes.keras", compile=False)

model = load_model()

# ⚠️ MUST MATCH raw_train_ds.class_names EXACTLY
class_names = [
    "Potato__Early_blight",
    "Potato__Late_blight",
    "Potato__healthy"
]

# ---------------- File uploader ----------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load and show image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------------- Preprocessing ----------------
    # IMPORTANT:
    # No resize / no normalization here
    # (Handled inside the model)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # (1,H,W,3)

    # ---------------- Prediction ----------------
    predictions = model.predict(img_array)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100
    entropy = prediction_entropy(predictions)

    # ---------------- Validation logic ----------------
    if confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
        st.error(
            "❌ **Invalid or unsupported image**\n\n"
            "Please upload a **clear potato leaf image**.\n\n"
            "Supported classes:\n"
            "- Early Blight\n"
            "- Late Blight\n"
            "- Healthy Leaf"
        )
        st.info(
            f"ℹ️ Model confidence: {confidence:.2f}%\n\n"
            f"ℹ️ Prediction uncertainty (entropy): {entropy:.2f}"
        )
    else:
        st.subheader("🔍 Prediction Probabilities")
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {predictions[i] * 100:.2f}%")

        st.success(f"✅ Prediction: **{predicted_class}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")
