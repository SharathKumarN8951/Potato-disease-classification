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
st.write("Upload a **clear potato leaf image** to predict the disease")

# ---------------- Thresholds ----------------
CONFIDENCE_THRESHOLD = 70.0   # %
ENTROPY_THRESHOLD = 1.2       # uncertainty

# ---------------- Utility ----------------
def prediction_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potatoes.keras", compile=False)

model = load_model()

# ⚠️ MUST MATCH TRAINING ORDER EXACTLY
class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# ---------------- File uploader ----------------
uploaded_file = st.file_uploader(
    "Choose an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------------- Model inference ----------------
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100
    entropy = prediction_entropy(predictions)

    # ---------------- NON-LEAF DECISION ----------------
    if confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
        st.error("❌ **Prediction: NON-LEAF IMAGE**")
        st.info(
            "This image does not appear to be a potato leaf.\n\n"
            "👉 Please upload a **clear, close-up potato leaf image**."
        )

        # Optional debug output
        with st.expander("🔍 Model output (debug)"):
            for i, cls in enumerate(class_names):
                st.write(f"{cls}: {predictions[i] * 100:.2f}%")
            st.write(f"Entropy: {entropy:.2f}")

        st.stop()

    # ---------------- VALID LEAF RESULT ----------------
    st.subheader("🔍 Prediction Probabilities")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {predictions[i] * 100:.2f}%")

    st.success(f"✅ Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
