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
ENTROPY_THRESHOLD = 0.9       # slightly relaxed

# ---------------- Utility functions ----------------
def prediction_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))


def is_probable_leaf(image: Image.Image) -> bool:
    """
    Improved leaf validation (RELAXED + ROBUST):
    - Accepts real potato leaves (dark/light/shadow)
    - Rejects blue, UI, objects, random images
    """
    img = np.array(image.resize((128, 128))).astype(float)

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    # Green dominance ratio
    green_ratio = np.mean(g / (r + b + 1e-6))

    # Overall brightness (avoid very dark / blank images)
    brightness = np.mean((r + g + b) / 3)

    return (green_ratio > 0.9) and (brightness > 40)


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
    "Choose an image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------------- STEP 1: Leaf validation ----------------
    if not is_probable_leaf(image):
        st.error(
            "❌ **Invalid image detected**\n\n"
            "This does not appear to be a potato leaf.\n\n"
            "👉 Please upload a **clear green potato leaf image**."
        )
        st.stop()

    # ---------------- STEP 2: Model inference ----------------
    # NO resizing / normalization here
    # (Handled inside the model)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100
    entropy = prediction_entropy(predictions)

    # ---------------- STEP 3: Uncertainty check ----------------
    if confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
        st.warning(
            "⚠️ **Uncertain prediction**\n\n"
            "The image is detected as a leaf, but confidence is low.\n\n"
            "Please upload a **well-lit, close-up potato leaf image**."
        )
        st.info(
            f"📊 Confidence: {confidence:.2f}%\n"
            f"📈 Uncertainty (entropy): {entropy:.2f}"
        )
    else:
        st.subheader("🔍 Prediction Probabilities")
        for i, cls in enumerate(class_names):
            st.write(f"{cls}: {predictions[i] * 100:.2f}%")

        st.success(f"✅ Prediction: **{predicted_class}**")
        st.info(f"📊 Confidence: **{confidence:.2f}%**")
