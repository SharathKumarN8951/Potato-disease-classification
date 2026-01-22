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
CONFIDENCE_THRESHOLD = 65.0   # %
ENTROPY_THRESHOLD = 1.1       # uncertainty

# ---------------- Utility functions ----------------
def prediction_entropy(probs):
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))


def is_valid_image(image: Image.Image) -> bool:
    """
    VERY LENIENT image validation
    Blocks only:
    - Blank images
    - UI screenshots
    - Extremely dark/bright images
    """
    img = np.array(image.resize((128, 128))).astype(float)

    brightness = np.mean(img)
    contrast = np.std(img)

    # Block only obvious invalid images
    if brightness < 30:      # too dark
        return False
    if brightness > 245:     # too bright / white
        return False
    if contrast < 15:        # very low texture (UI, icons)
        return False

    return True


# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("potatoes.keras", compile=False)

model = load_model()

# ⚠️ MUST MATCH training class names
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

    # ---------------- STEP 1: Basic validation ----------------
    if not is_valid_image(image):
        st.error(
            "❌ **Invalid image detected**\n\n"
            "This image does not appear suitable for disease detection.\n\n"
            "👉 Please upload a **clear potato leaf image**."
        )
        st.stop()

    # ---------------- STEP 2: Model inference ----------------
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100
    entropy = prediction_entropy(predictions)

    # ---------------- STEP 3: Confidence handling ----------------
    if confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
        st.warning(
            "⚠️ **Low confidence prediction**\n\n"
            "The image was processed, but the model is not confident.\n\n"
            "Try uploading a **clear, close-up potato leaf image**."
        )

    st.subheader("🔍 Prediction Probabilities")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {predictions[i] * 100:.2f}%")

    st.success(f"✅ Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")
