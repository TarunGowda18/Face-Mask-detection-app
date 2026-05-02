import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import timeimport streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)
# ---------------------- Title ----------------------
st.title("😷 Face Mask Detection")
st.write("Upload an image to detect whether a person is wearing a mask or not.")
# ---------------------- Load Model ----------------------
model = load_model("cnn_model.h5")
# ---------------------- Upload ----------------------
uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display Image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Prediction
    prediction = model.predict(img_array)[0][0]
    # Confidence
    confidence = prediction if prediction > 0.5 else 1 - prediction
    # ---------------------- Result ----------------------
    st.subheader("Prediction Result")
    if prediction > 0.5:
        st.error("❌ Without Mask")
    else:
        st.success("😷 With Mask")
    st.info(f"Confidence: {confidence*100:.2f}%")
# ---------------------- Footer ----------------------
st.write("---")
st.caption("💡 Tip: Upload clear face images for better accuracy")

# --- Setup ---
st.set_page_config(page_title="Pro Mask Detector", layout="wide")

# Optimization: Cache the model so it doesn't reload on every click
@st.cache_resource
def load_mask_model():
    # Ensure this file path is correct for your local setup
    return tf.keras.models.load_model("face_mask_detector.h5")

model = load_mask_model()

st.title("🛡️ Advanced Face Mask Guard")

# --- Sidebar Features ---
st.sidebar.header("Settings & History")
show_details = st.sidebar.checkbox("Show Raw Probabilities", value=False)

# --- Feature 1: Choose Input Method ---
option = st.radio("Select Input Method:", ("Upload Image", "Use Webcam"))

if option == "Upload Image":
    source = st.file_uploader("Pick an image...", type=["jpg", "png", "jpeg"])
else:
    source = st.camera_input("Take a snapshot")

# --- Processing Logic ---
if source is not None:
    # Feature 2: Loading Animation
    with st.spinner('Analyzing face...'):
        img = Image.open(source).convert("RGB")
        
        # Preprocessing
        processed_img = img.resize((224, 224))
        img_array = np.array(processed_img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Prediction
        prediction_raw = model.predict(img_array)[0][0]
        
        # FIX: Convert numpy.float32 to standard Python float for Streamlit compatibility
        prediction = float(prediction_raw)
        
        time.sleep(0.5) # Short delay for UX

    # --- Feature 3: Visual Feedback Layout ---
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Analyzed Image", use_container_width=True)

    with col2:
        st.subheader("Analysis Result")
        
        # Logic for Mask/No Mask (assuming 0 = Mask, 1 = No Mask)
        if prediction > 0.5:
            percent = prediction * 100
            st.error(f"NO MASK DETECTED")
            # Fixed: 'prediction' is now a standard float
            st.progress(prediction) 
        else:
            percent = (1 - prediction) * 100
            st.success(f"MASK DETECTED")
            # Fixed: Ensuring the subtraction result is also treated as a float
            st.progress(1.0 - prediction)

        st.metric(label="Certainty", value=f"{percent:.1f}%")

        # Feature 5: Conditional Details
        if show_details:
            st.write(f"Raw Model Output: `{prediction}`")
