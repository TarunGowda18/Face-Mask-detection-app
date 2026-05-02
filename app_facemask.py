import streamlit as st
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
