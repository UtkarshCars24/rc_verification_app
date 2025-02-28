import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("rc_verification_model.h5")

# Function to process images
def process_image(image_path):
    IMG_SIZE = (224, 224)
    image = cv2.imread(image_path)
    image = cv2.resize(image, IMG_SIZE)
    image = np.expand_dims(image, axis=0) / 255.0
    return image

# Function to predict RC authenticity
def predict_rc(image_path):
    image = process_image(image_path)
    prediction = model.predict(image)[0][0]
    label = "✅ Original RC" if prediction > 0.5 else "❌ Photocopy RC"
    confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
    return label, round(confidence, 2)

# Streamlit Web UI
st.title("🚗 RC Verification Web App")
st.write("Upload an RC image to check if it's **Original** or a **Photocopy**.")

uploaded_file = st.file_uploader("Upload RC Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = "uploaded_rc.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    st.image(file_path, caption="Uploaded RC Image", use_column_width=True)

    # Get Prediction
    label, confidence = predict_rc(file_path)

    # Show Prediction Result
    st.subheader("🔍 Prediction Result")
    st.write(f"**Result:** {label}")
    st.write(f"**Confidence Score:** {confidence}%")

