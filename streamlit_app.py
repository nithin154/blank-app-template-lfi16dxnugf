import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Function to load and preprocess the image
def load_image(image_file):
    img = Image.open(image_file)
    img = img.resize((256, 256))
    img = np.array(img)
    img = img / 255.0  # Normalize the image
    return img

# Function to perform image segmentation
def segment_image(model, image):
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    prediction = (prediction > 0.5).astype(np.uint8)  # Threshold the prediction
    prediction = np.squeeze(prediction)  # Remove batch dimension
    return prediction

# Load your pre-trained model (update the path to your model)
model_path = 'path/to/your_model.h5'
model = tf.keras.models.load_model(model_path)

# Streamlit app
st.title("Image Segmentation with U-Net")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = load_image(uploaded_file)

    # Display the original image
    st.image(image, caption='Original Image', use_column_width=True)

    # Perform segmentation
    segmented_image = segment_image(model, image)

    # Display the segmented image
    st.image(segmented_image, caption='Segmented Image', use_column_width=True)

    # Optionally, overlay the segmentation mask on the original image
    overlay_image = cv2.addWeighted(image, 0.7, np.stack((segmented_image,)*3, axis=-1), 0.3, 0)
    st.image(overlay_image, caption='Overlay Image', use_column_width=True)
