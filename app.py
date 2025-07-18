import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import pandas as pd

# Load model and label dictionary
model = tf.keras.models.load_model("traffic_sign_model.keras")
with open("labels.pickle", "rb") as f:
    label_dict = pickle.load(f)

# Streamlit UI
st.title("Traffic Sign Classifier")
st.write("Upload a traffic sign image (32x32 or any, it will be resized).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # shape: (1, 32, 32, 3)

    # Predict
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    
    df = pd.read_csv("label_names.csv")

    st.write("### Predicted Class:")
    st.success(f"{df['SignName'][predicted_class]}")

