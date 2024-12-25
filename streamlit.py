import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# Load the trained model
model = load_model("handdigit_ann_model.h5")

# Set up the Streamlit app
st.title("Handwritten Digit Recognition")
st.write("Upload an image of a digit to recognize it.")

# Upload the image
uploaded_file = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Open and preprocess the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Convert image to grayscale and resize to 28x28
    image = ImageOps.grayscale(image)
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, 28, 28)
    
    # Make prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)
    
    # Display result
    st.write(f"Predicted Digit: {predicted_digit}")
