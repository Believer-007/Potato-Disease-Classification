import pandas as pd 
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
import os
import cv2
import joblib

# Load the trained model
model = joblib.load('/Users/himanshuyadav/Desktop/PD/Final_PD.pkl' , 'rb')  # Replace with the actual path to your trained model

# Disease classes (replace with the classes your model was trained on)
class_names = ['Potato__Early_Blight','Potato__Late_Blight','Potato__Healthy']  # Example classes

# Streamlit app title
st.title('Potato Leaf Disease Detection')

# Upload image section
st.write("Upload an image of a potato leaf to detect the disease:")

# Image upload from the user
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

def model_prediction(model , image):
        # img = tf.keras.preprocessing.image.img_to_array(image.numpy())
        # img = tf.expand_dims(img,0)

        predictions = model.predict(image)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100* (np.max(predictions[0])), 2)
        return predicted_class , confidence

if uploaded_file is not None:
    # Display the uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    img = image_data.resize((256, 256))  # Resize to the input size expected by the model
    img_array = np.array(img)  # Convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict the class
    with st.spinner('Classifying...'):
        
        predicted_class , confidence = model_prediction(model , img_array )
        
        # prediction = model.predict(img_array)
        # predicted_class = np.argmax(prediction, axis=1)[0]  # Get the index of the max probability class
        # predicted_label = class_names[predicted_class]
        # confidence = prediction[0][predicted_class] * 100  # Confidence level of the prediction

    # Display the result
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Add a brief description of the predicted disease
    if predicted_class == 'Potato__Early_Blight':
        st.write("Early Blight causes brown spots and yellowing on the leaves.")
    elif predicted_class == 'Potato__Late_Blight':
        st.write("Late Blight is a dangerous fungal disease that causes dark lesions on leaves and stems.")
    elif predicted_class == 'Potato__Healthy':
        st.write("The potato leaf appears to be healthy.")
    
# Optionally, you can add more information or help text
st.sidebar.header('About')
st.sidebar.text("""
This app helps detect diseases in potato leaves using a trained Convolutional Neural Network (CNN) model.
Upload an image of a potato leaf to classify whether it is healthy or has any diseases.
""")
