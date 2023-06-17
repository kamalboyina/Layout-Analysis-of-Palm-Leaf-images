import streamlit as st
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np

# Load the pre-trained models
model_paths = [
    'FCN_8_with_skip.h5',
]

models = []
for path in model_paths:
    model = tf.keras.models.load_model(path)
    models.append(model)

def normalize_data(image):
    return image/255.0

# Define helper function to generate predictions
def generate_predictions(image):
    predictions = []
    for model in models:
        # Preprocess the input image
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        new_size = (384, 96)
        image_resized = cv2.resize(image, new_size)
        normalized_image = normalize_data(image_resized)
        normalized_image = np.array([normalized_image])
        image_tensor = tf.convert_to_tensor(normalized_image)

        # Generate the prediction
        prediction = model.predict(image_tensor)
        predictions.append(prediction)

    return predictions

# Set Streamlit app title and layout
st.set_page_config(layout="wide")  # Move this line to after the function definitions
st.sidebar.empty()
st.sidebar.title("LAYOUT ANALYSIS OF PALM LEAF IMAGES")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
st.sidebar.image("red.png", width=50)
st.sidebar.write("- BACKGROUND")

st.sidebar.image("blue.png", width=50)
st.sidebar.write("- TEXT")

st.sidebar.image("green.png", width=50)
st.sidebar.write("- PUNCH HOLES")
if uploaded_file is not None:
    # Read the image file and display it
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Input Image", use_column_width=True)

    # Generate the predictions
    predictions = generate_predictions(input_image)

    # Display the output images
    st.header("Predicted Images")
    for i, prediction in enumerate(predictions):
        output_image = tf.keras.preprocessing.image.array_to_img(prediction[0])
        st.subheader(f"{model_paths[i]}")
        st.image(output_image, caption=f"Output Image from Model ", use_column_width=True)

    # Optional: Display the prediction values
