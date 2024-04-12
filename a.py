import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import requests
from io import BytesIO
#import cv2

# URL of your model in DigitalOcean Spaces
model_url = r'https://models-spaces30.blr1.digitaloceanspaces.com/CDD_2_2_9082per.keras'

# Download the model file from the URL
response = requests.get(model_url)
# Download the model file from DigitalOcean
with open("CDD_2_2_9082per.keras", "wb") as f:
    f.write(response.content)
# Load the model using TensorFlow
model = tf.keras.models.load_model("CDD_2_2_9082per.keras")

def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(224, 224))
    img = image.img_to_array(img)
    img = tf.image.rgb_to_hsv(img)  # Convert to HSV
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

def classes(predicted_class):
    class_mapping = {
        0: 'Apple_Leaf_Healthy',
    1: 'Apple_Rust_Leaf',
    2: 'Apple_Scab_Leaf',
    3: 'Bell_Pepper_Healthy_Leaf',
    4: 'Bell_Pepper_Leaf_Curl',
    5: 'Bell_Pepper_Leaf_Spot',
    6: 'Cashew_Anthracnose',
    7: 'Cashew_Gumosis',
    8: 'Cashew_Healthy',
    9: 'Cashew_Leaf_Miner',
    10: 'Cashew_Red_Rust',
    11: 'Cassava_Bacterial_Blight',
    12: 'Cassava_Brown_Spot',
    13: 'Cassava_Green_Miite',
    14: 'Cassava_Healthy',
    15: 'Cassava_Mosaic',
    16: 'Corn_Gray_Leaf_Spot',
    17: 'Corn_Leaf_Blight',
    18: 'Corn_Leaf_Healthy',
    19: 'Corn_Rust_Leaf',
    20: 'Grape_Healthy_Leaf',
    21: 'Grape_Leaf_Black_Rot',
    22: 'Peach_Healthy_Leaf',
    23: 'Peach_Bacterial_Spot',
    24: 'Potato_Healthy_Leaf',
    25: 'Potato_Leaf_Early_Blight',
    26: 'Potato_Leaf_Late_Blight',
    27: 'Raspberry_Healthy_Leaf',
    28: 'Soyabean_Healthy_Leaf',
    29: 'Soyabeen_Diseased_Leaf',
    30: 'Squash_Powdery_Mildew_Leaf',
    31: 'Strawberry_Angular_Leaf_Spot',
    32: 'Strawberry_Blossom_Blight',
    33: 'Strawberry_Graymold',
    34: 'Strawberry_Healthy_Leaf',
    35: 'Strawberry_Leaf_Scorch',
    36: 'Strawberry_Leafspot',
    37: 'Strawberry_Powdery_Mildew_Leaf',
    38: 'Tomato_Early_Blight_Leaf',
    39: 'Tomato_Leaf_Bacterial_Spot',
    40: 'Tomato_Leaf_Healthy',
    41: 'Tomato_Leaf_Late_Blight',
    42: 'Tomato_Leaf_Mosaic_Virus',
    43: 'Tomato_Mold_Leaf',
    44: 'Tomato_Septoria_Leaf_Spot',
    45: 'Tomato_Yellow_Leaf_Curl_Virus',
    46: 'Tomato__Spider_Mites_Two_Spotted_Spider_Mite'
    }
    predicted_class_name = class_mapping.get(predicted_class, 'Unknown Class')
    return predicted_class_name

def predict(image_file):
    img = preprocess_image(image_file)
    preds = model.predict(img)
    predicted_class = np.argmax(preds)
    predicted_class_name = classes(predicted_class)
    # Replace the prediction probabilities with your actual values
    prediction_probabilities = np.array(predictions)
    # Apply softmax
    probabilities = np.exp(prediction_probabilities) / np.sum(np.exp(prediction_probabilities))
    max_prob_index = np.argmax(probabilities)
    max_prob = probabilities[0, max_prob_index]

    # Get the indices of the probabilities sorted in descending order
    sorted_indices = np.argsort(probabilities)[0][::-1]
    
    # Extract the top four classes and their probabilities
    top_classes = sorted_indices[:4]
    top_probabilities = probabilities[0, top_classes]

    top_n_prediction = []

    for i, (class_idx, prob) in enumerate(zip(top_classes, top_probabilities), 1):
        prediction = f"Class {classes(class_idx)}: Probability {prob:.4f}"
        top_n_prediction.append(prediction)
        
    return predicted_class_name, top_n_prediction

def main():
    st.title("Crop Disease Detection Image Classifier")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        predicted_class, top_n_probability = predict(uploaded_file)
        #st.write("Prediction: ", predicted_class)
        st.markdown(f"**Prediction:** {predicted_class}", unsafe_allow_html=True)
        st.markdown("**Probability of top n classes with maximum classes:**")
        st.markdown(f"{top_n_probability}")

if __name__ == "__main__":
    main()
