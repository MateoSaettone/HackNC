import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# Load your friend’s embedding
def load_embedding(embedding_path):
    return np.load(embedding_path)

# Function to preprocess images for the model
def preprocess_image(image_path, image_size=(160, 160)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to calculate embeddings for an image
def calculate_embedding(model, image_path):
    preprocessed_image = preprocess_image(image_path)
    embedding = model.predict(preprocessed_image)
    return embedding

# Function to compare embeddings
def is_match(embedding1, embedding2, threshold=0.5):
    # Euclidean distance between the embeddings
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance < threshold

# Main testing function
def test_images(embedding_path, images_folder, model_path, threshold=0.5):
    # Load the model and friend’s embedding
    model = tf.keras.models.load_model(model_path)
    friend_embedding = load_embedding(embedding_path)
    
    # Iterate through images in the folder
    for image_file in Path(images_folder).rglob("*.[pj][pn]g"):  # finds .jpg or .png images
        image_embedding = calculate_embedding(model, str(image_file))
        
        # Check if the image matches the friend's embedding
        if is_match(friend_embedding, image_embedding, threshold):
            print(f"Friend detected in image: {image_file}")
        else:
            print(f"No match in image: {image_file}")

# Paths for the embedding, model, and images folder
embedding_path = "path/to/friend_embedding.npy"
images_folder = "path/to/sample_images"
model_path = "path/to/model.h5"

# Run the test
test_images(embedding_path, images_folder, model_path)
