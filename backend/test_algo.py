# test_face_detection.py
from face_recognition import load_friend_embedding, detect_faces_in_image, compare_faces
from PIL import Image
import torch

def test_face_detection():
    # Load the friend embedding
    friend_embedding_path = "friend_embedding.pt"  # Specify the path to your friend's embedding file
    friend_embedding = load_friend_embedding(friend_embedding_path)
    
    # Specify the path of the test image you want to use
    test_image_path = "path/to/test_image.jpg"  # Replace with the actual path to your test image
    
    # Load the test image
    test_image = Image.open(test_image_path)
    
    # Detect faces in the test image
    detected_faces = detect_faces_in_image(test_image)
    
    # Compare each detected face to the friend's embedding and store match results
    matches = [compare_faces(friend_embedding, face_embedding) for face_embedding in detected_faces]
    
    # Output the results
    for i, match in enumerate(matches):
        if match:
            print(f"Detected face {i + 1} matches the friend's face.")
        else:
            print(f"Detected face {i + 1} does NOT match the friend's face.")

# Run the test
if __name__ == "__main__":
    test_face_detection()
