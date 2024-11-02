import os
from pathlib import Path
from face_recognition import calculate_embedding, is_match, friend_embedding

# Main testing function
def test_images(images_folder, threshold=0.7):
    if friend_embedding is None:
        print("Friend embedding is not loaded. Exiting.")
        return

    for image_file in Path(images_folder).rglob("*.[pj][pn]g"):  # Finds .jpg or .png images
        print(f"\nTesting image: {image_file}")
        image_embeddings = calculate_embedding(image_file)

        friend_found = False  # Initialize friend found flag for this image

        if image_embeddings is not None:
            # Loop through each detected face's embedding
            for i, image_embedding in enumerate(image_embeddings):
                print(f"Comparing face {i+1} in image {image_file}")
                if is_match(friend_embedding, image_embedding, threshold):
                    print(f"Friend detected in face {i+1} of image: {image_file}")
                    friend_found = True  # Set flag to True if friend is found
                    break  # Stop searching if friend is found
                else:
                    print(f"No match for face {i+1} in image: {image_file}")
            
            # Summary for this image
            if friend_found:
                print(f"Summary: Friend detected in {image_file}")
            else:
                print(f"Summary: Friend NOT detected in {image_file}")
        else:
            print(f"No faces detected in image: {image_file}")

# Path to images folder
images_folder = "testpictures"  

# Run the test
test_images(images_folder)
