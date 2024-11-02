import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Load the MTCNN (face detector) and InceptionResnetV1 (face recognizer) models
mtcnn = MTCNN(image_size=160, margin=0)  # You can adjust the size/margin
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load your friend's image
img_path = "antonytest.jpg"  # Replace with the actual path to the photo
img = Image.open(img_path)

# Detect face and generate embedding
face = mtcnn(img)  # This will crop and align the face
if face is not None:
    embedding = resnet(face.unsqueeze(0))  # Generate embedding
    
    # Save the embedding to a .pt file
    torch.save(embedding, "friend_embedding.pt")
    print("Friend's embedding saved as friend_embedding.pt")
else:
    print("No face detected. Try using a different photo.")
