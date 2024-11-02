from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

# Initialize the models
mtcnn = MTCNN(keep_all=True)  # Set keep_all=True to detect multiple faces
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load the friend's embedding
friend_embedding = torch.load("/Users/mateosaettone/GitHub Repos/HackNC_2024/HackNC_0640/HackNC/backend/friend_embedding.pt")

# Function to process and get embeddings for multiple faces in an image
def calculate_embedding(image_path):
    img = Image.open(image_path)
    faces = mtcnn(img)  # Detect multiple faces

    if faces is not None:
        embeddings = []
        with torch.no_grad():
            for face in faces:
                embedding = resnet(face.unsqueeze(0))  # Get embedding for each face
                embeddings.append(embedding)
        return embeddings  # Return list of embeddings for all detected faces
    return None

# Function to compare embeddings
def is_match(embedding1, embedding2, threshold=0.6):
    distance = torch.dist(embedding1, embedding2).item()
    print(f"Distance between embeddings: {distance}")
    return distance < threshold
