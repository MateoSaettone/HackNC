from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

# Initialize models
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def load_friend_embedding(path="friend_embedding.pt"):
    """Load the friend's embedding from a file."""
    return torch.load(path)

# Load the friend's embedding
friend_embedding = load_friend_embedding()

def detect_faces_in_image(image_path):
    """Detect faces in an image and return face tensors."""
    image = Image.open(image_path)
    face_tensors, _ = mtcnn(image, return_prob=True)
    return face_tensors

async def process_photo(image):
    face_tensors = detect_faces_in_image(image)
    if face_tensors is None:
        return False

    for face in face_tensors:
        face_embedding = resnet(face.unsqueeze(0))
        if compare_embeddings(face_embedding, friend_embedding):
            return True
    return False

def compare_embeddings(embedding1, embedding2, threshold=0.6):
    distance = torch.dist(embedding1, embedding2).item()
    return distance < threshold
