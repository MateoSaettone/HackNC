from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image

# Initialize models
mtcnn = MTCNN(keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load the friend's embedding (this could be replaced with a database call if you store embeddings elsewhere)
friend_embedding = torch.load("friend_embedding.pt")

async def process_photo(image):
    face_tensors, _ = mtcnn(image, return_prob=True)
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
