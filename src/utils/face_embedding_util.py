import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import torchvision.transforms as transforms

class FaceNetEmbedding:
    def __init__(self, device=None):
        """Initialize FaceNet embedding model"""
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Load pre-trained FaceNet model
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        
        # Face preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"FaceNet model loaded on {self.device}")
    
    def preprocess_face(self, face_crop):
        """Preprocess face crop for FaceNet"""
        # Convert BGR to RGB
        if len(face_crop.shape) == 3:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(face_crop)
        
        # Apply preprocessing
        face_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        return face_tensor
    
    def get_embedding(self, face_crop):
        """
        Extract 512-dimensional embedding from face crop
        
        Args:
            face_crop: numpy array of face image (H, W, 3)
            
        Returns:
            numpy array of 512-dimensional embedding
        """
        if face_crop is None or face_crop.size == 0:
            return None
            
        try:
            # Preprocess face
            face_tensor = self.preprocess_face(face_crop)
            
            # Get embedding
            with torch.no_grad():
                embedding = self.model(face_tensor)
                
            # Convert to numpy and normalize
            embedding = embedding.cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)  # L2 normalize
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def compare_embeddings(self, emb1, emb2):
        """Compare two embeddings using cosine similarity"""
        if emb1 is None or emb2 is None:
            return 0.0
            
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return similarity