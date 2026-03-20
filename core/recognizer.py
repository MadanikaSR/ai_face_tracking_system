import onnxruntime as ort
import numpy as np
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity

class FaceRecognizer:
    def __init__(self, model_path="models/w600k_r50.onnx"):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            print(f"Error: Model not found at {self.model_path}. Please run download_models.py first.")
            self.session = None
            return

        # Load the ONNX model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, img):
        """Preprocess the image according to insightface requirements (112x112, etc.)"""
        # Resize to 112x112 (Standard InsightFace input size)
        img = cv2.resize(img, (112, 112))
        
        # Change color space if necessary (many models expect RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize (usually -0.5 to 0.5 or 0 to 1 depending on model, 
        # but Buffalo_L expects (x - 127.5) / 127.5)
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        
        # Change shape to (1, 3, 112, 112)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def get_embedding(self, face_image):
        if self.session is None:
            return None
            
        try:
            # Preprocess the cropped face image
            blob = self.preprocess(face_image)
            
            # Run inference
            embedding = self.session.run([self.output_name], {self.input_name: blob})[0]
            
            # Flattend to (512,)
            return embedding.flatten()
        except Exception as e:
            print(f"Recognition Error: {e}")
            return None

    def compare_embeddings(self, embedding1, embedding2):
        if embedding1 is None or embedding2 is None:
            return 0.0
        # Cosine similarity
        similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
        return float(similarity)
