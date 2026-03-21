import onnxruntime as ort
import numpy as np
import cv2
import os

class FaceRecognizer:
    """
    ArcFace-based face recognizer using InsightFace ONNX model.
    Returns L2-normalized embeddings for reliable cosine similarity.
    """
    def __init__(self, model_path="models/w600k_r50.onnx"):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            print(f"[ERROR] Recognizer model not found at {self.model_path}. Run download_models.py.")
            self.session = None
            return

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        print(f"[INFO] Recognizer loaded: {model_path}")

    def preprocess(self, img):
        """Standard InsightFace preprocessing: resize, normalize to [-1, 1]."""
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def get_embedding(self, face_image):
        """
        Returns L2-normalized ArcFace embedding (512-dim).
        L2 normalization makes dot product == cosine similarity.
        """
        if self.session is None:
            return None
        try:
            blob = self.preprocess(face_image)
            raw_emb = self.session.run([self.output_name], {self.input_name: blob})[0]
            emb = raw_emb.flatten().astype(np.float32)
            # L2 Normalize — Critical for reliable cosine similarity
            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                return None
            return emb / norm
        except Exception as e:
            return None

    def compare_embeddings(self, emb1, emb2):
        """
        Cosine similarity between two L2-normalized embeddings.
        Since embeddings are normalized, this is simply the dot product.
        Returns value in [-1, 1]; higher = more similar.
        """
        if emb1 is None or emb2 is None:
            return 0.0
        # Both embeddings are already L2-normalized
        sim = float(np.dot(emb1, emb2))
        return max(-1.0, min(1.0, sim))
