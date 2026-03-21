import cv2
import os
from core.scrfd_detector import SCRFD

class FaceDetector:
    def __init__(self, model_path="models/scrfd_500m_bnkps.onnx"):
        self.detector = SCRFD(model_path)

    def detect(self, frame, logger=None):
        # Strict Noise Filtering for Hackathon (Zero FP Goal)
        THRESHOLD = 0.25
        MIN_SIZE = 20
        MAX_ASPECT_RATIO = 2.2 # Filter out non-face shapes (bags, legs, hands)
        
        raw_detections = self.detector.detect(frame, conf_threshold=THRESHOLD)
        detections = []
        
        for det in raw_detections:
            conf = det["confidence"]
            bbox = det["bbox"]
            landmarks = det["landmarks"]
            
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            # 1. Size Check
            if w < MIN_SIZE or h < MIN_SIZE:
                continue
                
            # 2. Shape Check (Face boxes should be relatively square or 1:1.5)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio > MAX_ASPECT_RATIO:
                continue
            
            detections.append({
                "bbox": bbox,
                "confidence": conf,
                "landmarks": landmarks,
                "class": 0 # Face
            })
            
        return detections
