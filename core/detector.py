import cv2
import os
from core.scrfd_detector import SCRFD

class FaceDetector:
    def __init__(self, model_path="models/scrfd_500m_bnkps.onnx"):
        self.detector = SCRFD(model_path)

    def detect(self, frame, logger=None):
        # High-Recall Polish (Purdha Support)
        THRESHOLD = 0.15
        MIN_SIZE = 15
        MAX_ASPECT_RATIO = 3.0 # Allows for extreme veil draping
        
        raw_detections = self.detector.detect(frame, conf_threshold=THRESHOLD)
        detections = []
        
        for det in raw_detections:
            conf = det["confidence"]
            bbox = det["bbox"]
            landmarks = det["landmarks"]
            
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            # 1. Size & Aspect Ratio Check
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if w < MIN_SIZE or h < MIN_SIZE or aspect_ratio > MAX_ASPECT_RATIO:
                continue
                
            # 2. Geometry Check (Eye alignment)
            # Normal faces have horizontal eyes. Objects often have vertical "features".
            if landmarks is not None and len(landmarks) >= 2:
                eye_l, eye_r = landmarks[0], landmarks[1]
                dx = abs(eye_r[0] - eye_l[0])
                dy = abs(eye_r[1] - eye_l[1])
                # Reject if "eyes" are too vertical (> 45 degrees)
                if dy > dx * 1.5: 
                    continue
            
            detections.append({
                "bbox": bbox,
                "confidence": conf,
                "landmarks": landmarks,
                "class": 0 # Face
            })
            
        return detections
