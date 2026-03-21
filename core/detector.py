import cv2
import os
import numpy as np
from core.scrfd_detector import SCRFD

class FaceDetector:
    def __init__(self, model_path="models/scrfd_500m_bnkps.onnx"):
        self.detector = SCRFD(model_path)

    def detect(self, frame, logger=None):
        # Final Hackathon Polish (Ultra Recall 0.10 + Geometric Guard)
        THRESHOLD = 0.10
        MIN_SIZE = 15
        MAX_ASPECT_RATIO = 2.5 
        
        raw_detections = self.detector.detect(frame, conf_threshold=THRESHOLD)
        detections = []
        
        for det in raw_detections:
            conf = det["confidence"]
            bbox = [int(x) for x in det["bbox"]]
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            if w * h < 400: continue # Ignore tiny artifacts at 0.10 threshold
            
            landmarks = det["landmarks"]
            
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            
            # 1. Size & Aspect Ratio Check
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if w < MIN_SIZE or h < MIN_SIZE or aspect_ratio > MAX_ASPECT_RATIO:
                continue
                
            # 2. Geometry Check (Eye alignment & 2D Area)
            # Normal faces have horizontal eyes and a 2D landmark area.
            if landmarks is not None and len(landmarks) >= 3:
                eye_l, eye_r = landmarks[0], landmarks[1]
                dx = abs(eye_r[0] - eye_l[0])
                dy = abs(eye_r[1] - eye_l[1])
                # Reject if "eyes" are too vertical (> 45 degrees)
                if dy > dx * 1.5: continue
                
                # REJECT COLLINEAR NOISE (Cracks): 
                # Check area of triangle formed by eyes and nose (landmarks 0, 1, 2)
                l0, l1, l2 = np.array(landmarks[0]), np.array(landmarks[1]), np.array(landmarks[2])
                area = 0.5 * np.abs(np.cross(l1-l0, l2-l0))
                
                # ORIENTATION CHECK (FRONTAL ONLY)
                is_frontal = (abs(l1[0]-l0[0]) >= w*0.1) and (min(l0[0], l1[0])-10 <= l2[0] <= max(l0[0], l1[0])+10)
                
                # If shape is junk but confidence is high, it's a "Partial Face" (Purdha/Mask)
                if area < (w*h)*0.005 or not is_frontal:
                    if conf < 0.4: continue 
                    landmarks = None # Soft-Detections don't use alignment
            
            detections.append({
                "bbox": bbox,
                "confidence": float(conf),
                "landmarks": landmarks,
                "class": 0 # Face
            })
            
        return detections
