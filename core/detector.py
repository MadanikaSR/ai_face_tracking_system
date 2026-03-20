import cv2
import os
from core.scrfd_detector import SCRFD

class FaceDetector:
    def __init__(self, model_path="models/scrfd_500m_bnkps.onnx"):
        self.detector = SCRFD(model_path)

    def detect(self, frame, logger=None):
        raw_detections = self.detector.detect(frame)
        detections = []
        
        # Filtering parameters
        MIN_CONFIDENCE = 0.6
        MIN_SIZE = 30
        
        for det in raw_detections:
            conf = det["confidence"]
            bbox = det["bbox"] # [x1, y1, x2, y2]
            landmarks = det["landmarks"]
            
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Simple shape filtering
            aspect_ratio = max(width, height) / max(min(width, height), 1)
            
            if conf < MIN_CONFIDENCE:
                if logger: logger.debug(f"Rejected: low confidence ({conf:.2f})")
                continue
            if width < MIN_SIZE or height < MIN_SIZE:
                if logger: logger.debug(f"Rejected: too small ({width}x{height})")
                continue
            if aspect_ratio > 1.5:
                if logger: logger.debug(f"Rejected: invalid aspect ratio ({aspect_ratio:.2f})")
                continue
                
            detections.append({
                "bbox": bbox,
                "confidence": conf,
                "landmarks": landmarks,
                "class": 0 # Face
            })
            
        return detections
