import cv2
import os
import numpy as np
from core.scrfd_detector import SCRFD

class FaceDetector:
    """
    Production Face Detector using InsightFace SCRFD.
    Implements a strict face validation layer to reject non-faces.
    """
    MIN_FACE_SIZE = 20  # pixels
    CONF_THRESHOLD = 0.35
    EYE_LEVEL_RATIO = 0.35  # max |eye_y_diff| / face_height (generous for purdha)

    def __init__(self, model_path="models/scrfd_500m_bnkps.onnx"):
        self.detector = SCRFD(model_path)

    def detect(self, frame, logger=None):
        """
        Detects faces with a built-in validation layer.
        Returns only valid face detections with 5 landmarks.
        """
        raw_detections = self.detector.detect(frame, conf_threshold=self.CONF_THRESHOLD)
        validated = []

        for det in raw_detections:
            conf = float(det["confidence"])
            bbox = det["bbox"]  # [x1, y1, x2, y2]
            landmarks = det["landmarks"]  # 5 points [[x,y], ...]

            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1

            # 1. Minimum face size
            if w < self.MIN_FACE_SIZE or h < self.MIN_FACE_SIZE:
                continue

            # 2. Must have landmarks (SCRFD always provides 5 kps)
            if landmarks is None or len(landmarks) < 5:
                continue

            # 3. Eye Levelness Check (catches most back-of-heads)
            # Eyes are landmark[0] (left) and landmark[1] (right)
            left_eye = np.array(landmarks[0])
            right_eye = np.array(landmarks[1])
            eye_y_diff = abs(right_eye[1] - left_eye[1])
            if eye_y_diff > h * self.EYE_LEVEL_RATIO:
                continue

            # 4. Eye separation check (eyes should be horizontally apart)
            eye_x_diff = abs(right_eye[0] - left_eye[0])
            if eye_x_diff < w * 0.10:
                # Eyes too close together — likely not a real face
                continue

            validated.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "landmarks": landmarks,
                "class": 0
            })

        return validated
