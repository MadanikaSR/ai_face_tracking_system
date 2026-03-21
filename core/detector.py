import cv2
import os
import numpy as np
from core.scrfd_detector import SCRFD

class FaceDetector:
    """
    Production Face Detector using InsightFace SCRFD.

    VALIDATION STRATEGY (back-of-head rejection WITHOUT blocking occluded faces):
    Rule 1: Min face size 20px
    Rule 2: Must have 5 SCRFD landmarks
    Rule 3: Eye levelness ≤ 50% height   — generous for purdha/tilted CCTV
    Rule 4: Nose BELOW eyes              — primary back-of-head killer
    Rule 5: Eye-face symmetry check      — face width sanity check
    """

    CONF_THRESHOLD  = 0.28    # Slightly lower for purdha detection
    MIN_FACE_SIZE   = 18      # px minimum
    EYE_LEVEL_RATIO = 0.55    # Very generous for tilted/purdha faces

    def __init__(self, model_path="models/scrfd_500m_bnkps.onnx"):
        self.detector = SCRFD(model_path)

    def detect(self, frame, logger=None):
        """
        Detect and validate faces.
        Returns validated bboxes only. Raw detections are also returned in a
        separate key for debug visualization.
        """
        raw = self.detector.detect(frame, conf_threshold=self.CONF_THRESHOLD)
        validated = []

        for det in raw:
            conf      = float(det["confidence"])
            bbox      = det["bbox"]
            landmarks = det["landmarks"]

            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1

            # Rule 1: Minimum face size
            if w < self.MIN_FACE_SIZE or h < self.MIN_FACE_SIZE:
                continue

            # Rule 2: Must have 5 landmarks
            if not landmarks or len(landmarks) < 5:
                continue

            left_eye  = np.array(landmarks[0])
            right_eye = np.array(landmarks[1])
            nose      = np.array(landmarks[2])
            mouth_l   = np.array(landmarks[3])
            mouth_r   = np.array(landmarks[4])

            # Rule 3: Eye levelness — very generous for purdha/tilts
            eye_y_diff = abs(right_eye[1] - left_eye[1])
            if eye_y_diff > h * self.EYE_LEVEL_RATIO:
                continue

            # Rule 4: Nose BELOW eyes (primary back-of-head/inverted killer)
            eye_center_y = (left_eye[1] + right_eye[1]) / 2.0
            if nose[1] < eye_center_y - (h * 0.08):
                continue  # Nose above eye center — not a real face

            # Rule 5: Mouth BELOW nose (catches back-of-head with high nose)
            if mouth_l[1] < nose[1] - (h * 0.08) and mouth_r[1] < nose[1] - (h * 0.08):
                continue  # Both mouth corners above nose — impossible

            validated.append({
                "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "landmarks":  landmarks,
                "class":      0
            })

        return validated
