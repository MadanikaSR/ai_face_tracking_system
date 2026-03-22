import cv2
import os
import numpy as np
from core.scrfd_detector import SCRFD

class FaceDetector:
    """
    Production Face Detector using InsightFace SCRFD.

    BALANCED VALIDATION — 6 lightweight geometric checks.
    Priority: RECALL (catch all real faces) > PRECISION (reject non-faces).
    The eye-gap check (Rule 6) is the primary back-of-head guard.
    All other rules are soft size / orientation checks.

    Rules deliberately removed to avoid over-rejection:
    - Absolute px eye distance  (too strict for small faces far from camera)
    - Eye Y-ratio range         (too strict for CCTV overhead angles)
    - Nose lateral alignment    (too strict for partially-turned faces)
    """

    CONF_THRESHOLD  = 0.42    # Raised from 0.28 — real faces score >0.45, bags/hands score <0.42
    MIN_FACE_SIZE   = 18      # px — ignore truly tiny blobs only
    EYE_LEVEL_RATIO = 0.60    # max |eye_y_diff| / face_height — generous for tilts

    def __init__(self, model_path="models/scrfd_500m_bnkps.onnx"):
        self.detector = SCRFD(model_path)

    def detect(self, frame, logger=None):
        """
        Detect faces and apply balanced geometric validation.
        All landmark access is inside try-except; malformed detections are silently skipped.
        """
        raw = self.detector.detect(frame, conf_threshold=self.CONF_THRESHOLD)
        validated = []

        for det in raw:
            try:
                conf      = float(det["confidence"])
                bbox      = det["bbox"]
                landmarks = det["landmarks"]

                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                # Rule 1: Minimum face size
                if w < self.MIN_FACE_SIZE or h < self.MIN_FACE_SIZE:
                    continue  # Tiny blob — not a face

                # Rule 2: Must have at least 5 facial landmark keypoints
                if not landmarks or len(landmarks) < 5:
                    continue  # No landmarks — back-of-head / non-face object

                # Rule 3: Aspect ratio — faces are roughly square (0.35 to 1.9)
                aspect_ratio = w / float(h) if h > 0 else 0.0
                if not (0.35 <= aspect_ratio <= 1.9):
                    continue  # Extremely elongated — not a face

                # Safe landmark extraction
                left_eye  = np.array(landmarks[0], dtype=np.float32)
                right_eye = np.array(landmarks[1], dtype=np.float32)
                nose      = np.array(landmarks[2], dtype=np.float32)
                mouth_l   = np.array(landmarks[3], dtype=np.float32)
                mouth_r   = np.array(landmarks[4], dtype=np.float32)

                # Rule 4: Eye levelness — generous for purdha / tilted CCTV angles
                eye_y_diff = abs(float(right_eye[1]) - float(left_eye[1]))
                if eye_y_diff > h * self.EYE_LEVEL_RATIO:
                    continue  # Eyes at very different heights — not a valid face

                # Rule 5: Nose BELOW eyes — primary back-of-head / inverted killer
                eye_center_y = (float(left_eye[1]) + float(right_eye[1])) / 2.0
                if float(nose[1]) < eye_center_y - (h * 0.10):
                    continue  # Nose above eyes — impossible on a real face

                # Rule 6: Eye-gap coherence — definitive back-of-head guard
                # On a real face (front, side, purdha with eyes showing), the two eye
                # landmarks are always horizontally separated by >= 15% of face width.
                # On a back-of-head, SCRFD has no eyes to anchor to and places both
                # "eye" landmarks very close together (gap < 5% width) -> rejected.
                eye_x_gap = abs(float(right_eye[0]) - float(left_eye[0]))
                if eye_x_gap < w * 0.15:
                    continue  # Eye landmarks collapsed — back of head or object

                validated.append({
                    "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "landmarks":  landmarks,
                    "class":      0
                })

            except Exception:
                # Skip malformed detections silently — never crash on a single bad frame
                continue

        return validated
