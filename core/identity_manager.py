import numpy as np
import time
import cv2
from database.db import Database
from core.recognizer import FaceRecognizer

class IdentityManager:
    """
    Production Identity Manager.

    KEY PARAMETERS:
    - MATCH_THRESHOLD = 0.60   : High enough to prevent twin merging
    - TWIN_GUARD_SIM = 0.70    : Above this, biometrics are trusted even if colors differ
    - COLOR_SEP_THRESHOLD = 0.45: Below this correlation = different person (twin sep.)
    - MAX_EMBEDDINGS = 15      : Gallery cap per identity
    - SPATIAL_RADIUS = 600px   : Spatial guard distance
    """
    MATCH_THRESHOLD      = 0.65  # High threshold — prevents twin merging
    TWIN_GUARD_SIM       = 0.75  # Only trust biometrics above this (very high confidence)
    COLOR_SEP_THRESHOLD  = 0.45  # correl < this = likely different person (twin guard)
    MAX_EMBEDDINGS_PER_FACE = 15
    SPATIAL_RADIUS       = 600

    def __init__(self, config, db, recognizer):
        self.config = config
        self.db = db
        self.recognizer = recognizer
        self.gallery = self._load_gallery()
        self.color_gallery = {}   # face_id -> HSV hist (flattened)
        self.spatial_memory = {}  # face_id -> {bbox, timestamp}

    def _load_gallery(self):
        return self.db.get_all_faces()

    def _get_center(self, bbox):
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    def _max_similarity(self, query_emb, gallery_embs):
        """Max cosine similarity across all stored embeddings."""
        if not gallery_embs:
            return 0.0
        return max(self.recognizer.compare_embeddings(query_emb, g) for g in gallery_embs)

    def _extract_color_hist(self, frame, bbox):
        """HSV color histogram from body region (below face) for twin separation."""
        x1, y1, x2, y2 = bbox
        # Use upper-body region below the face for clothing color
        face_h = y2 - y1
        body_y1 = min(frame.shape[0], y2)
        body_y2 = min(frame.shape[0], y2 + int(face_h * 1.5))
        roi = frame[body_y1:body_y2, max(0, x1):x2]
        if roi.size == 0:
            # Fallback: use face region itself
            roi = frame[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten().astype(np.float32)

    def _color_similarity(self, hist_a, hist_b):
        """Bhattacharyya-based correlation between two histograms. Returns [0,1]."""
        if hist_a is None or hist_b is None:
            return 1.0  # Unknown — don't penalize
        return float(cv2.compareHist(
            hist_a.reshape(-1, 1),
            hist_b.reshape(-1, 1),
            cv2.HISTCMP_CORREL
        ))

    def match_or_register(self, embedding, forbidden_ids, current_bbox=None, frame=None):
        """
        Match query embedding against gallery.
        Returns (face_id, similarity) or registers new identity.
        """
        if embedding is None:
            return None, 0.0

        now = time.time()
        color_hist = self._extract_color_hist(frame, current_bbox) if (frame is not None and current_bbox is not None) else None

        best_id  = None
        best_sim = -1.0

        for face_id, gallery_embs in self.gallery.items():
            if face_id in forbidden_ids:
                continue

            # Spatial guard: skip if recently seen far away
            if current_bbox and face_id in self.spatial_memory:
                mem = self.spatial_memory[face_id]
                if (now - mem["timestamp"]) < 2.0:
                    dist = np.linalg.norm(self._get_center(current_bbox) - self._get_center(mem["bbox"]))
                    if dist > self.SPATIAL_RADIUS:
                        continue

            sim = self._max_similarity(embedding, gallery_embs)
            if sim < self.MATCH_THRESHOLD:
                continue

            # Twin color guard:
            # If there's a stored color for this face_id, check if colors match.
            # Only reject if biometric confidence is BELOW TWIN_GUARD_SIM (not a slam dunk)
            if color_hist is not None and face_id in self.color_gallery:
                c_sim = self._color_similarity(self.color_gallery[face_id], color_hist)
                if sim < self.TWIN_GUARD_SIM and c_sim < self.COLOR_SEP_THRESHOLD:
                    # Marginal biometric match AND very different colors → likely different person
                    continue

            if sim > best_sim:
                best_sim = sim
                best_id = face_id

        if best_id:
            # Update spatial memory
            if current_bbox:
                self.spatial_memory[best_id] = {"bbox": current_bbox, "timestamp": now}
            # Running-average color update
            if color_hist is not None:
                if best_id not in self.color_gallery:
                    self.color_gallery[best_id] = color_hist
                else:
                    self.color_gallery[best_id] = (0.7 * self.color_gallery[best_id] + 0.3 * color_hist)
            # Grow embedding gallery
            if len(self.gallery[best_id]) < self.MAX_EMBEDDINGS_PER_FACE:
                self.gallery[best_id].append(embedding)
                self.db.add_embedding(best_id, embedding.tobytes())
            return best_id, best_sim

        # New identity
        new_id = f"Face_{int(now)}_{len(self.gallery)}"
        self.gallery[new_id] = [embedding]
        self.db.add_face(new_id, embedding.tobytes())
        if current_bbox:
            self.spatial_memory[new_id] = {"bbox": current_bbox, "timestamp": now}
        if color_hist is not None:
            self.color_gallery[new_id] = color_hist
        return new_id, 1.0
