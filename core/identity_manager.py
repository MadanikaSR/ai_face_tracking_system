import numpy as np
import time
import cv2
from database.db import Database
from core.recognizer import FaceRecognizer

class IdentityManager:
    """
    Production Identity Manager.
    - Multi-embedding gallery: compare against ALL embeddings, use MAX similarity.
    - Match threshold: 0.45 (generous enough for purdha/occluded faces).
    - Color twin-guard: prevents identical-looking clothed people from merging.
    - Spatial guard: prevents wrong matches for people far apart.
    """
    MATCH_THRESHOLD = 0.45      # Minimum cosine similarity for a match
    COLOR_MATCH_THRESHOLD = 0.65  # HISTCMP_CORREL — >0.65 = same person color
    MAX_EMBEDDINGS_PER_FACE = 15  # Gallery size cap per identity
    SPATIAL_RADIUS = 600          # px — ignore match if too far and recently seen

    def __init__(self, config, db, recognizer):
        self.config = config
        self.db = db
        self.recognizer = recognizer
        # In-memory gallery: face_id -> list of L2-normalized embeddings
        self.gallery = self._load_gallery()
        # Color gallery: face_id -> HSV histogram
        self.color_gallery = {}
        # Spatial memory: face_id -> {bbox, timestamp}
        self.spatial_memory = {}

    def _load_gallery(self):
        """Load all face embeddings from database into memory."""
        return self.db.get_all_faces()

    def _get_center(self, bbox):
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

    def _max_similarity(self, query_emb, gallery_embs):
        """Compare query against all gallery embeddings, return max similarity."""
        if not gallery_embs:
            return 0.0
        sims = [self.recognizer.compare_embeddings(query_emb, g) for g in gallery_embs]
        return max(sims)

    def _extract_color_hist(self, frame, bbox):
        """Extract normalized HSV color histogram from face region."""
        x1, y1, x2, y2 = bbox
        roi = frame[max(0, y1):y2, max(0, x1):x2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def match_or_register(self, embedding, forbidden_ids, current_bbox=None, frame=None):
        """
        Core identity matching logic.
        Returns (face_id, similarity_score).
        """
        if embedding is None:
            return None, 0.0

        best_id = None
        best_sim = -1.0
        now = time.time()

        # Extract color histogram for this candidate
        color_hist = self._extract_color_hist(frame, current_bbox) if (frame is not None and current_bbox is not None) else None

        for face_id, gallery_embs in self.gallery.items():
            if face_id in forbidden_ids:
                continue

            # Spatial guard: if this person was seen <2s ago far away, skip
            if current_bbox and face_id in self.spatial_memory:
                mem = self.spatial_memory[face_id]
                if (now - mem["timestamp"]) < 2.0:
                    dist = np.linalg.norm(self._get_center(current_bbox) - self._get_center(mem["bbox"]))
                    if dist > self.SPATIAL_RADIUS:
                        continue

            sim = self._max_similarity(embedding, gallery_embs)
            if sim < self.MATCH_THRESHOLD:
                continue

            # Color twin-guard: if colors differ significantly, reject match
            if color_hist is not None and face_id in self.color_gallery:
                color_sim = cv2.compareHist(
                    self.color_gallery[face_id].reshape(-1, 1).astype(np.float32),
                    color_hist.reshape(-1, 1).astype(np.float32),
                    cv2.HISTCMP_CORREL
                )
                # Only reject if biometric similarity is marginal and colors are very different
                if sim < 0.60 and color_sim < 0.55:
                    continue

            if sim > best_sim:
                best_sim = sim
                best_id = face_id

        if best_id:
            # Update spatial memory
            if current_bbox:
                self.spatial_memory[best_id] = {"bbox": current_bbox, "timestamp": now}
            # Update color gallery (running average)
            if color_hist is not None:
                if best_id not in self.color_gallery:
                    self.color_gallery[best_id] = color_hist
                else:
                    self.color_gallery[best_id] = 0.7 * self.color_gallery[best_id] + 0.3 * color_hist
            # Add embedding to gallery (multi-embedding support)
            if len(self.gallery[best_id]) < self.MAX_EMBEDDINGS_PER_FACE:
                self.gallery[best_id].append(embedding)
                self.db.add_embedding(best_id, embedding.tobytes())
            return best_id, best_sim

        # No match — register as new identity
        new_id = f"Face_{int(now)}_{len(self.gallery)}"
        self.gallery[new_id] = [embedding]
        self.db.add_face(new_id, embedding.tobytes())
        if current_bbox:
            self.spatial_memory[new_id] = {"bbox": current_bbox, "timestamp": now}
        if color_hist is not None:
            self.color_gallery[new_id] = color_hist
        return new_id, 1.0
