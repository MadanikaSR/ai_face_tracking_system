import numpy as np
import time
from database.db import Database
from core.recognizer import FaceRecognizer

class IdentityManager:
    def __init__(self, config, db, recognizer):
        self.config = config
        self.db = db
        self.recognizer = recognizer
        self.recognition_threshold = self.config.get("recognition_threshold", 0.68)
        self.max_gallery_size = 12
        self.registered_faces = self._load_gallery()
        
        # Spatiotemporal Re-ID Cache
        self.recent_exits = {} # face_id -> {"bbox": [], "time": float}

    def _load_gallery(self):
        return self.db.get_all_faces()

    def add_exit_cache(self, face_id, bbox):
        """Caches the location of an identity as it leaves."""
        self.recent_exits[face_id] = {
            "bbox": bbox,
            "time": time.time()
        }
        # Cleanup old exits (30s)
        now = time.time()
        self.recent_exits = {k: v for k, v in self.recent_exits.items() if now - v["time"] < 30}

    def match_or_register(self, embedding, forbidden_ids, current_bbox=None):
        if embedding is None: return "Unknown", 0.0

        best_match_id = None
        max_global_sim = -1.0
        
        for face_id, gallery in self.registered_faces.items():
            # 1. Similarity Search
            similarities = [self.recognizer.compare_embeddings(embedding, g_emb) for g_emb in gallery]
            face_max_sim = max(similarities) if similarities else 0.0
            
            # 2. Spatiotemporal Bias
            effective_threshold = self.recognition_threshold
            if current_bbox and face_id in self.recent_exits:
                exit_data = self.recent_exits[face_id]
                dist = self._get_bbox_dist(current_bbox, exit_data["bbox"])
                if dist < 500 and (time.time() - exit_data["time"]) < 10:
                    effective_threshold -= 0.1
            
            if face_max_sim > effective_threshold and face_max_sim > max_global_sim:
                if face_id not in forbidden_ids: # STRICT CCTV ENFORCEMENT
                    max_global_sim = face_max_sim
                    best_match_id = face_id
                else:
                    # Match found but already taken this frame. 
                    # We continue searching to see if they match ANOTHER person 
                    # (very unlikely) or we eventually register a new ID.
                    pass

        # DUPLICATE PREVENTION: 
        # If the best match is already "locked" by another track in this frame,
        # return it but don't register it as a NEW face.
        if best_match_id:
            # We return it even if forbidden, the pipeline will decide not to log it.
            return best_match_id, max_global_sim
        else:
            # Only register a new face if truly no match found
            new_id = f"Face_{int(time.time())}_{len(self.registered_faces)}"
            self.db.add_face(new_id, embedding.tobytes())
            self.registered_faces[new_id] = [embedding]
            return new_id, 1.0

    def _get_bbox_dist(self, b1, b2):
        c1 = [(b1[0]+b1[2])//2, (b1[1]+b1[3])//2]
        c2 = [(b2[0]+b2[2])//2, (b2[1]+b2[3])//2]
        return np.linalg.norm(np.array(c1) - np.array(c2))
