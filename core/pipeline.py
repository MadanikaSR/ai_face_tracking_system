import os
import numpy as np
import cv2
from datetime import datetime
from core.detector import FaceDetector
from core.tracker import FaceTracker
from core.recognizer import FaceRecognizer
from core.identity_manager import IdentityManager
from core.event_manager import EventManager
from database.db import Database
from logging_system.logger import SystemLogger
from utils.alignment import align_face

class Pipeline:
    """
    Production AI Face Tracking Pipeline.

    Architecture:
        Frame → Detect (every frame) → Validate → Track →
        [Track-First Lock: if track has face_id, SKIP recognition]
        [If new track: collect 8 embeddings → match/register] →
        Ghost Registry (5s) → Log Entry/Exit

    KEY RULES:
        1. track_id → face_id is a PERMANENT LOCK (never changes)
        2. Recognition ONLY runs for Unknown tracks
        3. Ghost registry: re-link returning persons by proximity (no new ID)
        4. Overlap merge: IoU > 0.45 → keep only one
    """

    GHOST_TTL_SECONDS = 8       # How long to remember a lost identity's position
    GHOST_INHERIT_RADIUS = 130  # px — new track within this radius of ghost inherits ID
    MIN_EMBEDDINGS_TO_REGISTER = 5  # Wait for N good embeddings before registering
    OVERLAP_IOU_KILL = 0.45     # If two tracks overlap this much, one is a duplicate

    def __init__(self, config):
        self.config = config
        self.db = Database()
        self.logger = SystemLogger()
        self.detector = FaceDetector()
        self.tracker = FaceTracker(max_lost=60, iou_threshold=0.35)
        self.recognizer = FaceRecognizer()
        self.identity_manager = IdentityManager(config, self.db, self.recognizer)
        self.event_manager = EventManager(self.db, self.logger)

        # track_id → {face_id, vote_embs, status}
        self.track_registry = {}
        # face_id → {bbox, timestamp} — ghost memory for re-linking
        self.ghost_registry = {}

    def _get_iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / float(a1 + a2 - inter + 1e-6)

    def _get_center_dist(self, b1, b2):
        c1 = np.array([(b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2])
        c2 = np.array([(b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2])
        return float(np.linalg.norm(c1 - c2))

    def reset_system(self):
        self.logger.info("Resetting system data...")
        self.db.clear_data()
        self.logger.clear_logs()
        self.event_manager.clear_all()
        # Re-initialize all stateful components
        self.identity_manager = IdentityManager(self.config, self.db, self.recognizer)
        self.event_manager = EventManager(self.db, self.logger)
        self.track_registry = {}
        self.ghost_registry = {}
        self.logger.info("System reset complete.")

    def process_frame(self, frame, frame_count):
        """
        Main processing loop. Called for every frame.
        Returns: (tracked_objects, recognitions, confidences, raw_detections)
        """
        now = datetime.now()
        now_ts = now.timestamp()

        # ── 1. DETECT ─────────────────────────────────────────────────────────
        # Detect on every frame for maximum recall
        det_frame = cv2.resize(frame, (1280, 720))
        h_orig, w_orig = frame.shape[:2]
        h_det, w_det = det_frame.shape[:2]
        sx, sy = w_orig / w_det, h_orig / h_det

        raw_detections = self.detector.detect(det_frame)
        scaled_detections = []
        for det in raw_detections:
            x1, y1, x2, y2 = det["bbox"]
            lmarks = [[p[0] * sx, p[1] * sy] for p in det["landmarks"]] if det["landmarks"] else None
            scaled_detections.append({
                "bbox": [int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)],
                "confidence": det["confidence"],
                "landmarks": lmarks,
                "class": 0
            })

        # ── 2. TRACK ──────────────────────────────────────────────────────────
        tracked_objects = self.tracker.track(frame, scaled_detections)

        # ── 3. DEDUPLICATE overlapping tracks ─────────────────────────────────
        tracked_objects.sort(key=lambda x: x["id"])
        kill_set = set()
        for i in range(len(tracked_objects)):
            for j in range(i + 1, len(tracked_objects)):
                if tracked_objects[i]["id"] in kill_set:
                    break
                if self._get_iou(tracked_objects[i]["bbox"], tracked_objects[j]["bbox"]) > self.OVERLAP_IOU_KILL:
                    # Kill the newer (higher ID) track
                    kill_set.add(tracked_objects[j]["id"])
        tracked_objects = [obj for obj in tracked_objects if obj["id"] not in kill_set]

        recognitions = {}
        confidences = {}
        assigned_ids = set()  # IDs already assigned in this frame (prevent double-assign)

        # ── 4. PROCESS EACH TRACK ─────────────────────────────────────────────
        for obj in tracked_objects:
            tid = int(obj["id"])
            bbox = obj["bbox"]
            landmarks = obj.get("landmarks")

            # Initialize track record
            if tid not in self.track_registry:
                self.track_registry[tid] = {
                    "face_id": None,    # None = Unknown
                    "vote_embs": [],    # Accumulated embeddings
                    "first_seen": now_ts,
                    "last_seen": now_ts,
                }
            rec = self.track_registry[tid]
            rec["last_seen"] = now_ts

            face_id = rec["face_id"]

            # ── RULE 1: If already has face_id → DONE, reuse ──────────────────
            if face_id is not None:
                assigned_ids.add(face_id)
                recognitions[tid] = face_id
                confidences[tid] = 1.0

                # Update ghost registry with latest position
                self.ghost_registry[face_id] = {"bbox": bbox, "timestamp": now_ts}
                continue

            # ── RULE 2: Ghost Inheritance ─────────────────────────────────────
            # Check if this new track is near a recently-seen ghost (returning person)
            inherited = False
            for ghost_id, ghost_data in list(self.ghost_registry.items()):
                if ghost_id in assigned_ids:
                    continue
                age = now_ts - ghost_data["timestamp"]
                if age > self.GHOST_TTL_SECONDS:
                    continue  # Ghost expired
                dist = self._get_center_dist(bbox, ghost_data["bbox"])
                if dist <= self.GHOST_INHERIT_RADIUS:
                    # Inherit ghost's identity
                    rec["face_id"] = ghost_id
                    assigned_ids.add(ghost_id)
                    recognitions[tid] = ghost_id
                    confidences[tid] = 1.0
                    self.ghost_registry[ghost_id] = {"bbox": bbox, "timestamp": now_ts}
                    self.logger.info(f"[GHOST-INHERIT] Track {tid} → {ghost_id}")
                    inherited = True
                    break

            if inherited:
                continue

            # ── RULE 3: Accumulate Embeddings ────────────────────────────────
            x1, y1, x2, y2 = bbox
            if landmarks and len(landmarks) >= 5:
                aligned = align_face(frame, landmarks)
            else:
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                aligned = cv2.resize(crop, (112, 112)) if crop.size > 0 else None

            if aligned is not None:
                emb = self.recognizer.get_embedding(aligned)
                if emb is not None:
                    rec["vote_embs"].append(emb)

            # ── RULE 4: Register when enough embeddings collected ─────────────
            if len(rec["vote_embs"]) >= self.MIN_EMBEDDINGS_TO_REGISTER:
                mean_emb = np.mean(rec["vote_embs"], axis=0)
                # L2-normalize the mean embedding
                norm = np.linalg.norm(mean_emb)
                if norm > 1e-6:
                    mean_emb = mean_emb / norm

                matched_id, sim = self.identity_manager.match_or_register(
                    mean_emb,
                    forbidden_ids=assigned_ids,
                    current_bbox=bbox,
                    frame=frame
                )

                if matched_id and matched_id not in assigned_ids:
                    rec["face_id"] = matched_id
                    assigned_ids.add(matched_id)
                    recognitions[tid] = matched_id
                    confidences[tid] = sim
                    self.ghost_registry[matched_id] = {"bbox": bbox, "timestamp": now_ts}
                    # Log entry (EventManager deduplicates)
                    self.event_manager.log_entry(matched_id, frame, bbox)
                    self.logger.info(f"[PROD] Identity Registered: {matched_id} (sim={sim:.3f})")
                else:
                    recognitions[tid] = "Unknown"
                    confidences[tid] = 0.0
            else:
                recognitions[tid] = "Unknown"
                confidences[tid] = 0.0

        # ── 5. PRUNE STALE TRACKS ─────────────────────────────────────────────
        active_tids = {obj["id"] for obj in tracked_objects}
        stale = [
            tid for tid, rec in self.track_registry.items()
            if tid not in active_tids and (now_ts - rec["last_seen"]) > 5.0
        ]
        for tid in stale:
            rec = self.track_registry.pop(tid)
            fid = rec.get("face_id")
            if fid:
                self.event_manager.log_exit(fid)

        # ── 6. PRUNE EXPIRED GHOSTS ───────────────────────────────────────────
        expired = [gid for gid, g in self.ghost_registry.items()
                   if now_ts - g["timestamp"] > self.GHOST_TTL_SECONDS]
        for gid in expired:
            self.ghost_registry.pop(gid, None)

        return tracked_objects, recognitions, confidences, scaled_detections
