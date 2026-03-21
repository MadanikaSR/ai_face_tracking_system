import os
import time
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
from utils.helpers import Helpers
from utils.alignment import align_face

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.db = Database()
        self.logger = SystemLogger()
        self.detector = FaceDetector()
        self.tracker = FaceTracker()
        self.recognizer = FaceRecognizer()
        
        # Modular Management
        self.identity_manager = IdentityManager(config, self.db, self.recognizer)
        self.event_manager = EventManager(self.db, self.logger)
        self.active_tracks = {} # track_id -> dict
        
        # Settings - Polished for Absolute Stability
        self.exit_timeout = self.config.get("exit_timeout_seconds", 30)
        self.confirmation_frames = 5 # Require 5 stable frames for identity
        self.min_emb_norm = 18.0 # Filter non-face objects

    def reset_system(self):
        self.logger.info("Resetting system data...")
        self.db.clear_data()
        self.logger.clear_logs()
        self.event_manager.clear_all()
        self.identity_manager = IdentityManager(self.config, self.db, self.recognizer)
        self.active_tracks = {}
        self.logger.info("System reset complete.")

    def process_frame(self, frame, frame_count):
        # 1. High Resolution Pre-Processing
        h_orig, w_orig = frame.shape[:2]
        target_res = (1280, 720)
        det_frame = cv2.resize(frame, target_res, interpolation=cv2.INTER_LINEAR)
        h_det, w_det = det_frame.shape[:2]
        
        # 2. Detect & Track
        detections = self.detector.detect(det_frame, logger=self.logger)
        
        scale_x = w_orig / w_det
        scale_y = h_orig / h_det
        mapped_detections = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            mapped_bbox = [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]
            mapped_landmarks = [[lx * scale_x, ly * scale_y] for lx, ly in det["landmarks"]]
            mapped_detections.append({
                "bbox": mapped_bbox,
                "confidence": det["confidence"],
                "landmarks": mapped_landmarks,
                "class": det["class"]
            })

        tracked_objects = self.tracker.track(frame, mapped_detections)
        
        current_time = datetime.now()
        recognitions = {}
        confidences = {}
        ids_assigned_this_frame = set()

        for obj in tracked_objects:
            track_id = int(obj["id"])
            bbox = obj["bbox"]
            landmarks = obj.get("landmarks")
            
            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = {
                    "last_seen": current_time,
                    "face_id": "Unknown",
                    "confirmed": False,
                    "vote_embs": [],
                    "match_sim": 0.0,
                    "detection_count": 0,
                    "last_bbox": bbox
                }
            
            track_info = self.active_tracks[track_id]
            track_info["last_seen"] = current_time
            track_info["detection_count"] += 1
            track_info["last_bbox"] = bbox
            
            if track_info["confirmed"]:
                face_id = track_info["face_id"]
                sim = track_info["match_sim"]
            else:
                # Require 2 frames before even attempting alignment/embedding
                if track_info["detection_count"] >= 2:
                    x1, y1, x2, y2 = bbox
                    face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                    aligned_face = align_face(frame, landmarks) if landmarks else cv2.resize(face_crop, (112, 112))
                    
                    if aligned_face is not None:
                        emb = self.recognizer.get_embedding(aligned_face)
                        if emb is not None:
                            # Quality Check: filter non-face blobs
                            if np.linalg.norm(emb) > self.min_emb_norm:
                                track_info["vote_embs"].append(emb)
                    
                    # Confirm after 5 stable frames
                    if len(track_info["vote_embs"]) >= self.confirmation_frames:
                        mean_emb = np.mean(track_info["vote_embs"], axis=0)
                        # Use spatiotemporal bias (bbox) to prevent duplicate IDs
                        face_id, sim = self.identity_manager.match_or_register(mean_emb, ids_assigned_this_frame, current_bbox=bbox)
                        
                        track_info["face_id"] = face_id
                        track_info["confirmed"] = True
                        track_info["match_sim"] = float(sim)
                        
                        self.event_manager.log_entry(face_id, frame, bbox)
                        
                        if sim >= self.identity_manager.recognition_threshold:
                            self.logger.info(f"[RE-ID] Track {track_id} -> Face {face_id} (sim: {sim:.2f})")
                        else:
                            self.logger.info(f"[NEW] Track {track_id} -> Registered as {face_id}")
                
                face_id = track_info["face_id"]
                sim = track_info["match_sim"]

            recognitions[track_id] = face_id
            confidences[track_id] = sim
            ids_assigned_this_frame.add(face_id)
            
            if track_info["confirmed"] and face_id != "Unknown":
                self.db.update_face_last_seen(face_id)

        # Handle Exits
        exited_tracks = []
        for tid, data in self.active_tracks.items():
            if (current_time - data["last_seen"]).total_seconds() > self.exit_timeout:
                exited_tracks.append(tid)
        
        for tid in exited_tracks:
            data = self.active_tracks.pop(tid)
            if data["confirmed"] and data["face_id"] != "Unknown":
                # Cache exit for spatiotemporal re-id
                self.identity_manager.add_exit_cache(data["face_id"], data["last_bbox"])
                self.event_manager.log_exit(data["face_id"])

        return tracked_objects, recognitions, confidences, mapped_detections
