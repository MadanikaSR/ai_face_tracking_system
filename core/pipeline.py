import os
import time
import numpy as np
import cv2
from datetime import datetime
from core.detector import FaceDetector
from core.tracker import FaceTracker
from core.recognizer import FaceRecognizer
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
        
        # Identity Management
        self.registered_faces = self._load_registered_faces() # face_id -> list of embeddings
        self.active_tracks = {} # track_id -> dict
        
        # Settings
        self.recognition_threshold = self.config.get("recognition_threshold", 0.6)
        self.exit_timeout = self.config.get("exit_timeout_seconds", 30)
        self.confirmation_frames = 3

    def reset_system(self):
        self.logger.info("Resetting system data...")
        self.db.clear_data()
        self.logger.clear_logs()
        Helpers.clear_directory("logs/entries")
        Helpers.clear_directory("logs/exits")
        self.registered_faces = {}
        self.active_tracks = {}
        self.logger.info("System reset complete.")

    def _load_registered_faces(self):
        gallery = self.db.get_all_faces()
        self.logger.info(f"Loaded {len(gallery)} registered identities from database.")
        return gallery

    def process_frame(self, frame, frame_count):
        detections = self.detector.detect(frame, logger=self.logger)
        tracked_objects = self.tracker.track(frame, detections)
        
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
                    "match_sim": 0.0
                }
            
            track_info = self.active_tracks[track_id]
            track_info["last_seen"] = current_time
            
            # --- STRICT TRACK-FIRST LOGIC ---
            if track_info["confirmed"]:
                # ALWAYS REUSE - NO RECOGNITION
                face_id = track_info["face_id"]
                sim = track_info["match_sim"]
                if frame_count % 30 == 0: # Periodic log to avoid spam
                    self.logger.debug(f"Track {track_id} -> {face_id} (reused)")
            else:
                x1, y1, x2, y2 = bbox
                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                aligned_face = align_face(frame, landmarks) if landmarks else cv2.resize(face_crop, (112, 112))
                
                if aligned_face is not None:
                    emb = self.recognizer.get_embedding(aligned_face)
                    if emb is not None:
                        track_info["vote_embs"].append(emb)
                    
                    if len(track_info["vote_embs"]) >= self.confirmation_frames:
                        # Confirm Identity
                        mean_emb = np.mean(track_info["vote_embs"], axis=0)
                        face_id, sim = self._match_or_register(mean_emb, ids_assigned_this_frame)
                        
                        track_info["face_id"] = face_id
                        track_info["confirmed"] = True
                        track_info["match_sim"] = float(sim)
                        
                        image_path = Helpers.save_crop(frame, bbox, "logs", face_id, "entries")
                        self.db.log_event(face_id, "entry", image_path)
                        
                        if sim >= self.recognition_threshold:
                            self.logger.info(f"New track {track_id} -> matched Face_ID {face_id} ({sim:.2f})")
                        else:
                            self.logger.info(f"New face registered -> {face_id} (from track {track_id})")
                
                face_id = track_info["face_id"]
                sim = track_info["match_sim"]

            recognitions[track_id] = face_id
            confidences[track_id] = sim
            ids_assigned_this_frame.add(face_id)
            
            if track_info["confirmed"] and face_id != "Unknown":
                self.db.update_face_last_seen(face_id)

        # --- Cleanup Exits ---
        exited_tracks = []
        for tid, data in self.active_tracks.items():
            diff = (current_time - data["last_seen"]).total_seconds()
            if diff > self.exit_timeout:
                exited_tracks.append(tid)
        
        for tid in exited_tracks:
            data = self.active_tracks.pop(tid)
            if data["confirmed"] and data["face_id"] != "Unknown":
                self.db.log_event(data["face_id"], "exit", "logs/exits/last_known.jpg")
                self.logger.info(f"Identity {data['face_id']} exited.")

        return tracked_objects, recognitions, confidences

    def _match_or_register(self, embedding, forbidden_ids):
        if embedding is None: return "Unknown", 0.0

        best_match_id = None
        max_global_sim = -1.0
        
        for face_id, gallery in self.registered_faces.items():
            if face_id in forbidden_ids: continue
            
            similarities = [self.recognizer.compare_embeddings(embedding, g_emb) for g_emb in gallery]
            face_max_sim = max(similarities) if similarities else 0.0
            
            if face_max_sim > self.recognition_threshold and face_max_sim > max_global_sim:
                max_global_sim = face_max_sim
                best_match_id = face_id

        if best_match_id:
            if max_global_sim < 0.9 and len(self.registered_faces[best_match_id]) < 5:
                self.registered_faces[best_match_id].append(embedding)
                self.db.add_embedding(best_match_id, embedding.tobytes())
            return best_match_id, max_global_sim
        else:
            new_id = f"Face_{int(time.time())}_{len(self.registered_faces)}"
            self.db.add_face(new_id, embedding.tobytes())
            self.registered_faces[new_id] = [embedding]
            return new_id, 1.0
