import os
import time
import numpy as np
from datetime import datetime
from core.detector import FaceDetector
from core.tracker import FaceTracker
from core.recognizer import FaceRecognizer
from database.db import Database
from logging_system.logger import SystemLogger
from utils.helpers import Helpers

class Pipeline:
    def __init__(self, config):
        self.config = config
        self.db = Database()
        self.logger = SystemLogger()
        self.detector = FaceDetector()
        self.tracker = FaceTracker()
        self.recognizer = FaceRecognizer()
        
        self.active_tracks = {} # track_id -> {last_seen: timestamp, face_id: string}
        self.face_id_map = {} # track_id -> known_face_id
        self.registered_faces = self._load_registered_faces()

    def reset_system(self):
        """Resets the entire system: DB, logs, and images."""
        self.logger.info("Resetting system data...")
        # 1. Clear database
        self.db.clear_data()
        # 2. Clear logs
        self.logger.clear_logs()
        # 3. Clear images
        Helpers.clear_directory("logs/entries")
        Helpers.clear_directory("logs/exits")
        # 4. Reload (empty) registered faces
        self.registered_faces = {}
        self.active_tracks = {}
        self.face_id_map = {}
        self.logger.info("System reset complete.")

    def _load_registered_faces(self):
        """Loads all known faces from the database."""
        faces = self.db.get_all_faces()
        self.logger.info(f"Loaded {len(faces)} registered faces from database.")
        return {f[0]: np.frombuffer(f[1], dtype=np.float32) for f in faces}

    def process_frame(self, frame, frame_count):
        tracked_objects = self.tracker.track(frame)
        
        current_time = datetime.now()
        recognitions = {}
        confidences = {}
        ids_assigned_this_frame = set()

        for obj in tracked_objects:
            track_id = obj["id"]
            bbox = obj["bbox"]
            
            if track_id not in self.active_tracks:
                # New track identified
                x1, y1, x2, y2 = bbox
                face_crop = frame[y1:y2, x1:x2]
                embedding = self.recognizer.get_embedding(face_crop)
                
                # Match or register, ensuring ID is not already used in this frame
                face_id, confidence = self._match_or_register(embedding, ids_assigned_this_frame)
                
                self.active_tracks[track_id] = {"last_seen": current_time, "face_id": face_id}
                ids_assigned_this_frame.add(face_id)
                
                # Log entry
                image_path = Helpers.save_crop(frame, bbox, "logs", face_id, "entries")
                self.db.log_event(face_id, "entry", image_path)
                self.logger.info(f"Face {face_id} entered (similarity: {confidence:.2f}).")
            else:
                self.active_tracks[track_id]["last_seen"] = current_time
                face_id = self.active_tracks[track_id]["face_id"]
                ids_assigned_this_frame.add(face_id)
                confidence = 1.0 # Existing track
            
            recognitions[track_id] = face_id
            confidences[track_id] = confidence
            self.db.update_face_last_seen(face_id)

        # Handle Exits
        exited_tracks = []
        for track_id, data in self.active_tracks.items():
            if (current_time - data["last_seen"]).total_seconds() > self.config["exit_timeout_seconds"]:
                exited_tracks.append(track_id)
        
        for track_id in exited_tracks:
            data = self.active_tracks.pop(track_id)
            face_id = data["face_id"]
            self.db.log_event(face_id, "exit", "logs/exits/last_known.jpg")
            self.logger.info(f"Face {face_id} exited (timeout).")

        return tracked_objects, recognitions, confidences

    def _match_or_register(self, embedding, forbidden_ids):
        if embedding is None:
            return "Unknown", 0.0

        best_match_id = None
        max_similarity = -1.0
        threshold = self.config.get("recognition_threshold", 0.6)

        # 1. Look for the best match among registered faces
        for face_id, known_emb in self.registered_faces.items():
            # Skip if this ID is already assigned to someone else in the same frame
            if face_id in forbidden_ids:
                continue
                
            sim = self.recognizer.compare_embeddings(embedding, known_emb)
            if sim > threshold and sim > max_similarity:
                max_similarity = sim
                best_match_id = face_id

        if best_match_id:
            return best_match_id, max_similarity
        else:
            # 2. Register new face if no match found (or all matches are forbidden)
            new_id = f"Face_{int(time.time())}_{len(self.registered_faces)}"
            self.db.add_face(new_id, embedding.tobytes())
            self.registered_faces[new_id] = embedding
            return new_id, 1.0
