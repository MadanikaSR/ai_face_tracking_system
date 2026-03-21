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
        
        self.identity_manager = IdentityManager(config, self.db, self.recognizer)
        self.event_manager = EventManager(self.db, self.logger)
        
        self.active_tracks = {} # track_id -> dict
        self.global_registry = {} # face_id -> track_id
        
        # Fast Noise Heatmap
        self.spatial_noise = {} # grid -> {"frames": 0, "start_pos": []}
        self.static_blacklist = set()
        
        self.detection_skip = self.config.get("detection_frame_skip", 2)
        self.exit_timeout = self.config.get("exit_timeout_seconds", 120)
        self.min_emb_norm = 4.0

    def _get_grid(self, bbox):
        return ( (bbox[0]+bbox[2])//40, (bbox[1]+bbox[3])//40 )

    def reset_system(self):
        self.logger.info("Resetting system data...")
        self.db.clear_data()
        self.logger.clear_logs()
        self.event_manager.clear_all()
        self.identity_manager = IdentityManager(self.config, self.db, self.recognizer)
        self.active_tracks = {}
        self.global_registry = {}
        self.spatial_noise = {}
        self.static_blacklist = set()
        self.logger.info("System reset complete.")

    def _enhance_crop(self, crop):
        """Enhance contrast for Purdha/Low-light recovery."""
        if crop is None or crop.size == 0: return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def process_frame(self, frame, frame_count):
        h_orig, w_orig = frame.shape[:2]
        det_frame = cv2.resize(frame, (1280, 720))
        h_det, w_det = det_frame.shape[:2]
        
        mapped_detections = []
        if frame_count % self.detection_skip == 0:
            raw_detections = self.detector.detect(det_frame, logger=self.logger)
            scale_x, scale_y = w_orig / w_det, h_orig / h_det
            for det in raw_detections:
                x1, y1, x2, y2 = det["bbox"]
                bbox = [int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)]
                
                # RAPID CRACK KILLER (60 Frames / 2 Seconds)
                grid = self._get_grid(bbox)
                if grid in self.static_blacklist: continue
                
                if grid not in self.spatial_noise:
                    self.spatial_noise[grid] = {"frames": 0, "start_pos": bbox[:2]}
                self.spatial_noise[grid]["frames"] += 1
                
                if self.spatial_noise[grid]["frames"] > 60:
                    dist = np.linalg.norm(np.array(bbox[:2]) - np.array(self.spatial_noise[grid]["start_pos"]))
                    if dist < 15.0:
                        self.static_blacklist.add(grid)
                        self.logger.info(f"[ANNIHILATOR] Killed static noise zone {grid} (Floor Crack)")
                        continue

                mapped_detections.append({
                    "bbox": bbox, "confidence": det["confidence"], 
                    "landmarks": [[lx*scale_x, ly*scale_y] for lx, ly in det["landmarks"]] if det["landmarks"] else None,
                    "class": 0
                })

        tracked_objects = self.tracker.track(frame, mapped_detections)
        current_time = datetime.now()
        recognitions, confidences = {}, {}
        ids_this_frame = set()

        # Larger boxes = higher priority for ID assignment
        tracked_objects.sort(key=lambda x: (x["bbox"][2]-x["bbox"][0])*(x["bbox"][3]-x["bbox"][1]), reverse=True)

        for obj in tracked_objects:
            track_id = int(obj["id"])
            bbox, landmarks = obj["bbox"], obj.get("landmarks")
            
            if track_id not in self.active_tracks:
                self.active_tracks[track_id] = {
                    "first_seen": bbox, "last_seen": current_time,
                    "face_id": "Unknown", "confirmed": False, "vote_embs": [],
                    "match_sim": 0.0, "frames": 0, "max_norm": 0.0, "is_dead": False
                }
            
            track = self.active_tracks[track_id]
            if track["is_dead"]:
                recognitions[track_id], confidences[track_id] = "Unknown", 0.0
                continue

            track["last_seen"] = current_time
            track["frames"] += 1
            
            face_id, sim = "Unknown", 0.0

            if track["confirmed"]:
                fid = track["face_id"]
                # ABSOLUTE LOCKDOWN: Prevent duplicate handles for same ID
                if fid in ids_this_frame:
                    track["is_dead"] = True
                    self.logger.info(f"[LOCKDOWN] Killing redundant track {track_id} for ID {fid}")
                else:
                    face_id, sim = fid, track["match_sim"]
                    ids_this_frame.add(fid)
            else:
                # BIOMETRICS
                x1, y1, x2, y2 = bbox
                if landmarks:
                    aligned = align_face(frame, landmarks)
                else:
                    # Purdha Fallback + Enhanced Contrast
                    raw_crop = frame[max(0,y1):y2, max(0,x1):x2]
                    enhanced = self._enhance_crop(raw_crop)
                    aligned = cv2.resize(enhanced, (112, 112)) if enhanced is not None else None
                
                if aligned is not None:
                    emb = self.recognizer.get_embedding(aligned)
                    if emb is not None:
                        norm = np.linalg.norm(emb)
                        if norm > self.min_emb_norm:
                            track["vote_embs"].append(emb)
                            track["max_norm"] = max(track["max_norm"], norm)

                # STABILITY GATE
                target = 5 if track["max_norm"] >= 18.0 else 20
                if len(track["vote_embs"]) >= target:
                    dist = np.linalg.norm(np.array(bbox[:2]) - np.array(track["first_seen"][:2]))
                    if dist > 15.0:
                        mean_emb = np.mean(track["vote_embs"], axis=0)
                        mid, msim = self.identity_manager.match_or_register(mean_emb, ids_this_frame, current_bbox=bbox)
                        
                        if mid not in ids_this_frame:
                            track["face_id"], track["match_sim"], track["confirmed"] = mid, float(msim), True
                            self.global_registry[mid] = track_id
                            ids_this_frame.add(mid)
                            self.event_manager.log_entry(mid, frame, bbox)
                            self.logger.info(f"[ABSOLUTE-ID] {mid} @ {msim:.2f} (mov: {dist:.1f}px)")
                            face_id, sim = mid, msim
                        else:
                            track["is_dead"] = True
                    elif track["frames"] > 60: # Kill static noise
                        track["is_dead"] = True

            recognitions[track_id], confidences[track_id] = face_id, sim

        # Cleanup Exits
        exited = []
        for tid, data in self.active_tracks.items():
            if (current_time - data["last_seen"]).total_seconds() > self.exit_timeout: exited.append(tid)
        for tid in exited:
            data = self.active_tracks.pop(tid)
            if data["confirmed"] and not data["is_dead"]:
                self.identity_manager.add_exit_cache(data["face_id"], data["last_bbox"] if "last_bbox" in data else data["first_seen"])
                self.event_manager.log_exit(data["face_id"])

        return tracked_objects, recognitions, confidences, mapped_detections
