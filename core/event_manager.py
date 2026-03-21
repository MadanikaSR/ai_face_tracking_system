import os
from datetime import datetime
from utils.helpers import Helpers

class EventManager:
    def __init__(self, db, logger):
        self.db = db
        self.logger = logger
        self.log_dir = "logs"
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(f"{self.log_dir}/entries", exist_ok=True)
        os.makedirs(f"{self.log_dir}/exits", exist_ok=True)

    def log_entry(self, face_id, frame, bbox):
        """Logs a new person entering the scene."""
        image_path = Helpers.save_crop(frame, bbox, self.log_dir, face_id, "entries")
        self.db.log_event(face_id, "entry", image_path)
        return image_path

    def log_exit(self, face_id):
        """Logs a person exiting the scene."""
        # Note: Specific exit crop is usually just the last known position, 
        # but for hackathons, a database entry is often enough.
        self.db.log_event(face_id, "exit", f"{self.log_dir}/exits/last_known.jpg")
        self.logger.info(f"Identity {face_id} exited scene.")

    def clear_all(self):
        Helpers.clear_directory(f"{self.log_dir}/entries")
        Helpers.clear_directory(f"{self.log_dir}/exits")
