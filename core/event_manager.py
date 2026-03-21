import os
from datetime import datetime
from utils.helpers import Helpers

class EventManager:
    """
    Production Event Manager.
    - Deduplicates entry logs: each face_id logged ONCE per session.
    - Saves high-quality face crop on entry.
    - Logs exit event when track disappears.
    """
    def __init__(self, db, logger):
        self.db = db
        self.logger = logger
        self.log_dir = "logs"
        self._logged_entries = set()  # face_ids already logged this session
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(f"{self.log_dir}/entries", exist_ok=True)
        os.makedirs(f"{self.log_dir}/exits", exist_ok=True)

    def log_entry(self, face_id, frame, bbox):
        """Logs a new person entering the scene. Only once per face_id per session."""
        if face_id in self._logged_entries:
            return None  # Already logged — do NOT duplicate
        self._logged_entries.add(face_id)
        image_path = Helpers.save_crop(frame, bbox, self.log_dir, face_id, "entries")
        self.db.log_event(face_id, "entry", image_path)
        return image_path

    def log_exit(self, face_id):
        """Logs a person exiting the scene."""
        self.db.log_event(face_id, "exit", f"{self.log_dir}/exits/{face_id}_exit.jpg")
        self.logger.info(f"Identity {face_id} exited scene.")

    def clear_all(self):
        self._logged_entries.clear()
        Helpers.clear_directory(f"{self.log_dir}/entries")
        Helpers.clear_directory(f"{self.log_dir}/exits")
