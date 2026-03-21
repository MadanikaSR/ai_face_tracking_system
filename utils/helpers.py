import os
import cv2
from datetime import datetime

class Helpers:
    @staticmethod
    def save_crop(image, bbox, folder_path, face_id, event_type):
        """Saves a padded face crop to the specified folder."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        pad_x = int(w * 0.20)
        pad_y = int(h * 0.20)
        img_h, img_w = image.shape[:2]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_w, x2 + pad_x)
        y2 = min(img_h, y2 + pad_y)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        date_str = datetime.now().strftime("%Y-%m-%d")
        full_dir = os.path.join(folder_path, event_type, date_str)
        os.makedirs(full_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%H-%M-%S-%f")
        filename = f"{face_id}_{timestamp}.jpg"
        file_path = os.path.join(full_dir, filename)
        cv2.imwrite(file_path, crop)
        return file_path

    @staticmethod
    def draw_detections(frame, tracked_objects, recognitions, confidences=None, raw_detections=None):
        """
        Draws bounding boxes and labels on the frame.
        DEBUG LAYER: raw_detections drawn in RED (before validation)
        TRACK LAYER: validated tracks drawn in GREEN
        """
        # 1. Raw detections — RED boxes (debug: shows what SCRFD sees before filtering)
        if raw_detections:
            for det in raw_detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det.get("confidence", 0.0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 1)
                cv2.putText(frame, f"RAW {conf:.2f}", (x1, y2 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 200), 1)

        # 2. Tracked + ID-assigned faces — GREEN boxes
        for obj in tracked_objects:
            bbox     = obj["bbox"]
            track_id = obj["id"]
            face_id  = recognitions.get(track_id, "Unknown")
            sim      = confidences.get(track_id, 0.0) if confidences else 0.0
            x1, y1, x2, y2 = bbox

            color = (0, 200, 0) if face_id != "Unknown" else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Short face_id display (last 3 chars of ID suffix)
            short_id = face_id.split("_")[-1] if "_" in face_id else face_id
            label = f"T{track_id}|ID{short_id}"
            if 0 < sim < 1.0:
                label += f" {sim:.2f}"

            # Text background for readability
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame

    @staticmethod
    def clear_directory(directory_path):
        """Recursively deletes all files and subdirectories."""
        import shutil
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception:
                    pass

    @staticmethod
    def is_blurry(image, threshold=30):
        if image is None or image.size == 0: return True
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    @staticmethod
    def is_exposure_okay(image, min_brightness=40, max_brightness=220):
        if image is None or image.size == 0: return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        return min_brightness < brightness < max_brightness
