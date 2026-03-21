import os
import cv2
from datetime import datetime

class Helpers:
    @staticmethod
    def save_crop(image, bbox, folder_path, face_id, event_type):
        """Saves a padded face crop to the specified folder."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        
        # Add 20% padding
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        
        # Apply padding and clamp to image boundaries
        img_h, img_w, _ = image.shape
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_w, x2 + pad_x)
        y2 = min(img_h, y2 + pad_y)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
            
        date_str = datetime.now().strftime("%Y-%m-%d")
        full_dir = os.path.join(folder_path, event_type, date_str)
        if not os.path.exists(full_dir):
            os.makedirs(full_dir)
            
        timestamp = datetime.now().strftime("%H-%M-%S-%f")
        filename = f"{face_id}_{timestamp}.jpg"
        file_path = os.path.join(full_dir, filename)
        
        cv2.imwrite(file_path, crop)
        return file_path

    @staticmethod
    def draw_detections(frame, tracked_objects, recognitions, confidences=None, raw_detections=None):
        """Draws bounding boxes and labels on the frame."""
        # 1. Draw Raw Detections in Red (Debug)
        if raw_detections:
            for det in raw_detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frame, f"{conf:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 2. Draw Tracked Objects in Green
        for obj in tracked_objects:
            bbox = obj["bbox"]
            track_id = obj["id"]
            face_id = recognitions.get(track_id, "Unknown")
            sim = confidences.get(track_id, 0.0) if confidences else 0.0
            
            label = f"T:{track_id} F:{face_id}"
            if 0 < sim < 1.0:
                label += f" ({sim:.2f})"
            
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    @staticmethod
    def clear_directory(directory_path):
        """Recursively deletes all files and subdirectories in a directory."""
        import shutil
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    @staticmethod
    def is_blurry(image, threshold=30):
        """Checks if an image is blurry using Laplacian variance."""
        if image is None or image.size == 0: return True
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < threshold

    @staticmethod
    def is_exposure_okay(image, min_brightness=40, max_brightness=220):
        """Checks if an image is too dark or overexposed."""
        if image is None or image.size == 0: return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        return min_brightness < brightness < max_brightness
