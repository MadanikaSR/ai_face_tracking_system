import os
import cv2
from datetime import datetime

class Helpers:
    @staticmethod
    def save_crop(image, bbox, folder_path, face_id, event_type):
        """Saves a cropped face image to the specified folder."""
        x1, y1, x2, y2 = bbox
        # Ensure bbox is within image boundaries
        h, w, _ = image.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
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
    def draw_detections(frame, tracked_objects, recognitions, confidences=None):
        """Draws bounding boxes and labels on the frame."""
        for obj in tracked_objects:
            bbox = obj["bbox"]
            obj_id = obj["id"]
            name = recognitions.get(obj_id, f"ID: {obj_id}")
            
            # Get confidence score if provided
            conf_str = ""
            if confidences and obj_id in confidences:
                conf_str = f" ({confidences[obj_id]:.2f})"
            
            label = f"{name}{conf_str}"
            
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
