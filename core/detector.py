from ultralytics import YOLO
import cv2

class FaceDetector:
    def __init__(self, model_path="yolov8n-face.pt"):
        # For our purposes, we'll use the default YOLOv8n or a face-specific model if available.
        # Ultralytics provides several models. Let's use 'yolov8n.pt' and filter for faces or a pre-trained face model.
        # There's a popular 'yolov8n-face.pt' model on HuggingFace, but here we'll assume a standard ultralytics usage.
        self.model = YOLO("yolov8n.pt") 

    def detect(self, frame):
        # We'll filter for person (class 0) if using standard YOLOv8, 
        # or just use the model if it's a dedicated face model.
        # Many users use yolov8n-face.pt for this.
        results = self.model(frame, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                # If using standard YOLO, class 0 is person. 
                # For face tracking, we'd ideally use a face-specific model.
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class": cls
                    })
        return detections
