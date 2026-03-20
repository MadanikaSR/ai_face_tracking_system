from ultralytics import YOLO

class FaceTracker:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def track(self, frame, persist=True):
        # Ultralytics has built-in tracking. 
        # It's better to use it directly on the frame.
        results = self.model.track(frame, persist=persist, tracker="bytetrack.yaml", verbose=False)
        tracked_objects = []
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            for box, obj_id in zip(boxes, ids):
                tracked_objects.append({
                    "bbox": box.tolist(),
                    "id": int(obj_id)
                })
        return tracked_objects
