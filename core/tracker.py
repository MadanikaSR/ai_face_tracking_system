import numpy as np

def compute_iou(box1, box2):
    """Computes IOU between two bboxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / float(area1 + area2 - intersection + 1e-6)

class FaceTracker:
    def __init__(self, max_lost=30):
        self.next_id = 0
        self.tracks = {} # id -> {bbox, landmarks, lost}
        self.max_lost = max_lost # Persistent tracking across lost frames

    def track(self, frame, detections):
        """
        Updates tracks with new detections using IOU matching.
        detections: list of {"bbox": [x1,y1,x2,y2], "landmarks": [...], "confidence": ...}
        """
        new_tracks = {}
        track_ids = list(self.tracks.keys())
        
        # 1. Match current tracks with new detections
        matched_dets = set()
        matched_tracks = set()
        
        if track_ids:
            for tid_idx, tid in enumerate(track_ids):
                t_bbox = self.tracks[tid]["bbox"]
                best_iou = 0
                best_det_idx = -1
                
                for d_idx, det in enumerate(detections):
                    if d_idx in matched_dets: continue
                    iou = compute_iou(t_bbox, det["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = d_idx
                
                if best_iou > 0.3: # Match found
                    matched_tracks.add(tid)
                    matched_dets.add(best_det_idx)
                    new_tracks[tid] = {
                        "bbox": detections[best_det_idx]["bbox"],
                        "landmarks": detections[best_det_idx]["landmarks"],
                        "lost": 0
                    }
        
        # 2. Handle unmatched tracks (keep them alive if lost < max_lost)
        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track["lost"] += 1
                if track["lost"] <= self.max_lost:
                    new_tracks[tid] = track
        
        # 3. Handle unmatched detections (new tracks)
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                new_tracks[self.next_id] = {
                    "bbox": det["bbox"],
                    "landmarks": det["landmarks"],
                    "lost": 0
                }
                self.next_id += 1
        
        self.tracks = new_tracks
        
        # Return currently visible tracks
        tracked_objects = []
        for tid, data in self.tracks.items():
            if data["lost"] == 0:
                tracked_objects.append({
                    "id": tid,
                    "bbox": data["bbox"],
                    "landmarks": data.get("landmarks")
                })
        return tracked_objects
