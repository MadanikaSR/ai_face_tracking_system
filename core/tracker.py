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
    """
    Production IoU Tracker.
    - IoU threshold: 0.35 (lower for smoother tracking across frames)
    - max_lost: 60 frames (~2s at 30fps) — prevents flicker re-IDs
    """
    def __init__(self, max_lost=60, iou_threshold=0.35):
        self.next_id = 0
        self.tracks = {}  # id -> {bbox, landmarks, lost}
        self.max_lost = max_lost
        self.iou_threshold = iou_threshold

    def track(self, frame, detections):
        """
        Updates tracks with new detections using greedy IoU matching.
        Returns list of currently VISIBLE (not lost) tracked objects.
        """
        new_tracks = {}
        track_ids = list(self.tracks.keys())
        matched_dets = set()
        matched_tracks = set()

        # Greedy match: each track matches best available detection
        if track_ids and detections:
            # Build IOU matrix
            iou_matrix = np.zeros((len(track_ids), len(detections)))
            for ti, tid in enumerate(track_ids):
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = compute_iou(self.tracks[tid]["bbox"], det["bbox"])

            # Greedy assignment (sort by best IoU first)
            while True:
                if iou_matrix.size == 0: break
                max_val = iou_matrix.max()
                if max_val < self.iou_threshold: break
                ti, di = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                tid = track_ids[ti]
                if tid not in matched_tracks and di not in matched_dets:
                    matched_tracks.add(tid)
                    matched_dets.add(di)
                    new_tracks[tid] = {
                        "bbox": detections[di]["bbox"],
                        "landmarks": detections[di].get("landmarks"),
                        "lost": 0
                    }
                iou_matrix[ti, :] = -1
                iou_matrix[:, di] = -1

        # Keep unmatched tracks alive (up to max_lost frames)
        for tid in track_ids:
            if tid not in matched_tracks:
                track = self.tracks[tid]
                track["lost"] += 1
                if track["lost"] <= self.max_lost:
                    new_tracks[tid] = track

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_dets:
                new_tracks[self.next_id] = {
                    "bbox": det["bbox"],
                    "landmarks": det.get("landmarks"),
                    "lost": 0
                }
                self.next_id += 1

        self.tracks = new_tracks

        # Return only currently visible tracks (lost == 0)
        return [
            {"id": tid, "bbox": data["bbox"], "landmarks": data.get("landmarks"), "lost": 0}
            for tid, data in self.tracks.items()
            if data["lost"] == 0
        ]
