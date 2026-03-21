import onnxruntime as ort
import numpy as np
import cv2
import os

class SCRFD:
    def __init__(self, model_path="models/scrfd_500m_bnkps.onnx"):
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            print(f"Error: SCRFD model not found at {self.model_path}")
            self.session = None
            return

        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [x.name for x in self.session.get_outputs()]
        
        self.input_size = (640, 640)
        self.strides = [8, 16, 32]
        self._num_anchors = 2

    def preprocess(self, img):
        img = cv2.resize(img, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def detect(self, img, conf_threshold=0.25):
        if self.session is None: return []
        
        h_orig, w_orig = img.shape[:2]
        blob = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input_name: blob})
        
        return self._decode_outputs(outputs, conf_threshold, (h_orig, w_orig))

    def _decode_outputs(self, outputs, threshold, original_size):
        all_bboxes = []
        all_kpss = []
        all_scores = []
        
        h_orig, w_orig = original_size
        scale_x = w_orig / self.input_size[0]
        scale_y = h_orig / self.input_size[1]

        for i, stride in enumerate(self.strides):
            # SCRFD outputs usually have a batch dimension (1, N, C)
            # We take index 0 to get (N, C)
            score = outputs[i][0]
            bbox = outputs[i + 3][0]
            kps = outputs[i + 6][0]
            
            height, width = self.input_size[1] // stride, self.input_size[0] // stride
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers]*self._num_anchors, axis=1).reshape((-1,2))

            # Now score is (N, 1)
            pos_inds = np.where(score.flatten() >= threshold)[0]
            for idx in pos_inds:
                s = float(score[idx][0])
                reg = bbox[idx] * stride
                k = kps[idx] * stride
                cx, cy = anchor_centers[idx]
                
                # BBox: [x1, y1, x2, y2]
                x1 = (cx - reg[0]) * scale_x
                y1 = (cy - reg[1]) * scale_y
                x2 = (cx + reg[2]) * scale_x
                y2 = (cy + reg[3]) * scale_y
                
                w = float(x2 - x1)
                h = float(y2 - y1)
                
                all_bboxes.append([float(x1), float(y1), w, h])
                all_scores.append(s)
                
                # Landmarks
                landmarks = []
                for j in range(0, 10, 2):
                    lx = float((cx + k[j]) * scale_x)
                    ly = float((cy + k[j+1]) * scale_y)
                    landmarks.append([lx, ly])
                all_kpss.append(landmarks)

        if not all_bboxes: return []
        
        # NMS — IoU 0.60: only suppress if faces overlap heavily (>60%)
        # Higher = LESS aggressive = more faces kept for adjacent twins
        indices = cv2.dnn.NMSBoxes(all_bboxes, all_scores, threshold, 0.60)
        
        results = []
        if len(indices) > 0:
            # Handle different OpenCV NMS return formats
            it_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
            for i in it_indices:
                x, y, w, h = all_bboxes[i]
                results.append({
                    "bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "landmarks": all_kpss[i],
                    "confidence": float(all_scores[i])
                })
        return results
