import cv2
import numpy as np

def get_similarity_matrix(src_pts, dst_pts):
    """Computes a similarity transformation matrix."""
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    return cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]

def align_face(img, landmarks):
    """Aligns a face image to a standard 112x112 template using 5 landmarks."""
    # Standard InsightFace reference points (112x112)
    dst_pts = np.array([
        [30.2946, 51.6963],  # Left Eye
        [65.5318, 51.5014],  # Right Eye
        [48.0252, 71.7366],  # Nose
        [33.5493, 92.3655],  # Left Mouth
        [62.7299, 92.2041]   # Right Mouth
    ], dtype=np.float32)

    # Landmarks are expected as list of 5 points [[x, y], ...]
    src_pts = np.array(landmarks, dtype=np.float32)
    
    # Compute similarity transform
    affine_matrix = get_similarity_matrix(src_pts, dst_pts)
    
    if affine_matrix is None:
        return None
        
    # Warp image to 112x112
    aligned_face = cv2.warpAffine(img, affine_matrix, (112, 112), borderValue=0.0)
    return aligned_face
