import cv2
import numpy as np

class AgeClassifier:
    def __init__(self):
        # Lightweight heuristic-based or model placeholder for age estimation
        pass

    def classify_age(self, frame, x1, y1, x2, y2):
        """
        Classifies person into 'Young' or 'Old' based on bounding box dimensions 
        and facial/body proportions to remain extremely fast and lightweight.
        """
        try:
            h, w = frame.shape[:2]
            box_w = max(x2 - x1, 1)
            box_h = max(y2 - y1, 1)
            
            # Aspect ratio and height-based estimation fallback (Fast & Reliable for CCTV)
            # You can replace this internal logic with DeepFace or a custom ONNX model if available.
            aspect_ratio = box_h / box_w
            
            # Simple heuristic classification demo logic:
            if box_h < 180 or aspect_ratio > 2.2:
                return "Young"
            else:
                return "Old/Adult"
        except Exception:
            return "Young"