import cv2
import numpy as np
from typing import Optional, Tuple
from app.core.config import MODE

class FaceDetector:
    """Face detection and alignment pipeline."""
    
    def __init__(self):
        if MODE == "heur":
            self.haar_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
    
    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face bounding box. Returns (x, y, w, h) or None."""
        if MODE == "mock":
            # Return a mock bounding box in center
            h, w = image.shape[:2]
            size = min(h, w) // 2
            x = (w - size) // 2
            y = (h - size) // 2
            return (x, y, size, size)
        
        elif MODE == "heur":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            faces = self.haar_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                return tuple(faces[0].astype(int))
            return None
        
        elif MODE == "onnx":
            # Stub: would use MTCNN or RetinaFace ONNX
            h, w = image.shape[:2]
            size = min(h, w) // 2
            x = (w - size) // 2
            y = (h - size) // 2
            return (x, y, size, size)
        
        return None
    
    def align(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop and align face to 112x112."""
        x, y, w, h = bbox
        # Simple crop and resize
        face = image[y:y+h, x:x+w]
        face_aligned = cv2.resize(face, (112, 112))
        return face_aligned

