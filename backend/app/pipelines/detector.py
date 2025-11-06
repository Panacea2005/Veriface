import cv2
import numpy as np
from typing import Optional, Tuple
from app.core.config import MODE

class FaceDetector:
    """Face detection and alignment pipeline."""
    
    def __init__(self):
        # Heuristic Haar for fallback
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        # DeepFace availability
        try:
            from deepface import DeepFace  # noqa: F401
            self.use_deepface = True
        except Exception:
            self.use_deepface = False
    
    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face bounding box. Returns (x, y, w, h) or None."""
        # Prefer DeepFace across modes for robustness
        if self.use_deepface:
            try:
                from deepface import DeepFace
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
                # Use OpenCV backend to avoid extra heavy models; align for stable crops
                faces = DeepFace.extract_faces(
                    img_path=rgb,
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True
                )
                # DeepFace may return list of dicts (with 'facial_area') or np arrays depending on version
                if isinstance(faces, list) and len(faces) > 0:
                    first = faces[0]
                    region = None
                    if isinstance(first, dict):
                        # new API often has 'facial_area' or 'region'
                        region = first.get("facial_area") or first.get("region")
                    # region expected as dict with x,y,w,h
                    if isinstance(region, dict):
                        x = int(region.get("x", 0))
                        y = int(region.get("y", 0))
                        w = int(region.get("w", 0))
                        h = int(region.get("h", 0))
                        if w > 0 and h > 0:
                            return (x, y, w, h)
                # fallback to Haar if DeepFace didn't give a region
            except Exception:
                pass

        # Heuristic fallback (Haar)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.haar_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            return tuple(faces[0].astype(int))
        
        return None
    
    def align(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop and align face to 112x112."""
        x, y, w, h = bbox
        # Simple crop and resize
        face = image[y:y+h, x:x+w]
        face_aligned = cv2.resize(face, (112, 112))
        return face_aligned

