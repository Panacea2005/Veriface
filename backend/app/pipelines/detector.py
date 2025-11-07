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
        """Crop and align face to 112x112 using DeepFace alignment if available."""
        x, y, w, h = bbox
        
        # Try to use DeepFace's alignment (5-point landmark alignment) for better accuracy
        if self.use_deepface:
            try:
                from deepface import DeepFace
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
                # Use DeepFace extract_faces with align=True to get properly aligned face
                faces = DeepFace.extract_faces(
                    img_path=rgb,
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=True
                )
                if faces and len(faces) > 0:
                    first = faces[0]
                    if isinstance(first, dict):
                        aligned_face = first.get("face", first.get("aligned", None))
                    else:
                        aligned_face = first
                    if aligned_face is not None:
                        # Ensure uint8 format
                        if aligned_face.dtype != np.uint8:
                            if np.max(aligned_face) <= 1.0:
                                aligned_face = (aligned_face * 255.0).astype(np.uint8)
                            else:
                                aligned_face = np.clip(aligned_face, 0, 255).astype(np.uint8)
                        # Resize to 112x112 if needed
                        if aligned_face.shape[:2] != (112, 112):
                            aligned_face = cv2.resize(aligned_face, (112, 112))
                        # Convert back to BGR for consistency
                        if len(aligned_face.shape) == 3 and aligned_face.shape[2] == 3:
                            aligned_face = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                        return aligned_face
            except Exception:
                pass  # Fallback to simple crop and resize
        
        # Fallback: Simple crop and resize
        face = image[y:y+h, x:x+w]
        face_aligned = cv2.resize(face, (112, 112))
        return face_aligned

