import numpy as np
import cv2
from typing import Literal

class EmbedModel:
    """Face embedding model interface (512-D vectors)."""
    
    def __init__(self, model_type: Literal["A", "B"] = "A"):
        self.model_type = model_type
        self.session = None
        # Always try ONNX for the chosen model; fallback to DeepFace if unavailable
        try:
            import onnxruntime as ort
            model_path = f"app/models/embedding_{model_type}.onnx"
            from pathlib import Path
            full_path = Path(__file__).parent.parent.parent / model_path
            if full_path.exists():
                self.session = ort.InferenceSession(str(full_path))
            else:
                print(f"Warning: ONNX model not found at {full_path}, will use DeepFace fallback")
        except Exception as e:
            print(f"Warning: Failed to initialize ONNX Runtime or session: {e}. Using DeepFace fallback")
    
    def extract(self, img: np.ndarray) -> np.ndarray:
        """Extract 512-D embedding vector (ONNX if available, else DeepFace ArcFace)."""
        # Prefer ONNX session when available for selected A/B model
        if self.session is not None:
            # Preprocess image for ONNX (RGB, [0,1], CHW)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
            img_resized = cv2.resize(img_rgb, (112, 112))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_chw = np.transpose(img_normalized, (2, 0, 1))
            img_batch = np.expand_dims(img_chw, axis=0)
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: img_batch})
            embedding = output[0].flatten()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            if len(embedding) < 512:
                embedding = np.pad(embedding, (0, 512 - len(embedding)))
            elif len(embedding) > 512:
                embedding = embedding[:512]
            return embedding.astype(np.float32)

        # DeepFace ArcFace fallback
        try:
            from deepface import DeepFace
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
            rep = DeepFace.represent(img_path=rgb, model_name="ArcFace", enforce_detection=False)
            if rep and len(rep) > 0:
                embedding = np.array(rep[0]['embedding'], dtype=np.float32)
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                if len(embedding) < 512:
                    embedding = np.pad(embedding, (0, 512 - len(embedding)))
                elif len(embedding) > 512:
                    embedding = embedding[:512]
                return embedding.astype(np.float32)
        except Exception:
            pass
        
        # Last resort
        return np.zeros(512, dtype=np.float32)

