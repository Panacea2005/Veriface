import numpy as np
import hashlib
import cv2
from typing import Literal
from app.core.config import MODE

class EmbedModel:
    """Face embedding model interface (512-D vectors)."""
    
    def __init__(self, model_type: Literal["A", "B"] = "A"):
        self.model_type = model_type
        self.session = None
        if MODE == "onnx":
            try:
                import onnxruntime as ort
                model_path = f"app/models/embedding_{model_type}.onnx"
                import os
                from pathlib import Path
                full_path = Path(__file__).parent.parent.parent / model_path
                if full_path.exists():
                    self.session = ort.InferenceSession(str(full_path))
                else:
                    print(f"Warning: ONNX model not found at {full_path}, using mock mode")
            except Exception as e:
                print(f"Warning: Failed to load ONNX model: {e}, using mock mode")
    
    def extract(self, img: np.ndarray) -> np.ndarray:
        """Extract 512-D embedding vector."""
        if MODE == "mock":
            # Deterministic hash-based embedding
            img_bytes = img.tobytes()
            seed = int(hashlib.sha256((img_bytes + self.model_type.encode())).hexdigest()[:16], 16)
            np.random.seed(seed)
            vec = np.random.randn(512).astype(np.float32)
            # Normalize to unit vector
            vec = vec / np.linalg.norm(vec)
            return vec
        
        elif MODE == "heur":
            # Simple features: histogram + moments
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            moments = cv2.moments(gray)
            features = np.concatenate([
                hist / hist.sum(),
                [moments["m00"], moments["m10"], moments["m01"]]
            ])
            # Pad/truncate to 512-D
            if len(features) < 512:
                features = np.pad(features, (0, 512 - len(features)))
            else:
                features = features[:512]
            features = features / (np.linalg.norm(features) + 1e-8)
            return features.astype(np.float32)
        
        elif MODE == "onnx":
            if self.session is None:
                # Fallback to mock if model not loaded
                img_bytes = img.tobytes()
                seed = int(hashlib.sha256((img_bytes + self.model_type.encode())).hexdigest()[:16], 16)
                np.random.seed(seed)
                vec = np.random.randn(512).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                return vec
            
            # Preprocess image for ONNX (normalize to [0,1], resize to model input size)
            img_resized = cv2.resize(img, (112, 112))
            img_normalized = img_resized.astype(np.float32) / 255.0
            # Convert HWC to CHW and add batch dimension
            img_chw = np.transpose(img_normalized, (2, 0, 1))
            img_batch = np.expand_dims(img_chw, axis=0)
            
            # Get input name from model
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: img_batch})
            embedding = output[0].flatten()
            
            # Normalize to unit vector
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            # If output is not 512-D, pad or truncate
            if len(embedding) < 512:
                embedding = np.pad(embedding, (0, 512 - len(embedding)))
            elif len(embedding) > 512:
                embedding = embedding[:512]
            
            return embedding.astype(np.float32)
        
        return np.zeros(512, dtype=np.float32)

