import numpy as np
import hashlib
import json
import cv2
from typing import Dict
from app.core.config import MODE, LABEL_MAP_PATH

class EmotionModel:
    """Emotion classification model interface."""
    
    def __init__(self):
        with open(LABEL_MAP_PATH) as f:
            self.label_map = json.load(f)
        self.labels = list(self.label_map.values())
        self.session = None
        if MODE == "onnx":
            try:
                import onnxruntime as ort
                from pathlib import Path
                model_path = Path(__file__).parent.parent.parent / "app" / "models" / "emotion.onnx"
                if model_path.exists():
                    self.session = ort.InferenceSession(str(model_path))
                else:
                    print(f"Warning: Emotion ONNX model not found at {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load emotion ONNX model: {e}")
    
    def predict(self, img: np.ndarray) -> Dict[str, any]:
        """Returns {label: str, confidence: float}"""
        if MODE == "mock":
            # Deterministic emotion based on image hash
            img_hash = int(hashlib.md5(img.tobytes()).hexdigest(), 16)
            idx = img_hash % len(self.labels)
            label = self.labels[idx]
            # Mock confidence
            confidence = 0.6 + (img_hash % 40) / 100.0
            return {"label": label, "confidence": float(confidence)}
        
        elif MODE == "heur":
            # Simple heuristic: use image statistics
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            mean_intensity = np.mean(gray)
            # Simple rule: brighter = happy, darker = sad/neutral
            if mean_intensity > 180:
                label = "happy"
            elif mean_intensity < 80:
                label = "sad"
            else:
                label = "neutral"
            confidence = 0.65
            return {"label": label, "confidence": float(confidence)}
        
        elif MODE == "onnx":
            if self.session is None:
                # Fallback to heuristic
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                mean_intensity = np.mean(gray)
                if mean_intensity > 180:
                    label = "happy"
                elif mean_intensity < 80:
                    label = "sad"
                else:
                    label = "neutral"
                return {"label": label, "confidence": 0.65}
            
            # Preprocess for ONNX - check model input shape
            input_shape = self.session.get_inputs()[0].shape
            input_name_info = self.session.get_inputs()[0]
            
            # Get expected dimensions
            if len(input_shape) == 4:
                # Handle dynamic/None dimensions
                batch_dim = input_shape[0] if isinstance(input_shape[0], int) else 1
                channels = input_shape[1] if isinstance(input_shape[1], int) else None
                height = input_shape[2] if isinstance(input_shape[2], int) else None
                width = input_shape[3] if isinstance(input_shape[3], int) else None
                
                # Default to 112x112 if not specified
                if height is None or width is None:
                    height, width = 112, 112
                    
                # Resize to expected size
                img_resized = cv2.resize(img, (width, height))
                
                # Handle channels
                if channels == 3:
                    # Model expects RGB (3 channels)
                    if len(img_resized.shape) == 2:
                        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
                    # Already BGR from face_aligned, convert to RGB
                    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
                    img_normalized = img_rgb.astype(np.float32) / 255.0
                    # Convert HWC to CHW and add batch
                    img_chw = np.transpose(img_normalized, (2, 0, 1))
                    img_batch = np.expand_dims(img_chw, axis=0)  # (1, 3, H, W)
                else:
                    # Model expects grayscale (1 channel)
                    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if len(img_resized.shape) == 3 else img_resized
                    img_normalized = img_gray.astype(np.float32) / 255.0
                    img_batch = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=0)  # (1, 1, H, W)
            else:
                # Fallback to default preprocessing
                img_resized = cv2.resize(img, (64, 64))
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if len(img_resized.shape) == 3 else img_resized
                img_normalized = img_gray.astype(np.float32) / 255.0
                img_batch = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=0)
            
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: img_batch})
            logits = output[0].flatten()
            
            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            idx = int(np.argmax(probs))
            label = self.labels[idx] if idx < len(self.labels) else "neutral"
            confidence = float(probs[idx])
            
            return {"label": label, "confidence": confidence}
        
        return {"label": "neutral", "confidence": 0.0}

