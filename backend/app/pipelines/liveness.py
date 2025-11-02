import numpy as np
import hashlib
import cv2
from typing import Dict
from app.core.config import MODE, THRESHOLDS_PATH
import yaml

class LivenessModel:
    """Liveness detection model interface."""
    
    def __init__(self):
        with open(THRESHOLDS_PATH) as f:
            self.config = yaml.safe_load(f)
        self.threshold = self.config.get("anti_spoof", {}).get("threshold", 0.5)
        self.session = None
        if MODE == "onnx":
            try:
                import onnxruntime as ort
                from pathlib import Path
                model_path = Path(__file__).parent.parent.parent / "app" / "models" / "liveness.onnx"
                if model_path.exists():
                    self.session = ort.InferenceSession(str(model_path))
                else:
                    print(f"Warning: Liveness ONNX model not found at {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load liveness ONNX model: {e}")
    
    def predict(self, img: np.ndarray) -> Dict[str, any]:
        """Returns {score: float, passed: bool}"""
        if MODE == "mock":
            # Deterministic based on image hash
            img_hash = int(hashlib.md5(img.tobytes()).hexdigest(), 16)
            score = (img_hash % 100) / 100.0
            return {"score": score, "passed": score > self.threshold}
        
        elif MODE == "heur":
            # Improved variance-based liveness (real faces have texture/variation)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            
            # Multiple features for better detection
            variance = np.var(gray)
            
            # Edge density (real faces have more edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Laplacian variance (measures local contrast/texture)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize and combine features
            variance_score = min(variance / 800.0, 1.0)  # Adjusted normalization
            edge_score = min(edge_density * 2.0, 1.0)  # Edge density contribution
            laplacian_score = min(laplacian_var / 500.0, 1.0)  # Texture contribution
            
            # Weighted combination (variance is most important)
            score = (0.5 * variance_score + 0.3 * edge_score + 0.2 * laplacian_score)
            score = float(max(0.0, min(1.0, score)))  # Ensure [0, 1]
            
            import sys
            print(f"[DEBUG] Liveness heuristic - variance: {variance:.2f} ({variance_score:.3f}), "
                  f"edges: {edge_density:.3f} ({edge_score:.3f}), "
                  f"laplacian: {laplacian_var:.2f} ({laplacian_score:.3f}), "
                  f"final: {score:.3f}, threshold: {self.threshold:.3f}", file=sys.stderr)
            
            return {"score": score, "passed": score > self.threshold}
        
        elif MODE == "onnx":
            if self.session is None:
                # Fallback to improved heuristic if model not loaded
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                
                # Use same improved heuristic as "heur" mode
                variance = np.var(gray)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                
                variance_score = min(variance / 800.0, 1.0)
                edge_score = min(edge_density * 2.0, 1.0)
                laplacian_score = min(laplacian_var / 500.0, 1.0)
                
                score = (0.5 * variance_score + 0.3 * edge_score + 0.2 * laplacian_score)
                score = float(max(0.0, min(1.0, score)))
                
                import sys
                print(f"[DEBUG] Liveness heuristic (ONNX fallback) - score: {score:.3f}, threshold: {self.threshold:.3f}", file=sys.stderr)
                
                return {"score": score, "passed": score > self.threshold}
            
            # Preprocess for ONNX
            img_resized = cv2.resize(img, (112, 112))
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_chw = np.transpose(img_normalized, (2, 0, 1))
            img_batch = np.expand_dims(img_chw, axis=0)
            
            input_name = self.session.get_inputs()[0].name
            output = self.session.run(None, {input_name: img_batch})
            score = float(output[0].flatten()[0])
            
            # Ensure score is in [0, 1]
            score = max(0.0, min(1.0, score))
            
            return {"score": score, "passed": score > self.threshold}
        
        return {"score": 0.0, "passed": False}

