import numpy as np
import hashlib
import json
import cv2
from typing import Dict
from app.core.config import LABEL_MAP_PATH
import os
from pathlib import Path

class EmotionModel:
    """Emotion classification via DeepFace (real-only)."""
    
    def __init__(self):
        with open(LABEL_MAP_PATH) as f:
            self.label_map = json.load(f)
        self.labels = list(self.label_map.values())
        # Lazy init: avoid importing DeepFace at app startup
        self.use_deepface = False
    
    def predict(self, img: np.ndarray) -> Dict[str, any]:
        """Returns {label: str, confidence: float, probs: Dict[str, float], age: int, gender: str, race: str}"""
        # Ensure DeepFace imports even if tensorflow lacks __version__ attribute
        try:
            import tensorflow as tf  # type: ignore
            if not hasattr(tf, "__version__"):
                setattr(tf, "__version__", "2.17.0")
        except Exception:
            # If TF import fails, DeepFace may still load some backends; continue
            pass

        from deepface import DeepFace
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        result = DeepFace.analyze(
            img_path=rgb,
            actions=["emotion", "age", "gender", "race"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True  # Disable progress bars
        )
        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        probs_raw = result.get("emotion", {}) or {}
        # Convert to floats and normalize so sum=1
        values = [float(v) for v in probs_raw.values()]
        # Handle DeepFace returning percentages or probabilities
        if max(values or [0.0]) > 1.0:
            values = [v / 100.0 for v in values]
        total = sum(values) or 1.0
        norm_probs = {k: float(v) / total for (k, v) in zip(probs_raw.keys(), values)}

        # Dominant emotion from normalized probabilities
        label = str(result.get("dominant_emotion", max(norm_probs, key=norm_probs.get) if norm_probs else "neutral"))
        confidence = float(norm_probs.get(label, 0.0))
        
        # Extract age, gender, race
        age = int(result.get("age", 0))
        gender_data = result.get("gender", {})
        gender = str(result.get("dominant_gender", max(gender_data, key=gender_data.get) if gender_data else "Unknown"))
        race_data = result.get("race", {})
        race = str(result.get("dominant_race", max(race_data, key=race_data.get) if race_data else "Unknown"))
        
        return {
            "label": label, 
            "confidence": confidence, 
            "probs": norm_probs,
            "age": age,
            "gender": gender,
            "gender_confidence": float(gender_data.get(gender, 0.0)) if gender_data else 0.0,
            "race": race,
            "race_confidence": float(race_data.get(race, 0.0)) if race_data else 0.0
        }

