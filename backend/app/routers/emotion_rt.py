from fastapi import APIRouter, File, UploadFile, HTTPException
import numpy as np
from PIL import Image
import io
import cv2

from app.pipelines.emotion import EmotionModel
from app.core.config import LABEL_MAP_PATH
import json

router = APIRouter()
emotion_model = EmotionModel()

@router.post("/api/emotion")
async def analyze_emotion(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = np.array(Image.open(io.BytesIO(contents)))
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif len(img.shape) != 3 or img.shape[2] != 3:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        result = emotion_model.predict(img)
        if not isinstance(result, dict) or "label" not in result:
            raise HTTPException(status_code=500, detail="Emotion inference failed")
        return {
            "label": result.get("label", "neutral"),
            "confidence": float(result.get("confidence", 0.0)),
            "probs": result.get("probs", {}),
            "age": result.get("age", 0),
            "gender": result.get("gender", "Unknown"),
            "gender_confidence": float(result.get("gender_confidence", 0.0)),
            "race": result.get("race", "Unknown"),
            "race_confidence": float(result.get("race_confidence", 0.0)),
        }
    except HTTPException:
        raise
    except Exception as e:
        # Return a neutral fallback with full probs keys for UI rendering
        try:
            with open(LABEL_MAP_PATH) as f:
                label_map = json.load(f)
            labels = list(label_map.values())
        except Exception:
            labels = ["angry","disgust","fear","happy","sad","surprise","neutral"]
        probs = {k: 0.0 for k in labels}
        return {"label": "neutral", "confidence": 0.0, "probs": probs}


