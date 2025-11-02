from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional, Literal
import numpy as np
from PIL import Image
import cv2
import base64
import io
import sys

from app.pipelines.detector import FaceDetector
from app.pipelines.embedding import EmbedModel
from app.pipelines.liveness import LivenessModel
from app.pipelines.registry import FaceRegistry

router = APIRouter()
detector = FaceDetector()
registry = FaceRegistry()
liveness_model = LivenessModel()

@router.post("/api/register")
async def register(
    user_id: str = Form(...),
    model: Literal["A", "B"] = Form("A"),
    image: Optional[UploadFile] = File(None),
    image_b64: Optional[str] = Form(None)
):
    """
    Register a new user face.
    Accepts either multipart file or base64 image.
    """
    try:
        # Load image
        if image:
            contents = await image.read()
            img = np.array(Image.open(io.BytesIO(contents)))
        elif image_b64:
            img_data = base64.b64decode(image_b64.split(',')[-1])
            img = np.array(Image.open(io.BytesIO(img_data)))
        else:
            raise HTTPException(status_code=400, detail="Either 'image' or 'image_b64' required")
        
        # Validate image dimensions
        if len(img.shape) < 2:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Convert RGB to BGR for OpenCV (only if 3 channels)
        if len(img.shape) == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[2] == 4:
                # RGBA - convert to RGB first
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported image channels: {img.shape[2]}")
        elif len(img.shape) == 2:
            # Grayscale - convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Detect and align face
        try:
            bbox = detector.detect(img)
            if bbox is None:
                raise HTTPException(status_code=400, detail="No face detected in image")
            
            face_aligned = detector.align(img, bbox)
            if face_aligned is None or face_aligned.size == 0:
                raise HTTPException(status_code=400, detail="Failed to align face")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
        
        # Optional: Liveness check (can be skipped in register for simplicity)
        # liveness = liveness_model.predict(face_aligned)
        # if not liveness["passed"]:
        #     raise HTTPException(status_code=400, detail="Liveness check failed")
        
        # Extract embedding
        try:
            embed_model = EmbedModel(model_type=model)
            embedding = embed_model.extract(face_aligned)
            if embedding is None or embedding.size == 0:
                raise ValueError("Failed to extract embedding")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")
        
        # Save to registry
        try:
            registry.add_user(user_id, embedding)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save to registry: {str(e)}")
        
        return {
            "status": "success",
            "user_id": user_id,
            "model": model,
            "embedding_shape": list(embedding.shape)
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = str(e)
        error_type = type(e).__name__
        tb = traceback.format_exc()
        
        # Print to stderr (console)
        print("=" * 80, file=sys.stderr)
        print(f"ERROR in register endpoint: {error_type}", file=sys.stderr)
        print(f"Error message: {error_detail}", file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        print(tb, file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        
        raise HTTPException(status_code=500, detail=f"Registration failed: {error_type}: {error_detail}")

