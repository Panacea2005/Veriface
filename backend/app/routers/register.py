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
            
            print(f"[DEBUG] Register: Detected face bbox: {bbox}", file=sys.stderr)
            face_aligned = detector.align(img, bbox)
            if face_aligned is None or face_aligned.size == 0:
                raise HTTPException(status_code=400, detail="Failed to align face")
            
            # Debug: Check face image stats to ensure different faces produce different images
            face_mean = np.mean(face_aligned)
            face_std = np.std(face_aligned)
            face_min = np.min(face_aligned)
            face_max = np.max(face_aligned)
            print(f"[DEBUG] Register: Aligned face stats: shape={face_aligned.shape}, mean={face_mean:.2f}, std={face_std:.2f}, min={face_min:.0f}, max={face_max:.0f}", file=sys.stderr)
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
            model_type_used = "PyTorch" if embed_model.model is not None else "DeepFace"
            print(f"[DEBUG] Register: Using {model_type_used} model for embedding extraction", file=sys.stderr)
            embedding = embed_model.extract(face_aligned)
            embedding_norm = np.linalg.norm(embedding) if embedding is not None else 0.0
            embedding_mean = np.mean(embedding) if embedding is not None else 0.0
            embedding_std = np.std(embedding) if embedding is not None else 0.0
            embedding_min = np.min(embedding) if embedding is not None else 0.0
            embedding_max = np.max(embedding) if embedding is not None else 0.0
            # Sample first 5 values for debugging
            embedding_sample = embedding[:5].tolist() if embedding is not None and len(embedding) >= 5 else []
            print(f"[DEBUG] Register: Embedding shape: {embedding.shape if embedding is not None else None}, norm: {embedding_norm:.6f}, mean: {embedding_mean:.6f}, std: {embedding_std:.6f}, min: {embedding_min:.6f}, max: {embedding_max:.6f}", file=sys.stderr)
            print(f"[DEBUG] Register: Embedding sample (first 5): {embedding_sample}", file=sys.stderr)
            
            # Check if embedding is all zeros or constant (indicates model failure)
            if embedding is not None:
                if embedding_norm < 1e-6:
                    raise ValueError("Embedding is all zeros - model may not be working correctly")
                if embedding_std < 1e-6:
                    raise ValueError("Embedding has zero variance - model may not be working correctly")
            
            if embedding is None or embedding.size == 0:
                raise ValueError("Failed to extract embedding")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")
        
        # Save to registry
        try:
            # Check if this embedding already exists for this user (avoid duplicates)
            existing_vectors = registry.get_user_vectors(user_id)
            if existing_vectors:
                # Compare with existing embeddings
                for idx, existing_vec in enumerate(existing_vectors):
                    existing_array = np.array(existing_vec, dtype=np.float32)
                    # Normalize for comparison
                    existing_norm = np.linalg.norm(existing_array)
                    if existing_norm > 0.1:
                        existing_array = existing_array / existing_norm
                    similarity = np.dot(embedding, existing_array)
                    distance = np.linalg.norm(embedding - existing_array)
                    print(f"[DEBUG] Register: Comparing with existing embedding {idx+1}: similarity={similarity:.6f}, distance={distance:.6f}", file=sys.stderr)
                    if distance < 1e-6 or similarity > 0.9999:
                        print(f"[WARNING] Register: New embedding is nearly identical to existing embedding {idx+1} (distance: {distance:.6f}, similarity: {similarity:.6f})", file=sys.stderr)
            
            registry.add_user(user_id, embedding)
            print(f"[DEBUG] Register: Successfully saved embedding for user '{user_id}'", file=sys.stderr)
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

