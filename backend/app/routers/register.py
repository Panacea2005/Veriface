from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional, Literal
import numpy as np
from PIL import Image
import cv2
import base64
import io
import sys

from app.pipelines.detector import FaceDetector
from app.pipelines.embedding import EmbedModel, extract_dual_embeddings
from app.pipelines.liveness import LivenessModel
from app.pipelines.registry import FaceRegistry
from app.core.config import MODEL_TYPE

router = APIRouter()
detector = FaceDetector()
registry = FaceRegistry()
liveness_model = LivenessModel()

@router.post("/api/register")
async def register(
    name: str = Form(...),
    user_id: Optional[str] = Form(None),  # Optional: if provided, add to existing user
    image: Optional[UploadFile] = File(None),
    image_b64: Optional[str] = Form(None)
):
    """
    Register a new user face.
    
    Args:
        name: User's full name (required)
        user_id: Optional user_id. If provided and exists, adds embedding to that user.
                 If not provided, auto-generates new ID (SWS00001, SWS00002, etc.)
        image: Face image (multipart file)
        image_b64: Face image (base64 string)
    
    Returns:
        user_id: Generated or existing user ID
        name: User's name
        embedding_count: Total embeddings for this user
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
        
        # Extract embeddings using Model A
        try:
            embeddings_by_model, embed_model = extract_dual_embeddings(face_aligned, preferred_model=MODEL_TYPE)
            if not embeddings_by_model or "torch" not in embeddings_by_model:
                raise ValueError("Embedding extraction returned no vectors")
            torch_vector = embeddings_by_model.get("torch")
            if torch_vector is not None:
                vec_norm = float(np.linalg.norm(torch_vector))
                vec_std = float(np.std(torch_vector))
                vec_sample = torch_vector[:5].tolist() if len(torch_vector) >= 5 else torch_vector.tolist()
                print(f"[DEBUG] Register: Embedding norm={vec_norm:.6f}, std={vec_std:.6f}, sample={vec_sample}", file=sys.stderr)
                if vec_norm < 1e-6 or vec_std < 1e-6:
                    raise ValueError(f"Embedding invalid (norm={vec_norm:.6f}, std={vec_std:.6f})")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")
        
        # Save to registry (replace existing embeddings to ensure only 1 front-facing embedding)
        try:
            torch_embedding = embeddings_by_model.get("torch")
            deepface_embedding = embeddings_by_model.get("deepface")
            
            # Remove existing embeddings if user_id provided (to ensure only 1 embedding)
            if user_id:
                existing_vectors = registry.get_user_vectors(user_id, model="torch")
                if existing_vectors:
                    print(f"[DEBUG] Register: Removing existing embeddings for user '{user_id}' to ensure single front-facing embedding", file=sys.stderr)
                    registry.remove_user(user_id)
            
            final_user_id = user_id
            # Save embedding with replace=True to ensure only 1 embedding per user
            if torch_embedding is not None:
                final_user_id = registry.add_user(name=name, embedding=torch_embedding, user_id=final_user_id, model="torch", replace=True)
                print(f"[DEBUG] Register: Saved embedding for user '{final_user_id}'", file=sys.stderr)
            if deepface_embedding is not None:
                final_user_id = registry.add_user(name=name, embedding=deepface_embedding, user_id=final_user_id, model="deepface", replace=True)
            if final_user_id is None:
                raise ValueError("Failed to persist embeddings for this user")
            
            counts = registry.get_user_embedding_counts(final_user_id)
            total_embeddings = counts["torch"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save to registry: {str(e)}")
        
        primary_vector = torch_embedding if torch_embedding is not None else deepface_embedding
        return {
            "status": "success",
            "user_id": final_user_id,
            "name": name,
            "embedding_count": total_embeddings,
            "model": MODEL_TYPE,
            "embedding_shape": list(primary_vector.shape) if primary_vector is not None else []
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

@router.post("/api/register/batch")
async def register_batch(
    name: str = Form(...),
    user_id: Optional[str] = Form(None),  # Optional: if provided, adds to existing user
    images: list[UploadFile] = File(...)
):
    """
    Register a user with multiple images but only save the front-facing image embedding.
    Only the first valid face image (assumed to be front-facing) will be saved.
    This simplifies verification by matching against a single front-facing embedding.
    
    Args:
        name: User's full name (required)
        user_id: Optional user_id. If provided and exists, replaces embeddings for that user.
                 If not provided, auto-generates new ID (SWS00001, SWS00002, etc.)
        images: List of face images (only first valid front-facing image will be saved)
    """
    if not images or len(images) == 0:
        raise HTTPException(status_code=400, detail="At least one image is required")
    
    try:
        # Prepare primary embed model once (reuse for all images)
        # Model A/B always runs (PyTorch model)
        embed_model = EmbedModel(model_type=MODEL_TYPE)
        
        # Process all images: first one is saved, others are for visualization
        front_embeddings = None  # Embedding to save (first valid image)
        processing_results = []  # All processing results for visualization
        
        for idx, image_file in enumerate(images):
            try:
                # Load image
                contents = await image_file.read()
                img = np.array(Image.open(io.BytesIO(contents)))
                
                # Validate image dimensions
                if len(img.shape) < 2:
                    print(f"[WARN] Register Batch: Image {idx+1} has invalid format, skipping", file=sys.stderr)
                    processing_results.append({
                        "image_index": idx + 1,
                        "status": "skipped",
                        "reason": "invalid_format"
                    })
                    continue
                
                # Convert RGB to BGR for OpenCV
                if len(img.shape) == 3:
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    else:
                        print(f"[WARN] Register Batch: Image {idx+1} has unsupported channels, skipping", file=sys.stderr)
                        processing_results.append({
                            "image_index": idx + 1,
                            "status": "skipped",
                            "reason": "unsupported_channels"
                        })
                        continue
                elif len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                
                # Detect and align face
                bbox = detector.detect(img)
                if bbox is None:
                    print(f"[WARN] Register Batch: No face detected in image {idx+1}, skipping", file=sys.stderr)
                    processing_results.append({
                        "image_index": idx + 1,
                        "status": "skipped",
                        "reason": "no_face_detected"
                    })
                    continue
                
                # Use standard alignment (front-facing) for all images
                face_aligned = detector.align(img, bbox, preserve_angle=False)
                if face_aligned is None or face_aligned.size == 0:
                    print(f"[WARN] Register Batch: Failed to align face in image {idx+1}, skipping", file=sys.stderr)
                    processing_results.append({
                        "image_index": idx + 1,
                        "status": "skipped",
                        "reason": "alignment_failed"
                    })
                    continue
                
                # Extract embeddings (reuse model instance)
                embeddings_by_model, _ = extract_dual_embeddings(face_aligned, preferred_model=MODEL_TYPE, reuse_model=embed_model)
                if not embeddings_by_model:
                    print(f"[WARN] Register Batch: Failed to extract embeddings from image {idx+1}, skipping", file=sys.stderr)
                    processing_results.append({
                        "image_index": idx + 1,
                        "status": "skipped",
                        "reason": "embedding_extraction_failed"
                    })
                    continue
                
                # Calculate embedding stats for visualization
                torch_emb = embeddings_by_model.get("torch")
                result_info = {
                    "image_index": idx + 1,
                    "status": "processed",
                    "has_torch": torch_emb is not None
                }
                
                if torch_emb is not None:
                    result_info["torch_norm"] = float(np.linalg.norm(torch_emb))
                    result_info["torch_mean"] = float(np.mean(torch_emb))
                    result_info["torch_std"] = float(np.std(torch_emb))
                
                # Calculate similarity with first embedding if available (for visualization)
                if front_embeddings is not None and torch_emb is not None:
                    first_torch = front_embeddings.get("torch")
                    if first_torch is not None:
                        similarity = float(np.dot(torch_emb, first_torch) / (np.linalg.norm(torch_emb) * np.linalg.norm(first_torch) + 1e-8))
                        result_info["similarity_to_first"] = similarity
                
                processing_results.append(result_info)
                
                # Save first valid embedding (for registry)
                if front_embeddings is None:
                    front_embeddings = embeddings_by_model
                    print(f"[DEBUG] Register Batch: Found front-facing image at position {idx+1}, will save this embedding", file=sys.stderr)
                else:
                    print(f"[DEBUG] Register Batch: Processed image {idx+1} for visualization (embedding extracted but not saved)", file=sys.stderr)
                
            except Exception as e:
                print(f"[WARN] Register Batch: Error processing image {idx+1}: {str(e)}, skipping", file=sys.stderr)
                processing_results.append({
                    "image_index": idx + 1,
                    "status": "error",
                    "error": str(e)
                })
                continue
        
        if front_embeddings is None:
            raise HTTPException(status_code=400, detail="No valid front-facing face detected in any image")
        
        # Remove existing embeddings for this user to avoid duplicates
        try:
            if user_id:
                existing_vectors = registry.get_user_vectors(user_id)
                if existing_vectors:
                    print(f"[DEBUG] Register Batch: Removing existing embeddings for user '{user_id}'", file=sys.stderr)
                    registry.remove_user(user_id)
        except Exception as e:
            print(f"[WARN] Register Batch: Could not remove existing embeddings: {str(e)}", file=sys.stderr)
        
        # Save only the front-facing embedding (single embedding per model)
        try:
            final_user_id = None
            torch_embedding = front_embeddings.get("torch")
            deepface_embedding = front_embeddings.get("deepface")
            
            # Save embedding with replace=True to ensure only 1 embedding per user
            if torch_embedding is not None:
                final_user_id = registry.add_user(name=name, embedding=torch_embedding, user_id=user_id, model="torch", replace=True)
                print(f"[DEBUG] Register Batch: Saved embedding for user '{final_user_id}'", file=sys.stderr)
            
            if deepface_embedding is not None:
                final_user_id = registry.add_user(name=name, embedding=deepface_embedding, user_id=user_id or final_user_id, model="deepface", replace=True)
            
            if final_user_id is None:
                raise ValueError("Failed to save embeddings for this user")
            
            print(f"[DEBUG] Register Batch: Successfully saved embedding for user '{final_user_id}' (name: '{name}')", file=sys.stderr)
            
            counts = registry.get_user_embedding_counts(final_user_id)
            total_embeddings = counts["torch"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save to registry: {str(e)}")
        
        shape_vec = torch_embedding if torch_embedding is not None else deepface_embedding
        return {
            "status": "success",
            "user_id": final_user_id,
            "name": name,
            "embedding_count": total_embeddings,
            "model": MODEL_TYPE,
            "images_processed": len([r for r in processing_results if r.get("status") == "processed"]),  # All processed images
            "images_total": len(images),
            "embeddings_saved": total_embeddings,  # Only 1 embedding saved (first image)
            "embedding_shape": list(shape_vec.shape) if shape_vec is not None else [],
            "processing_results": processing_results  # All processing results for visualization
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = str(e)
        error_type = type(e).__name__
        tb = traceback.format_exc()
        
        print("=" * 80, file=sys.stderr)
        print(f"ERROR in register_batch endpoint: {error_type}", file=sys.stderr)
        print(f"Error message: {error_detail}", file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        print(tb, file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        
        raise HTTPException(status_code=500, detail=f"Batch registration failed: {error_type}: {error_detail}")

