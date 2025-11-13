from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional, Literal
import numpy as np
from PIL import Image
import cv2
import base64
import io
import sys

from app.pipelines.detector import FaceDetector
from app.core.config import SIMILARITY_METRIC
from app.pipelines.embedding import EmbedModel
from app.pipelines.liveness import LivenessModel
from app.pipelines.emotion import EmotionModel
from app.pipelines.similarity import SimilarityMatcher
from app.pipelines.registry import FaceRegistry

router = APIRouter()
detector = FaceDetector()
registry = FaceRegistry()
liveness_model = LivenessModel()
emotion_model = EmotionModel()
similarity_matcher = SimilarityMatcher()

@router.post("/api/verify")
async def verify(
    image: Optional[UploadFile] = File(None),
    image_b64: Optional[str] = Form(None)
):
    """
    Verify a face against the registry.
    Returns liveness, match result, emotion, etc.
    """
    try:
        metric: Literal["cosine", "euclidean"] = SIMILARITY_METRIC  # env-driven
        print(f"[DEBUG] Verify request - metric: {metric}", file=sys.stderr)
        
        # Load image
        if image:
            print(f"[DEBUG] Loading image from file upload...", file=sys.stderr)
            contents = await image.read()
            print(f"[DEBUG] Image size: {len(contents)} bytes", file=sys.stderr)
            img = np.array(Image.open(io.BytesIO(contents)))
            print(f"[DEBUG] Image shape: {img.shape}", file=sys.stderr)
        elif image_b64:
            print(f"[DEBUG] Loading image from base64...", file=sys.stderr)
            img_data = base64.b64decode(image_b64.split(',')[-1])
            img = np.array(Image.open(io.BytesIO(img_data)))
            print(f"[DEBUG] Image shape: {img.shape}", file=sys.stderr)
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
            print(f"[DEBUG] Detecting face...", file=sys.stderr)
            bbox = detector.detect(img)
            print(f"[DEBUG] BBox result: {bbox}", file=sys.stderr)
            if bbox is None:
                raise HTTPException(status_code=400, detail="No face detected in image")
            
            print(f"[DEBUG] Aligning face...", file=sys.stderr)
            face_aligned = detector.align(img, bbox)
            print(f"[DEBUG] Face aligned shape: {face_aligned.shape if face_aligned is not None else None}", file=sys.stderr)
            if face_aligned is None or face_aligned.size == 0:
                raise HTTPException(status_code=400, detail="Failed to align face")
        except HTTPException:
            raise
        except Exception as e:
            print(f"[DEBUG] Face detection exception: {type(e).__name__}: {e}", file=sys.stderr)
            raise HTTPException(status_code=400, detail=f"Face detection failed: {str(e)}")
        
        # Liveness check using DeepFace's MiniFASNet (Silent-Face-Anti-Spoofing)
        # MiniFASNet analyzes texture, depth, moiré patterns, and color to detect spoofs
        # Works directly on the full image to get proper context
        try:
            print(f"[DEBUG] Running liveness check with DeepFace MiniFASNet...", file=sys.stderr)
            # Pass the full frame (not just cropped face) for better context
            liveness = liveness_model.predict(img)
            print(f"[DEBUG] Liveness result: {liveness}", file=sys.stderr)
            if not isinstance(liveness, dict) or "passed" not in liveness:
                raise ValueError("Invalid liveness response")
        except Exception as e:
            print(f"Warning: Liveness check failed: {e}, setting to 0% (blocked)", file=sys.stderr)
            liveness = {"score": 0.0, "passed": False}
        
        # Block verification if liveness check fails (spoof detected)
        if not liveness.get("passed", False):
            x, y, w, h = bbox
            liveness_score = liveness.get('score', 0.0)
            raise HTTPException(
                status_code=400,
                detail=f"Anti-spoof detection failed: Spoof detected (liveness score: {liveness_score:.2f}). Verification blocked for security. Please use a real face."
            )
        
        # Prepare initial result
        result = {
            "liveness": liveness,
            "matched_id": None,
            "score": None,
            "metric": metric,
            "threshold": None,
            "emotion_label": None,
            "emotion_confidence": None,
            "all_scores": []
        }
        
        # Emotion detection (do this early so it's available for attendance logging)
        try:
            print(f"[DEBUG] Detecting emotion...", file=sys.stderr)
            emotion = emotion_model.predict(face_aligned)
            print(f"[DEBUG] Emotion result: {emotion}", file=sys.stderr)
            if not isinstance(emotion, dict) or "label" not in emotion:
                raise ValueError("Invalid emotion response")
            result["emotion_label"] = emotion.get("label", "neutral")
            result["emotion_confidence"] = emotion.get("confidence", 0.0)
            if isinstance(emotion.get("probs"), dict):
                result["emotion_probs"] = emotion["probs"]

            # Append to emotion logs (JSONL) for history
            try:
                from datetime import datetime
                from app.core.config import STORE_DIR
                import json as _json
                log_entry = {
                    "ts": datetime.utcnow().isoformat() + "Z",
                    "label": result["emotion_label"],
                    "confidence": result["emotion_confidence"],
                    "liveness": result["liveness"],
                }
                with open(STORE_DIR / "emotion_logs.jsonl", "a", encoding="utf-8") as f:
                    f.write(_json.dumps(log_entry) + "\n")
            except Exception as log_e:
                print(f"[WARN] Failed to write emotion log: {log_e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Emotion detection failed: {e}, defaulting to neutral", file=sys.stderr)
            result["emotion_label"] = "neutral"
            result["emotion_confidence"] = 0.0
        
        # Extract embedding
        try:
            print(f"[DEBUG] Extracting embedding...", file=sys.stderr)
            embed_model = EmbedModel()
            model_type_used = "PyTorch" if embed_model.model is not None else "DeepFace"
            print(f"[DEBUG] Using {model_type_used} model for embedding extraction", file=sys.stderr)
            embedding = embed_model.extract(face_aligned)
            embedding_norm = np.linalg.norm(embedding) if embedding is not None else 0.0
            embedding_mean = np.mean(embedding) if embedding is not None else 0.0
            embedding_std = np.std(embedding) if embedding is not None else 0.0
            embedding_sample = embedding[:5].tolist() if embedding is not None and len(embedding) >= 5 else []
            print(f"[DEBUG] Embedding shape: {embedding.shape if embedding is not None else None}, norm: {embedding_norm:.6f}, mean: {embedding_mean:.6f}, std: {embedding_std:.6f}", file=sys.stderr)
            print(f"[DEBUG] Embedding sample (first 5): {embedding_sample}", file=sys.stderr)
            
            # Check if embedding is all zeros or constant (indicates model failure)
            if embedding is not None:
                if embedding_norm < 1e-6:
                    raise ValueError("Embedding is all zeros - model may not be working correctly")
                if embedding_std < 1e-6:
                    raise ValueError("Embedding has zero variance - model may not be working correctly")
            
            if embedding is None or embedding.size == 0:
                raise ValueError("Failed to extract embedding")
        except Exception as e:
            print(f"[DEBUG] Embedding extraction exception: {type(e).__name__}: {e}", file=sys.stderr)
            raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")
        
        # Similarity matching
        try:
            print(f"[DEBUG] Matching against registry...", file=sys.stderr)
            registry_data = registry.get_all()
            print(f"[DEBUG] Registry has {len(registry_data)} users", file=sys.stderr)
            
            # Get all scores for display
            all_scores = similarity_matcher.get_all_scores(embedding, registry_data, metric=metric)
            def euclidean_to_percent(d: float) -> float:
                # With unit-normalized embeddings, distance in [0, 2]
                d_clamped = max(0.0, min(float(d), 2.0))
                return (1.0 - d_clamped / 2.0) * 100.0

            result["all_scores"] = [
                {
                    "user_id": user_id,
                    "score": score,
                    "percentage": (score * 100.0) if metric == "cosine" else euclidean_to_percent(score),
                    "embeddings_count": count
                }
                for user_id, score, count in all_scores
            ]
            
            # Get best match
            match = similarity_matcher.match(embedding, registry_data, metric=metric)
            print(f"[DEBUG] Match result: {match}", file=sys.stderr)
            
            # Always set threshold
            if metric == "cosine":
                result["threshold"] = similarity_matcher.cosine_threshold
            else:
                result["threshold"] = similarity_matcher.euclidean_threshold
            
            # Always set best score (even if below threshold)
            if all_scores:
                best_user_id, best_score, _ = all_scores[0]
                result["score"] = float(best_score)
                
                # Only set matched_id if match passed threshold
                if match:
                    matched_id, score = match
                    result["matched_id"] = matched_id
                    
                    # Get user name from registry
                    matched_name = registry.get_user_name(matched_id)
                    result["matched_name"] = matched_name
                    
                    print(f"[DEBUG] Match PASSED: {matched_id} ({matched_name}) with score {score:.4f}", file=sys.stderr)
                    
                    # Log attendance for successful match
                    try:
                        from app.routers.attendance import _log_attendance, _determine_check_type
                        from datetime import datetime
                        check_type = _determine_check_type(matched_id)
                        _log_attendance({
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "user_id": matched_id,
                            "type": check_type,
                            "match_score": float(score),
                            "liveness_score": float(liveness.get("score", 0.0)),
                            "emotion_label": result.get("emotion_label"),
                            "emotion_confidence": result.get("emotion_confidence")
                        })
                        result["check_type"] = check_type
                        print(f"[DEBUG] Attendance logged for user: {matched_id}, type: {check_type}", file=sys.stderr)
                    except Exception as log_e:
                        print(f"[WARN] Failed to log attendance: {log_e}", file=sys.stderr)
                else:
                    result["matched_id"] = None
                    print(f"[DEBUG] No match (best score {best_score:.4f} below threshold)", file=sys.stderr)
            else:
                result["score"] = None
                result["matched_id"] = None
                print(f"[DEBUG] No scores calculated (empty registry or error)", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Matching failed: {e}", file=sys.stderr)
            result["all_scores"] = []
            result["score"] = None
            result["matched_id"] = None
        
        print(f"[DEBUG] Returning result: {result}", file=sys.stderr)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = str(e)
        error_type = type(e).__name__
        tb = traceback.format_exc()
        
        # Print to stderr (console)
        print("=" * 80, file=sys.stderr)
        print(f"ERROR in verify endpoint: {error_type}", file=sys.stderr)
        print(f"Error message: {error_detail}", file=sys.stderr)
        print("-" * 80, file=sys.stderr)
        print(tb, file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        
        raise HTTPException(status_code=500, detail=f"Verification failed: {error_type}: {error_detail}")

