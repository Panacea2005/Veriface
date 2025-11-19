from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional, Literal, Dict
import numpy as np
from PIL import Image
import cv2
import base64
import io
import sys

from app.pipelines.detector import FaceDetector
from app.core.config import SIMILARITY_METRIC, MODEL_TYPE, MODEL_TORCH_WEIGHT, MODEL_DEEPFACE_WEIGHT, ENABLE_DEEPFACE
from app.pipelines.embedding import extract_dual_embeddings
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
        
        # Liveness check (anti-spoofing)
        # Works directly on the full image to get proper context
        try:
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
        x, y, w, h = bbox
        result = {
            "liveness": liveness,
            "matched_id": None,
            "score": None,
            "metric": metric,
            "threshold": None,
            "emotion_label": None,
            "emotion_confidence": None,
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
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
                if vec_norm < 1e-6 or vec_std < 1e-6:
                    raise ValueError(f"Embedding invalid (norm={vec_norm:.6f}, std={vec_std:.6f})")
        except Exception as e:
            print(f"[DEBUG] Embedding extraction exception: {type(e).__name__}: {e}", file=sys.stderr)
            raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")
        
        # Similarity matching
        try:
            registry_data = registry.get_all()
            
            registries_by_model: Dict[str, Dict[str, list]] = {"torch": {}, "deepface": {}}
            counts_map: Dict[str, int] = {}
            for uid, user_data in registry_data.items():
                torch_vecs = user_data.get("embeddings", []) if isinstance(user_data, dict) else user_data
                deep_vecs = user_data.get("embeddings_deepface", []) if isinstance(user_data, dict) else []
                if torch_vecs:
                    registries_by_model["torch"][uid] = torch_vecs
                if deep_vecs:
                    registries_by_model["deepface"][uid] = deep_vecs
                counts_map[uid] = len(torch_vecs or [])

            model_results = {}
            score_maps: Dict[str, Dict[str, float]] = {}
            
            # Process embeddings and calculate similarity scores
            for model_name, embedding_vec in embeddings_by_model.items():
                registry_subset = registries_by_model.get(model_name, {})
                if embedding_vec is None or not registry_subset:
                    continue
                scores = similarity_matcher.get_all_scores(embedding_vec, registry_subset, metric=metric)
                model_results[model_name] = {
                    "all_scores": scores,
                    "match": similarity_matcher.match(embedding_vec, registry_subset, metric=metric)
                }
                score_maps[model_name] = {uid: score for uid, score, _ in scores}
            

            # Calculate final similarity scores
            weight_map = {"torch": MODEL_TORCH_WEIGHT, "deepface": MODEL_DEEPFACE_WEIGHT}
            if ENABLE_DEEPFACE and "torch" in score_maps and "deepface" in score_maps:
                combined_scores: Dict[str, float] = {}
                combined_weights: Dict[str, float] = {}
                for model_name, scores in score_maps.items():
                    weight = weight_map.get(model_name, 0.0)
                    if weight <= 0:
                        continue
                    for uid, score in scores.items():
                        combined_scores[uid] = combined_scores.get(uid, 0.0) + weight * score
                        combined_weights[uid] = combined_weights.get(uid, 0.0) + weight
                for uid in list(combined_scores.keys()):
                    combined_scores[uid] /= combined_weights.get(uid, 1.0)
            else:
                if "torch" in score_maps and score_maps["torch"]:
                    combined_scores = dict(score_maps["torch"])
                elif "deepface" in score_maps and score_maps["deepface"]:
                    combined_scores = dict(score_maps["deepface"])
                else:
                    combined_scores = {}
                    print(f"[WARNING] Verify: No scores available!", file=sys.stderr)

            # Helper function to convert euclidean distance to percentage
            def euclidean_to_percent(d: float) -> float:
                d_clamped = max(0.0, min(float(d), 2.0))
                return (1.0 - d_clamped / 2.0) * 100.0

            if metric == "cosine":
                sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            else:
                sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1])

            result["all_scores"] = [
                {
                    "user_id": user_id,
                    "score": score,
                    "percentage": (score * 100.0) if metric == "cosine" else euclidean_to_percent(score),
                    "embeddings_count": counts_map.get(user_id, 0)
                }
                for user_id, score in sorted_combined
            ]

            # Threshold check
            if metric == "cosine":
                threshold = similarity_matcher.cosine_threshold
            else:
                threshold = similarity_matcher.euclidean_threshold
            result["threshold"] = threshold

            matched_id = None
            matched_score = None
            matched_name = None

            if sorted_combined:
                best_user_id, best_score = sorted_combined[0]
                result["score"] = float(best_score)
                # Calculate percentage for the best score
                if metric == "cosine":
                    result["percentage"] = float(best_score * 100.0)
                else:
                    result["percentage"] = float(euclidean_to_percent(best_score))
                
                # Threshold check
                if metric == "cosine":
                    passes = best_score >= threshold
                else:
                    passes = best_score <= threshold
                
                if passes:
                    matched_id = best_user_id
                    matched_score = best_score
                    try:
                        matched_name = registry.get_user_name(matched_id)
                    except Exception as name_e:
                        print(f"[WARN] Failed to get user name for {matched_id}: {name_e}", file=sys.stderr)
                        matched_name = None
                    
                    result["matched_id"] = matched_id
                    result["matched_name"] = matched_name
                    print(f"[INFO] Verification result: MATCH - {matched_id} ({result['percentage']:.2f}%)", file=sys.stderr)
                    try:
                        from app.routers.attendance import _log_attendance, _determine_check_type
                        from datetime import datetime
                        check_type = _determine_check_type(matched_id)
                        _log_attendance({
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "user_id": matched_id,
                            "type": check_type,
                            "match_score": float(matched_score),
                            "liveness_score": float(liveness.get("score", 0.0)),
                            "emotion_label": result.get("emotion_label"),
                            "emotion_confidence": result.get("emotion_confidence")
                        })
                        result["check_type"] = check_type
                    except Exception as log_e:
                        print(f"[WARN] Failed to log attendance: {log_e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                else:
                    result["matched_id"] = None
                    result["matched_name"] = None
                    print(f"[INFO] Verification result: NO MATCH - best score {result['percentage']:.2f}% (threshold: {threshold*100:.0f}%)", file=sys.stderr)
            else:
                result["score"] = None
                result["percentage"] = None
                result["matched_id"] = None

        except Exception as e:
            print(f"Warning: Matching failed: {e}", file=sys.stderr)
            result["all_scores"] = []
            result["score"] = None
            result["matched_id"] = None
        
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

