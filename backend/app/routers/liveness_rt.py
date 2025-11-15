"""
Real-time Liveness Detection Endpoint

This endpoint is optimized for real-time streaming checks:
- Lightweight response format
- Fast processing (~300-500ms per frame)
- Error tolerance (returns safe defaults on failure)
- No blocking operations
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
import numpy as np
from PIL import Image
import io
import cv2
import sys
from typing import Dict, Any
import time

from app.pipelines.liveness import LivenessModel

router = APIRouter()

# Singleton instance for reuse (avoid re-initialization overhead)
_liveness_model_cache = None

def get_liveness_model():
    """Get or create cached liveness model instance."""
    global _liveness_model_cache
    if _liveness_model_cache is None:
        print("[INFO] Initializing liveness model for real-time endpoint...", file=sys.stderr)
        _liveness_model_cache = LivenessModel()
    return _liveness_model_cache


@router.post("/api/liveness/realtime")
async def check_liveness_realtime(image: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Real-time liveness check endpoint.
    
    Optimized for continuous streaming from webcam:
    - Returns lightweight response
    - Fast processing with DeepFace MiniFASNet
    - Graceful error handling (no exceptions thrown to client)
    
    Args:
        image: Uploaded image frame (JPEG/PNG)
        
    Returns:
        {
            "score": float [0, 1] - confidence that face is real,
            "passed": bool - whether liveness check passed,
            "is_real": bool - alias for passed,
            "processing_time_ms": float - time taken for inference,
            "status": str - "success" or "error",
            "message": str - optional message
        }
    """
    start_time = time.time()
    
    try:
        # Read image
        contents = await image.read()
        img = np.array(Image.open(io.BytesIO(contents)))
        
        # Handle different image formats
        if len(img.shape) == 2:
            # Grayscale - convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3:
            if img.shape[2] == 4:
                # RGBA - convert to RGB then BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif img.shape[2] == 3:
                # RGB - convert to BGR
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            return {
                "score": 0.0,
                "passed": False,
                "is_real": False,
                "processing_time_ms": 0,
                "status": "error",
                "message": f"Unsupported image format: {img.shape}"
            }
        
        # Get liveness model
        liveness_model = get_liveness_model()
        
        # Run liveness detection
        result = liveness_model.predict(img)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Return structured response
        return {
            "score": float(result.get("score", 0.0)),
            "passed": bool(result.get("passed", False)),
            "is_real": bool(result.get("passed", False)),  # Alias for clarity
            "processing_time_ms": round(processing_time, 2),
            "status": "success",
            "message": "Liveness check completed"
        }
        
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        
        # Log error but return safe response (don't crash client)
        error_msg = str(e)
        print(f"[ERROR] Real-time liveness check failed: {error_msg}", file=sys.stderr)
        
        # Return safe defaults with error info
        return {
            "score": 0.0,
            "passed": False,
            "is_real": False,
            "processing_time_ms": round(processing_time, 2),
            "status": "error",
            "message": f"Liveness check failed: {error_msg[:100]}"  # Truncate long errors
        }


@router.get("/api/liveness/health")
async def liveness_health_check():
    """
    Health check for real-time liveness endpoint.
    
    Returns model status and performance info.
    """
    try:
        model = get_liveness_model()
        return {
            "status": "healthy",
            "model_loaded": model.model_loaded,
            "backend": getattr(model, "backend", "opencv"),
            "ready": True,
            "message": "Real-time liveness detection ready"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "model_loaded": False,
            "ready": False,
            "error": str(e),
            "message": "Real-time liveness detection not available"
        }
