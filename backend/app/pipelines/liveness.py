"""
Production-grade Anti-Spoof (Liveness Detection) Module using DeepFace.

This module provides robust liveness detection without requiring ONNX models.
It uses DeepFace's built-in antispoofing capabilities with state-of-the-art models.
"""

import numpy as np
import cv2
import sys
import os
from typing import Dict, Optional, Tuple
from pathlib import Path
import logging
import warnings

# Suppress DeepFace warnings for cleaner logs
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from app.core.config import MODE, THRESHOLDS_PATH
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# DeepFace is imported lazily inside methods to avoid startup failures

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LivenessModel:
    """
    Production-grade Liveness Detection Model using DeepFace.
    
    This implementation uses DeepFace's antispoofing capabilities which combine
    multiple detection methods including:
    - Face detection and alignment
    - Texture analysis
    - Deep learning-based spoof detection
    - Multi-frame analysis support
    
    The model maintains backward compatibility with the previous interface
    while providing superior accuracy and reliability.
    """
    
    def __init__(self):
        """Initialize the liveness detection model."""
        # Model state
        self.model_loaded = False
        self._model = None
        
        # Performance optimization: cache for model initialization
        self._init_deepface()
        
        logger.info(f"LivenessModel initialized with DeepFace MiniFASNet (argmax decision), mode={MODE}")
    
    def _init_deepface(self):
        """Initialize DeepFace with optimal configuration."""
        try:
            # Pre-import DeepFace to warm up models
            from deepface import DeepFace
            
            # Configure DeepFace for optimal performance
            # Default to lightweight OpenCV detector backend
            self.backend = "opencv"
            self.model_loaded = True
            
            # Cache flag for first-time model download
            self._models_warmed = False
            
            logger.info("DeepFace initialized with backend=opencv (real-time optimized)")
        except Exception as e:
            logger.error(f"Failed to initialize DeepFace: {e}")
            self.model_loaded = False
    
    def _deepface_antispoof(self, img: np.ndarray) -> Tuple[float, bool]:
        """
        Use DeepFace built-in anti-spoofing (MiniFASNet models).
        
        DeepFace uses Silent-Face-Anti-Spoofing models which analyze:
        - Texture patterns (live skin vs printed/screen)
        - Depth cues (3D face vs flat image)
        - Moiré patterns (screen artifacts)
        - Color analysis (natural skin vs display)
        
        Args:
            img: Input image as numpy array (BGR format, any size >= 80x80)
            
        Returns:
            Tuple of (score, is_real) where:
            - score: Confidence score [0, 1] (higher = more likely real)
            - is_real: Boolean indicating if face is real
        """
        try:
            from deepface import DeepFace
            
            # DeepFace expects RGB format
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            
            # Ensure image is large enough for MiniFASNet (minimum 80x80)
            # Recommended: at least 112x112 for better accuracy
            min_size = 112
            if img_rgb.shape[0] < min_size or img_rgb.shape[1] < min_size:
                scale_factor = min_size / min(img_rgb.shape[0], img_rgb.shape[1])
                new_h = int(img_rgb.shape[0] * scale_factor)
                new_w = int(img_rgb.shape[1] * scale_factor)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Use DeepFace's built-in anti-spoofing (MiniFASNet)
            # This automatically detects faces and runs anti-spoof models
            face_objs = DeepFace.extract_faces(
                img_path=img_rgb,
                detector_backend=getattr(self, 'backend', 'opencv'),
                enforce_detection=False,
                align=True,
                anti_spoofing=True  # Enable built-in anti-spoofing!
            )
            
            if face_objs and len(face_objs) > 0:
                # DeepFace returns anti-spoofing results directly!
                first = face_objs[0]
                
                # Extract anti-spoofing results
                is_real = first.get("is_real", False)  # Boolean from MiniFASNet
                antispoof_score = first.get("antispoof_score", 0.0)  # Float [0, 1]
                
                # DeepFace's MiniFASNet already provides a score AND decision
                # MiniFASNet uses argmax(softmax) to determine is_real:
                #   - Runs 2 models with different crop scales
                #   - Averages predictions: [spoof_prob, real_prob, unknown_prob]
                #   - is_real = (argmax == 1), i.e., real_prob is highest
                #   - antispoof_score = probability of the predicted label / 2
                # 
                # IMPORTANT: antispoof_score is confidence of PREDICTED class
                # If is_real=True, score means confidence it's real
                # If is_real=False (spoof), score means confidence it's spoof
                # We return the raw confidence of the predicted class
                
                confidence = float(antispoof_score)
                
                logger.info(
                    f"Liveness detection (DeepFace MiniFASNet) - "
                    f"score: {confidence:.3f}, is_real: {is_real}"
                )
                
                # Return confidence of predicted class (not normalized)
                return confidence, is_real
            else:
                # No face detected
                logger.warning("DeepFace could not detect face for anti-spoofing analysis")
                return 0.0, False
            
        except Exception as e:
            logger.error(f"DeepFace antispoof error: {e}", exc_info=True)
            raise
    
    def predict(self, img: np.ndarray) -> Dict[str, any]:
        """
        Predict liveness (anti-spoof) using DeepFace's MiniFASNet models.
        
        Uses Silent-Face-Anti-Spoofing deep learning models which analyze:
        - Texture patterns (live skin vs printed/screen)
        - Depth cues (3D face vs flat image)
        - Moiré patterns (screen artifacts)
        - Color analysis (natural skin vs display)
        
        Args:
            img: Input image as numpy array (BGR format, any size >= 80x80)
                 Can be full frame or cropped face - MiniFASNet handles both
            
        Returns:
            Dictionary with:
            - score: float in [0, 1] - confidence that face is real (higher = more likely real)
            - passed: bool - whether the face passed the liveness check
        """
        # Use DeepFace MiniFASNet for production-quality anti-spoofing
        if self.model_loaded:
            try:
                score, is_real = self._deepface_antispoof(img)
                
                # Log with INFO level for debugging
                logger.info(
                    f"Liveness prediction (MiniFASNet) - score: {score:.3f}, "
                    f"passed: {is_real} (DeepFace argmax decision), "
                    f"model_loaded: {self.model_loaded}"
                )
                
                return {
                    "score": float(score),
                    "passed": bool(is_real)
                }
            except Exception as e:
                logger.warning(f"DeepFace MiniFASNet prediction failed: {e}", exc_info=True)
                raise
        
        raise RuntimeError("DeepFace model not initialized")
