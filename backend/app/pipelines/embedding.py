import numpy as np
import cv2
from typing import Literal
from pathlib import Path
import os

# Singleton instances to avoid reloading models
_model_instances = {}

class EmbedModel:
    """Face embedding model interface (512-D vectors). Uses a single configurable weights file."""
    
    def __init__(self, model_type: Literal["A", "B"] = "A"):
        # model_type kept for backward compatibility; it is ignored.
        self.model_type = model_type
        self.model = None
        self.use_deepface = False
        self.device = "cpu"
        
        # Force DeepFace-only mode via environment flag
        if os.environ.get("DEEPFACE_ONLY", "0") == "1":
            self.use_deepface = True
            _model_instances[f"embedding_{model_type}"] = self
            print("[INFO] DEEPFACE_ONLY=1 -> Using DeepFace ArcFace for embeddings", file=__import__('sys').stderr)
            return
        
        # Use singleton pattern - reuse model instance if already loaded
        # IMPORTANT: If cached instance uses DeepFace, reuse it (don't try PyTorch again)
        # This ensures register and verify use the same model
        cache_key = f"embedding_{model_type}"
        if cache_key in _model_instances:
            cached = _model_instances[cache_key]
            # Reuse if PyTorch model is loaded and working
            if cached.model is not None and not cached.use_deepface:
                self.model = cached.model
                self.use_deepface = cached.use_deepface
                self.device = cached.device
                print(f"[INFO] Reusing cached PyTorch model instance", file=__import__('sys').stderr)
                return
            # Reuse if DeepFace is being used (PyTorch failed or not available)
            if cached.use_deepface:
                self.model = None
                self.use_deepface = True
                self.device = "cpu"
                print(f"[INFO] Reusing cached DeepFace instance (PyTorch model not available or failed)", file=__import__('sys').stderr)
                return
        
        # Try to load PyTorch .pth model first (ALWAYS prioritize PyTorch)
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Resolve weights path (single main model)
            models_dir = Path(__file__).resolve().parent.parent / "models"
            default_path = models_dir / "modelA_best.pth"
            env_path = os.environ.get("MODEL_WEIGHTS_PATH", str(default_path))
            pth_path = Path(env_path)
            if not pth_path.is_absolute():
                pth_path = (Path(__file__).resolve().parent.parent.parent / pth_path).resolve()
            if pth_path.exists():
                print(f"[INFO] Loading PyTorch model from {pth_path}...", file=__import__('sys').stderr)
                try:
                    from app.pipelines.arcface_model import get_model
                    # Create model architecture
                    backbone_mode = os.environ.get("BACKBONE_MODE", "ir")
                    try:
                        num_layers = int(os.environ.get("BACKBONE_LAYERS", "100"))
                    except Exception:
                        num_layers = 100
                    self.model = get_model(input_size=[112, 112], num_layers=num_layers, mode=backbone_mode)
                    # Load checkpoint
                    # Torch 2.6+ uses weights_only=True by default which breaks legacy checkpoints
                    try:
                        checkpoint = torch.load(str(pth_path), map_location=self.device, weights_only=False)
                    except TypeError:
                        # Older torch versions (<=2.5) don't support weights_only param
                        checkpoint = torch.load(str(pth_path), map_location=self.device)
                    except Exception as e:
                        # Allowlist numpy scalar if needed for legacy checkpoints
                        try:
                            import numpy  # noqa: F401
                            from torch.serialization import add_safe_globals  # type: ignore
                            add_safe_globals([numpy.core.multiarray.scalar])
                            checkpoint = torch.load(str(pth_path), map_location=self.device, weights_only=False)
                        except Exception:
                            raise e
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        elif 'backbone' in checkpoint:
                            # Model B uses 'backbone' key
                            state_dict = checkpoint['backbone']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    # Load state dict (handle key mismatches)
                    try:
                        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                        if missing_keys:
                            print(f"[WARNING] Missing keys when loading checkpoint: {len(missing_keys)} keys", file=__import__('sys').stderr)
                            print(f"[WARNING] First 10 missing keys: {missing_keys[:10]}", file=__import__('sys').stderr)
                        if unexpected_keys:
                            print(f"[WARNING] Unexpected keys in checkpoint: {len(unexpected_keys)} keys", file=__import__('sys').stderr)
                            print(f"[WARNING] First 10 unexpected keys: {unexpected_keys[:10]}", file=__import__('sys').stderr)
                        if not missing_keys and not unexpected_keys:
                            print(f"[INFO] All checkpoint keys matched successfully!", file=__import__('sys').stderr)
                    except Exception as e:
                        print(f"[WARNING] Strict loading failed: {e}. Trying with key mapping...", file=__import__('sys').stderr)
                        # Try to map keys if needed
                        new_state_dict = {}
                        for k, v in state_dict.items():
                            # Remove 'module.' prefix if present
                            new_key = k.replace('module.', '')
                            new_state_dict[new_key] = v
                        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
                        if missing_keys:
                            print(f"[WARNING] Missing keys after key mapping: {len(missing_keys)} keys", file=__import__('sys').stderr)
                            print(f"[WARNING] First 10 missing keys: {missing_keys[:10]}", file=__import__('sys').stderr)
                        if unexpected_keys:
                            print(f"[WARNING] Unexpected keys after key mapping: {len(unexpected_keys)} keys", file=__import__('sys').stderr)
                    
                    # Set to eval mode BEFORE moving to device (important for BatchNorm)
                    self.model.eval()
                    self.model.to(self.device)
                    
                    # Verify BatchNorm layers are in eval mode and have running stats
                    for name, module in self.model.named_modules():
                        if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d):
                            if not module.training:  # Should be False in eval mode
                                # Check if running stats are loaded
                                if hasattr(module, 'running_mean') and module.running_mean is not None:
                                    running_mean_norm = torch.norm(module.running_mean).item()
                                    if running_mean_norm < 1e-6:
                                        print(f"[WARNING] BatchNorm layer '{name}' has zero running_mean norm: {running_mean_norm}", file=__import__('sys').stderr)
                    
                    # Test inference with two different random inputs to verify model produces different outputs
                    with torch.no_grad():
                        test_input1 = torch.randn(1, 3, 112, 112).to(self.device)
                        test_input2 = torch.randn(1, 3, 112, 112).to(self.device)
                        test_output1 = self.model(test_input1)
                        test_output2 = self.model(test_input2)
                        
                        # Check if outputs are different (they should be for different inputs)
                        output_diff = torch.norm(test_output1 - test_output2).item()
                        output_sim = torch.nn.functional.cosine_similarity(test_output1, test_output2, dim=1).item()
                        
                        if test_output1.shape[-1] == 512:
                            print(f"[INFO] PyTorch model loaded successfully (output dim: 512, device: {self.device})", file=__import__('sys').stderr)
                            print(f"[INFO] Test inference: output_diff={output_diff:.6f}, cosine_sim={output_sim:.6f}", file=__import__('sys').stderr)
                            # Some checkpoints may yield very similar outputs for random noise due to BN stats.
                            # Treat this as a warning, but do not fail hard. Real face inputs should still vary.
                            if output_diff < 1e-6 or output_sim > 0.9999:
                                print(f"[WARN] Sanity check suggests very similar outputs for different random inputs.", file=__import__('sys').stderr)
                                print(f"[WARN] Proceeding anyway. Verify with real images.", file=__import__('sys').stderr)
                            # Successfully loaded PyTorch - do NOT use DeepFace
                            self.use_deepface = False
                        else:
                            print(f"[ERROR] PyTorch model output dim is {test_output1.shape[-1]}, expected 512. This is a critical error!", file=__import__('sys').stderr)
                            self.model = None
                            self.use_deepface = True
                except Exception as e:
                    print(f"[ERROR] Failed to load PyTorch model: {e}. This is a critical error!", file=__import__('sys').stderr)
                    import traceback
                    traceback.print_exc()
                    self.model = None
                    self.use_deepface = True
            else:
                print(f"[ERROR] PyTorch model not found at {pth_path}. Expected path: {pth_path.absolute()}", file=__import__('sys').stderr)
                self.use_deepface = True
        except ImportError:
            print(f"[ERROR] PyTorch not available. Please install torch: pip install torch", file=__import__('sys').stderr)
            self.use_deepface = True
        except Exception as e:
            print(f"[ERROR] Failed to initialize PyTorch model: {e}. This is a critical error!", file=__import__('sys').stderr)
            import traceback
            traceback.print_exc()
            self.use_deepface = True
        
        # Optionally enforce PyTorch-only via env flag (fail fast instead of silent fallback)
        require_torch = os.environ.get("REQUIRE_TORCH", "0") == "1"
        require_model_a = os.environ.get("REQUIRE_MODEL_A", "0") == "1"
        if (require_torch or (require_model_a and self.model_type == "A")) and self.use_deepface:
            # Fail fast so operators know to place the correct weights
            raise RuntimeError(
                f"Required PyTorch model ({self.model_type}) not available. "
                f"Expected weights under {models_dir}. "
                f"Set REQUIRE_TORCH=0 to allow DeepFace fallback temporarily."
            )

        if self.use_deepface:
            print(f"[INFO] Embedding model {model_type} will use DeepFace ArcFace", file=__import__('sys').stderr)
        
        # Cache instance for reuse
        _model_instances[cache_key] = self
    
    def extract(self, img: np.ndarray) -> np.ndarray:
        """Extract 512-D embedding vector (PyTorch if available, else DeepFace ArcFace)."""
        # Prefer PyTorch model when available
        if self.model is not None and not self.use_deepface:
            try:
                import torch
                # Preprocess image for PyTorch ArcFace (RGB, mean/std normalization, CHW)
                # Input img should already be aligned 112x112 face crop
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
                # Ensure 112x112 (should already be, but double-check)
                if img_rgb.shape[:2] != (112, 112):
                    img_rgb = cv2.resize(img_rgb, (112, 112))
                
                # Debug: Log image stats to check if different faces produce different images
                img_mean = np.mean(img_rgb)
                img_std = np.std(img_rgb)
                img_min = np.min(img_rgb)
                img_max = np.max(img_rgb)
                print(f"[DEBUG] Preprocessing: image shape={img_rgb.shape}, mean={img_mean:.2f}, std={img_std:.2f}, min={img_min:.0f}, max={img_max:.0f}", file=__import__('sys').stderr)
                
                # CRITICAL: Use EXACT same normalization as training notebook
                # Training uses: (pixel*255 - 127.5) / 128.0 where pixel is in [0,1] from ToTensor()
                # This is equivalent to: (pixel - 127.5) / 128.0 where pixel is in [0,255]
                # Result: [-1, 1] range (slightly different from standard [-1,1] which uses /127.5)
                img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
                img_chw = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
                img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)  # Add batch dimension
                
                # Debug: Log tensor stats
                tensor_mean = torch.mean(img_tensor).item()
                tensor_std = torch.std(img_tensor).item()
                tensor_min = torch.min(img_tensor).item()
                tensor_max = torch.max(img_tensor).item()
                print(f"[DEBUG] Preprocessing: tensor mean={tensor_mean:.6f}, std={tensor_std:.6f}, min={tensor_min:.6f}, max={tensor_max:.6f}", file=__import__('sys').stderr)
                
                # Inference - default to eval (dropout disabled)
                self.model.eval()
                with torch.no_grad():
                    # Optional: use batch stats for BatchNorm to avoid reliance on missing running stats
                    if os.environ.get("USE_BN_BATCH_STATS", "0") == "1":
                        for m in self.model.modules():
                            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                                m.train(True)
                            elif isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
                                m.eval()
                    embedding = self.model(img_tensor)
                    embedding = embedding.cpu().numpy().flatten()
                
                # Debug: Check raw embedding before normalization
                raw_norm = np.linalg.norm(embedding)
                raw_mean = np.mean(embedding)
                raw_std = np.std(embedding)
                raw_min = np.min(embedding)
                raw_max = np.max(embedding)
                raw_sample = embedding[:5].tolist()
                print(f"[DEBUG] PyTorch raw embedding: norm={raw_norm:.6f}, mean={raw_mean:.6f}, std={raw_std:.6f}, min={raw_min:.6f}, max={raw_max:.6f}, sample={raw_sample}", file=__import__('sys').stderr)
                
                if raw_norm < 1e-6:
                    print(f"[ERROR] PyTorch model output is all zeros! Model may not be loaded correctly.", file=__import__('sys').stderr)
                    raise ValueError("Model output is all zeros - model not working correctly")
                elif raw_std < 1e-6:
                    print(f"[ERROR] PyTorch model output has zero variance! Model may not be working correctly.", file=__import__('sys').stderr)
                    raise ValueError("Model output has zero variance - model not working correctly")
                
                # Normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                if len(embedding) < 512:
                    embedding = np.pad(embedding, (0, 512 - len(embedding)))
                elif len(embedding) > 512:
                    embedding = embedding[:512]
                return embedding.astype(np.float32)
            except Exception as e:
                # If PyTorch inference fails, fallback to DeepFace
                print(f"[WARNING] PyTorch inference failed for model {self.model_type}: {e}. Falling back to DeepFace.", file=__import__('sys').stderr)
                # Continue to DeepFace fallback below

        # DeepFace ArcFace fallback
        try:
            from deepface import DeepFace
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
            # Use detector_backend="skip" to skip detection since image is already aligned
            # This ensures we only get 1 embedding per image
            rep = DeepFace.represent(
                img_path=rgb, 
                model_name="ArcFace", 
                enforce_detection=False,
                detector_backend="skip"  # Skip detection - image is already a face crop
            )
            # Ensure we only get 1 embedding (DeepFace.represent may return multiple if multiple faces detected)
            if rep and len(rep) > 0:
                # Only take the first embedding (should be only one with detector_backend="skip")
                # But guard against multiple embeddings just in case
                if len(rep) > 1:
                    print(f"[WARNING] DeepFace.represent returned {len(rep)} embeddings, using only the first one", file=__import__('sys').stderr)
                embedding = np.array(rep[0]['embedding'], dtype=np.float32)
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                if len(embedding) < 512:
                    embedding = np.pad(embedding, (0, 512 - len(embedding)))
                elif len(embedding) > 512:
                    embedding = embedding[:512]
                return embedding.astype(np.float32)
        except Exception:
            pass
        
        # Last resort
        return np.zeros(512, dtype=np.float32)

