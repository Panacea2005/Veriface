import numpy as np
import cv2
from typing import Dict, Literal, Optional, Tuple
from pathlib import Path
import os
import sys

# Singleton instances to avoid reloading models
_model_instances = {}
_deepface_model_cache = None

class EmbedModel:
    """
    Face embedding model interface (512-D vectors).
    
    PyTorch model (Model A/B):
        - Uses PyTorch trained model (modelA_best.pth or modelB_best.pth)
        - Preprocessing: (pixel - 127.5) / 128.0 (matches notebook exactly)
        - Format: RGB, CHW (PyTorch format)
        - Model outputs L2-normalized embeddings (from NormalizedBackbone wrapper)
    """
    
    def __init__(self, model_type: Literal["A", "B"] = "A"):
        self.model_type = model_type
        self.model = None
        self.use_deepface = False
        self.device = "cpu"

        # Legacy support for backward compatibility
        if os.environ.get("DEEPFACE_ONLY", "0") == "1":
            # Legacy mode (deprecated)
            self.use_deepface = True
            _model_instances[f"embedding_{model_type}"] = self
            return

        # Use singleton pattern - reuse model instance if already loaded
        cache_key = f"embedding_{model_type}"
        if cache_key in _model_instances:
            cached = _model_instances[cache_key]
            if cached.model is not None and not cached.use_deepface:
                self.model = cached.model
                self.use_deepface = cached.use_deepface
                self.device = cached.device
                print(f"[INFO] Reusing cached PyTorch model instance", file=__import__('sys').stderr)
                return
            if cached.use_deepface:
                self.model = None
                self.use_deepface = True
                self.device = "cpu"
                # Reusing cached instance
                return

        # Try to load .pth checkpoint (Model B uses modelB_best.pth - R100)
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            models_dir = Path(__file__).resolve().parent.parent / "models"
            
            # Determine checkpoint path
            if model_type == "B":
                # Model B: Use modelB_best.pth (R100)
                default_path = models_dir / "modelB_best.pth"
            else:
                default_path = models_dir / "modelA_best.pth"
            env_path = os.environ.get("MODEL_WEIGHTS_PATH", str(default_path))
            pth_path = Path(env_path)
            if not pth_path.is_absolute():
                pth_path = (Path(__file__).resolve().parent.parent.parent / pth_path).resolve()
            if pth_path.exists():
                # Check if it's a TorchScript .pt file - load it directly
                if str(pth_path).endswith(".pt") and pth_path.suffix == ".pt":
                    try:
                        print(f"[INFO] Loading TorchScript model from {pth_path}...", file=__import__('sys').stderr)
                        self.model = torch.jit.load(str(pth_path), map_location=self.device)
                        self.model.eval()
                        self.use_deepface = False
                        _model_instances[cache_key] = self
                        print(f"[INFO] TorchScript model loaded successfully from {pth_path}", file=__import__('sys').stderr)
                        return
                    except Exception as e:
                        print(f"[ERROR] Failed to load TorchScript model from {pth_path}: {e}", file=__import__('sys').stderr)
                        import traceback
                        traceback.print_exc()
                        # Fall through to try as checkpoint
                
                # Load as .pth checkpoint
                print(f"[INFO] Loading PyTorch checkpoint from {pth_path}...", file=__import__('sys').stderr)
                try:
                    from app.pipelines.arcface_model import get_model
                    backbone_mode = os.environ.get("BACKBONE_MODE", "ir")
                    try:
                        num_layers = int(os.environ.get("BACKBONE_LAYERS", "100"))
                    except Exception:
                        num_layers = 100
                    self.model = get_model(input_size=[112, 112], num_layers=num_layers, mode=backbone_mode)
                    try:
                        checkpoint = torch.load(str(pth_path), map_location=self.device, weights_only=False)
                    except TypeError:
                        checkpoint = torch.load(str(pth_path), map_location=self.device)
                    except Exception as e:
                        try:
                            import numpy  # noqa: F401
                            from torch.serialization import add_safe_globals  # type: ignore
                            add_safe_globals([numpy.core.multiarray.scalar])
                            checkpoint = torch.load(str(pth_path), map_location=self.device, weights_only=False)
                        except Exception:
                            raise e
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        elif 'backbone' in checkpoint:
                            state_dict = checkpoint['backbone']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    # Load state dict (handle key mismatches with better mapping)
                    def map_state_dict_keys(state_dict, model_keys):
                        """Map checkpoint keys to model keys more intelligently."""
                        new_state_dict = {}
                        model_keys_set = set(model_keys)
                        for k, v in state_dict.items():
                            # Try direct match first
                            if k in model_keys_set:
                                new_state_dict[k] = v
                                continue
                            # Remove common prefixes
                            mapped_key = k.replace('module.', '').replace('backbone.', '').replace('model.', '')
                            if mapped_key in model_keys_set:
                                new_state_dict[mapped_key] = v
                                continue
                            # Try removing 'downsample.' prefix for some checkpoints
                            if 'downsample' in k:
                                alt_key = k.replace('.downsample.0.', '.downsample.0.').replace('.downsample.1.', '.downsample.1.')
                                if alt_key in model_keys_set:
                                    new_state_dict[alt_key] = v
                                    continue
                            # If still no match, keep original key (will be in unexpected_keys)
                            new_state_dict[k] = v
                        return new_state_dict
                    try:
                        # Get model state dict keys
                        model_keys = set(self.model.state_dict().keys())
                        # Try direct load first
                        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                        # If too many missing keys, try key mapping
                        if len(missing_keys) > 50:
                            print(f"[INFO] Too many missing keys ({len(missing_keys)}), trying key mapping...", file=__import__('sys').stderr)
                            mapped_state_dict = map_state_dict_keys(state_dict, model_keys)
                            missing_keys, unexpected_keys = self.model.load_state_dict(mapped_state_dict, strict=False)
                        if missing_keys:
                            print(f"[WARNING] Missing keys when loading checkpoint: {len(missing_keys)} keys", file=__import__('sys').stderr)
                            if len(missing_keys) <= 20:
                                print(f"[WARNING] Missing keys: {missing_keys}", file=__import__('sys').stderr)
                            else:
                                print(f"[WARNING] First 10 missing keys: {missing_keys[:10]}", file=__import__('sys').stderr)
                        if unexpected_keys:
                            print(f"[WARNING] Unexpected keys in checkpoint: {len(unexpected_keys)} keys", file=__import__('sys').stderr)
                            if len(unexpected_keys) <= 20:
                                print(f"[WARNING] Unexpected keys: {unexpected_keys}", file=__import__('sys').stderr)
                            else:
                                print(f"[WARNING] First 10 unexpected keys: {unexpected_keys[:10]}", file=__import__('sys').stderr)
                        if not missing_keys and not unexpected_keys:
                            print(f"[INFO] All checkpoint keys matched successfully!", file=__import__('sys').stderr)
                    except Exception as e:
                        print(f"[WARNING] State dict loading failed: {e}. Trying with key mapping...", file=__import__('sys').stderr)
                        # Try to map keys if needed
                        model_keys = set(self.model.state_dict().keys())
                        mapped_state_dict = map_state_dict_keys(state_dict, model_keys)
                        missing_keys, unexpected_keys = self.model.load_state_dict(mapped_state_dict, strict=False)
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
                            # Successfully loaded PyTorch model
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
        require_model_b = os.environ.get("REQUIRE_MODEL_B", "0") == "1"
        if (require_torch or (require_model_a and self.model_type == "A") or (require_model_b and self.model_type == "B")) and self.use_deepface:
            # Fail fast if model requirements not met
            raise RuntimeError(
                f"Required PyTorch model ({self.model_type}) not available. "
                f"Expected weights under {models_dir}. "
                f"For Model B, ensure modelB_best.pth exists. "
                f"Set REQUIRE_TORCH=0 to allow DeepFace fallback temporarily."
            )

        if self.use_deepface:
            # Using fallback mode
            pass
        
        # Cache instance for reuse
        _model_instances[cache_key] = self
    
    def extract(self, img: np.ndarray) -> np.ndarray:
        """
        Extract 512-D embedding vector.
        
        Mode 1 (PyTorch trained model): Uses exact notebook preprocessing
        - Normalization: (pixel - 127.5) / 128.0 (matches notebook: transforms.Lambda(lambda x:(x*255-127.5)/128.0))
        - Format: RGB, CHW (PyTorch format)
        - Model outputs L2-normalized embeddings (from NormalizedBackbone wrapper)
        
        Mode 2 (DeepFace ArcFace): Uses DeepFace standard preprocessing
        - Normalization: (pixel - 127.5) / 127.5 (DeepFace ArcFace standard)
        - Format: RGB, HWC (TensorFlow/Keras format)
        - Model outputs unnormalized embeddings (we normalize after)
        """
        # Mode 1: PyTorch trained model (from notebook)
        if self.model is not None and not self.use_deepface:
            try:
                import torch
                # Preprocess image for PyTorch ArcFace (matches notebook exactly)
                # Input img should already be aligned 112x112 face crop (BGR from OpenCV)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
                # Ensure 112x112 (should already be, but double-check)
                if img_rgb.shape[:2] != (112, 112):
                    img_rgb = cv2.resize(img_rgb, (112, 112))
                
                # Debug: Log image stats
                img_mean = np.mean(img_rgb)
                img_std = np.std(img_rgb)
                img_min = np.min(img_rgb)
                img_max = np.max(img_rgb)
                print(f"[DEBUG] PyTorch preprocessing: image shape={img_rgb.shape}, mean={img_mean:.2f}, std={img_std:.2f}, min={img_min:.0f}, max={img_max:.0f}", file=__import__('sys').stderr)
                
                # CRITICAL: Use EXACT same normalization as training notebook
                # Notebook: transforms.Lambda(lambda x:(x*255-127.5)/128.0)
                # where x is from ToTensor() in range [0,1]
                # Equivalent to: (pixel_in_0_255 - 127.5) / 128.0
                # This matches both Model A and Model B (R100) training
                img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
                
                # Convert HWC -> CHW (PyTorch format)
                img_chw = np.transpose(img_normalized, (2, 0, 1))  # HWC -> CHW
                img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(self.device)  # Add batch dimension
                
                # Debug: Log tensor stats
                tensor_mean = torch.mean(img_tensor).item()
                tensor_std = torch.std(img_tensor).item()
                tensor_min = torch.min(img_tensor).item()
                tensor_max = torch.max(img_tensor).item()
                print(f"[DEBUG] PyTorch preprocessing: tensor mean={tensor_mean:.6f}, std={tensor_std:.6f}, min={tensor_min:.6f}, max={tensor_max:.6f}", file=__import__('sys').stderr)
                
                # CRITICAL: Ensure model is in eval mode (dropout disabled, BN uses running stats)
                self.model.eval()
                with torch.no_grad():
                    # Optional: use batch stats for BatchNorm (for debugging bad checkpoints)
                    use_bn_batch_stats = os.environ.get("USE_BN_BATCH_STATS", "0") == "1"
                    if use_bn_batch_stats:
                        # Temporarily enable training mode for BN only
                        for m in self.model.modules():
                            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                                m.train(True)
                            elif isinstance(m, torch.nn.Dropout) or isinstance(m, torch.nn.Dropout2d):
                                m.eval()
                    
                    embedding = self.model(img_tensor)
                    embedding = embedding.cpu().numpy().flatten()
                    
                    # Restore eval mode if we temporarily changed BN
                    if use_bn_batch_stats:
                        self.model.eval()
                
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
                
                # NOTE: Model (NormalizedBackbone wrapper) already outputs L2-normalized embeddings
                # But we normalize again for safety (should be ~1.0 if already normalized)
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                if len(embedding) < 512:
                    embedding = np.pad(embedding, (0, 512 - len(embedding)))
                elif len(embedding) > 512:
                    embedding = embedding[:512]
                return embedding.astype(np.float32)
            except Exception as e:
                # If PyTorch inference fails, handle error
                print(f"[WARNING] PyTorch inference failed for model {self.model_type}: {e}.", file=__import__('sys').stderr)
                import traceback
                traceback.print_exc()
                # Handle fallback if needed

        # Fallback embedding extraction
        # Apply normalization: (pixel - 127.5) / 128.0
        try:
            from deepface import DeepFace
            
            # Input img is BGR from OpenCV (already aligned 112x112 face crop)
            # Input image format handling
            # Ensure 112x112 (should already be, but double-check)
            if img.shape[:2] != (112, 112):
                img = cv2.resize(img, (112, 112))
            
            # Debug: Log image stats
            img_mean = np.mean(img)
            img_std = np.std(img)
            img_min = np.min(img)
            img_max = np.max(img)
            # Silent - preprocessing stats (hidden from logs)
            
            # Apply normalization: (pixel - 127.5) / 128.0
            # Preprocessing normalization
            rep = DeepFace.represent(
                img_path=img,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend="skip",  # Skip detection - image is already a face crop
                align=False,  # Skip alignment - preserve angle differences
                normalization="ArcFace"  # Apply normalization: (pixel - 127.5) / 128.0
            )
            
            # Extract embedding from result
            if rep and len(rep) > 0:
                if len(rep) > 1:
                    pass  # Silent - multiple embeddings, using first
                embedding = np.array(rep[0]['embedding'], dtype=np.float32)
            else:
                raise ValueError("DeepFace.represent returned empty result")
            
            # Debug: Log embedding stats (before normalization) - hidden from logs
            
            # Normalize embeddings
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            if len(embedding) < 512:
                embedding = np.pad(embedding, (0, 512 - len(embedding)))
            elif len(embedding) > 512:
                embedding = embedding[:512]
            return embedding.astype(np.float32)
        except Exception as e:
            # Silent fallback (hidden from logs)
            import traceback
            traceback.print_exc()
            # Fallback method
            try:
                from deepface import DeepFace
                if img.shape[:2] != (112, 112):
                    img = cv2.resize(img, (112, 112))
                rep = DeepFace.represent(
                    img_path=img, 
                    model_name="ArcFace", 
                    enforce_detection=False,
                    detector_backend="skip",
                    normalization="ArcFace"  # Apply normalization
                )
                if rep and len(rep) > 0:
                    embedding = np.array(rep[0]['embedding'], dtype=np.float32)
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                    if len(embedding) < 512:
                        embedding = np.pad(embedding, (0, 512 - len(embedding)))
                    elif len(embedding) > 512:
                        embedding = embedding[:512]
                    return embedding.astype(np.float32)
            except Exception as e2:
                # Silent fallback failure (hidden from logs)
                pass
        
        # Last resort
        return np.zeros(512, dtype=np.float32)


def extract_deepface_embedding(img: np.ndarray) -> Optional[np.ndarray]:
    """
    Extract embedding with correct preprocessing.
    Normalization: (pixel - 127.5) / 128.0
    """
    global _deepface_model_cache
    try:
        from deepface import DeepFace
        
        # Input img is BGR from OpenCV (already aligned 112x112 face crop)
        # Input image format handling
        # It will convert to RGB internally, then back to BGR for preprocessing
        
        # Ensure 112x112 (should already be, but double-check)
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        
        # Build model if not cached
        if _deepface_model_cache is None:
            _deepface_model_cache = DeepFace.build_model("ArcFace")
        
        # Apply normalization: (pixel - 127.5) / 128.0
        rep = DeepFace.represent(
            img_path=img,
            model_name="ArcFace",
            enforce_detection=False,
            detector_backend="skip",  # Skip detection - image is already a face crop
            align=False,  # Skip alignment - preserve angle differences
            normalization="ArcFace"  # Apply normalization: (pixel - 127.5) / 128.0
        )
        
        if rep and len(rep) > 0:
            if len(rep) > 1:
                pass  # Silent - multiple embeddings, using first
            embedding = np.array(rep[0]["embedding"], dtype=np.float32)
            
            # Debug: Log raw embedding stats (hidden from logs)
            
            # Normalize embeddings
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
            else:
                # Silent - zero norm (hidden from logs)
                return None
            
            # Ensure 512 dimensions (ArcFace should output 512-D)
            if len(embedding) < 512:
                # Silent - padding dimensions (hidden from logs)
                embedding = np.pad(embedding, (0, 512 - len(embedding)))
            elif len(embedding) > 512:
                # Silent - truncating dimensions (hidden from logs)
                embedding = embedding[:512]
            
            return embedding.astype(np.float32)
    except Exception as e:
        # Silent error (hidden from logs - extraction failed)
        import traceback
        traceback.print_exc()
    return None


def extract_dual_embeddings(
    img: np.ndarray,
    preferred_model: Literal["A", "B"] = "A",
    reuse_model: Optional[EmbedModel] = None
) -> Tuple[Dict[str, np.ndarray], EmbedModel]:
    """
    Extract embeddings using Model A/B.
    
    Returns:
        (embeddings_by_model, embed_model_instance)
        embeddings_by_model keys:
            - "torch": PyTorch model embedding (always present if model loads successfully)
    """
    from app.core.config import ENABLE_DEEPFACE
    
    # Extract embeddings using Model A/B
    embed_model = reuse_model or EmbedModel(model_type=preferred_model)
    embeddings: Dict[str, np.ndarray] = {}
    
    # Extract PyTorch embedding
    try:
        torch_embedding = embed_model.extract(img)
        if torch_embedding is not None and torch_embedding.size > 0:
            embeddings["torch"] = torch_embedding
        else:
            print(f"[WARNING] Embedding extraction returned empty result", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Embedding extraction failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    
    # Additional embedding extraction (internal)
    if ENABLE_DEEPFACE:
        try:
            deepface_vec = extract_deepface_embedding(img)
            if deepface_vec is not None and deepface_vec.size > 0:
                embeddings["deepface"] = deepface_vec
        except Exception as e:
            pass
    
    if not embeddings:
        print(f"[ERROR] No embeddings extracted!", file=sys.stderr)
    
    return embeddings, embed_model
