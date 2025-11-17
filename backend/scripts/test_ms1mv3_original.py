"""Test original MS1MV3 pretrained model to check if it produces diverse embeddings.

This script tests the original ms1mv3_arcface_r100_fp16.pth model to see if it works correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MODELS_DIR = BACKEND_ROOT / "app" / "models"
MS1MV3_PATH = MODELS_DIR / "ms1mv3_arcface_r100_fp16.pth"

print("=" * 80)
print("ORIGINAL MS1MV3 PRETRAINED MODEL TEST")
print("=" * 80)
print()

# ============================================================================
# 1. Load MS1MV3 Model
# ============================================================================
print("1. LOADING MS1MV3 MODEL")
print("-" * 80)

if not MS1MV3_PATH.exists():
    print(f"[ERROR] MS1MV3 model not found at: {MS1MV3_PATH}")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# Try to load as InsightFace iResNet100
try:
    # Download iResNet implementation if needed
    import importlib.util
    import urllib.request
    import os
    
    iresnet_path = Path(__file__).parent.parent.parent / "iresnet_insightface.py"
    if not iresnet_path.exists():
        URL = 'https://raw.githubusercontent.com/deepinsight/insightface/master/recognition/arcface_torch/backbones/iresnet.py'
        urllib.request.urlretrieve(URL, str(iresnet_path))
        print(f"[INFO] Downloaded iResNet implementation")
    
    # Load the iresnet module
    spec = importlib.util.spec_from_file_location('iresnet_insightface', str(iresnet_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['iresnet_insightface'] = mod
    spec.loader.exec_module(mod)
    
    # Build iResNet100 backbone
    backbone = mod.iresnet100(num_features=512).to(device)
    
    # Load pretrained weights
    checkpoint = torch.load(str(MS1MV3_PATH), map_location=device)
    missing, unexpected = backbone.load_state_dict(checkpoint, strict=False)
    print(f"[OK] MS1MV3 model loaded")
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    
except Exception as e:
    print(f"[ERROR] Failed to load MS1MV3 model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 2. Wrap with NormalizedBackbone (like training notebook)
# ============================================================================
print("\n2. WRAPPING WITH NORMALIZEDBACKBONE")
print("-" * 80)

class NormalizedBackbone(torch.nn.Module):
    """Wrapper that adds L2 normalization to any backbone's forward pass"""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, x):
        x = self.backbone(x)
        # CRITICAL: L2 normalize embeddings (project onto unit sphere)
        x = F.normalize(x, p=2, dim=1)
        return x

model = NormalizedBackbone(backbone).to(device)
model.eval()
print(f"[OK] Model wrapped with NormalizedBackbone")

# ============================================================================
# 3. Test with Random Inputs
# ============================================================================
print("\n3. TESTING WITH RANDOM INPUTS")
print("-" * 80)
print("Creating 10 different random inputs...")

embeddings_random = []
for i in range(10):
    # Create random input (simulating different faces)
    test_input = torch.randn(1, 3, 112, 112).to(device)
    
    with torch.no_grad():
        output = model(test_input)
        embedding = output.cpu().numpy().flatten()
        embeddings_random.append(embedding)
    
    norm = np.linalg.norm(embedding)
    print(f"  Input {i+1}: norm={norm:.6f}")

# Check diversity
print("\nChecking diversity between random inputs...")
similarities_random = []
for i in range(len(embeddings_random)):
    for j in range(i + 1, len(embeddings_random)):
        sim = np.dot(embeddings_random[i], embeddings_random[j])
        similarities_random.append(sim)

mean_sim = np.mean(similarities_random)
std_sim = np.std(similarities_random)
min_sim = np.min(similarities_random)
max_sim = np.max(similarities_random)

print(f"  Pairs: {len(similarities_random)}")
print(f"  Mean similarity:  {mean_sim:.4f} ({mean_sim*100:.2f}%)")
print(f"  Std:              {std_sim:.4f}")
print(f"  Range:            [{min_sim:.4f}, {max_sim:.4f}]")

if mean_sim > 0.9:
    print(f"  [ERROR] Model COLLAPSED - All embeddings too similar!")
elif mean_sim > 0.7:
    print(f"  [WARN] Model may be collapsed - Mean similarity {mean_sim:.4f} > 0.7")
else:
    print(f"  [OK] Model produces diverse embeddings for random inputs")

# ============================================================================
# 4. Test with Production Preprocessing (ArcFace normalization)
# ============================================================================
print("\n4. TESTING WITH PRODUCTION PREPROCESSING (ArcFace)")
print("-" * 80)
print("Creating 10 different images with ArcFace normalization...")

embeddings_prod = []
for i in range(10):
    # Create random image [0, 255]
    img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    img_rgb = img.astype(np.float32)
    
    # ArcFace normalization: (pixel - 127.5) / 128.0
    img_normalized = (img_rgb - 127.5) / 128.0
    
    # Convert to CHW tensor
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        embedding = output.cpu().numpy().flatten()
        embeddings_prod.append(embedding)
    
    norm = np.linalg.norm(embedding)
    print(f"  Image {i+1}: norm={norm:.6f}")

# Check diversity
print("\nChecking diversity between production inputs...")
similarities_prod = []
for i in range(len(embeddings_prod)):
    for j in range(i + 1, len(embeddings_prod)):
        sim = np.dot(embeddings_prod[i], embeddings_prod[j])
        similarities_prod.append(sim)

mean_sim_prod = np.mean(similarities_prod)
std_sim_prod = np.std(similarities_prod)
min_sim_prod = np.min(similarities_prod)
max_sim_prod = np.max(similarities_prod)

print(f"  Pairs: {len(similarities_prod)}")
print(f"  Mean similarity:  {mean_sim_prod:.4f} ({mean_sim_prod*100:.2f}%)")
print(f"  Std:              {std_sim_prod:.4f}")
print(f"  Range:            [{min_sim_prod:.4f}, {max_sim_prod:.4f}]")

if mean_sim_prod > 0.9:
    print(f"  [ERROR] Model COLLAPSED - All embeddings too similar!")
elif mean_sim_prod > 0.7:
    print(f"  [WARN] Model may be collapsed - Mean similarity {mean_sim_prod:.4f} > 0.7")
else:
    print(f"  [OK] Model produces diverse embeddings with production preprocessing")

# ============================================================================
# 5. Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nOriginal MS1MV3 pretrained model:")
print(f"  Random inputs - Mean similarity: {mean_sim:.4f} ({mean_sim*100:.2f}%)")
print(f"  Production preprocessing - Mean similarity: {mean_sim_prod:.4f} ({mean_sim_prod*100:.2f}%)")

if mean_sim > 0.9 or mean_sim_prod > 0.9:
    print(f"\n[ERROR] ORIGINAL MS1MV3 MODEL COLLAPSED")
    print("  -> Original model outputs similar embeddings for all inputs")
    print("  -> Problem is in the original pretrained model")
elif mean_sim > 0.7 or mean_sim_prod > 0.7:
    print(f"\n[WARN] ORIGINAL MS1MV3 MODEL MAY BE COLLAPSED")
    print("  -> Original model outputs somewhat similar embeddings")
    print("  -> May have trouble distinguishing different people")
else:
    print(f"\n[OK] ORIGINAL MS1MV3 MODEL PRODUCES DIVERSE EMBEDDINGS")
    print("  -> Original model works correctly")
    print("  -> Problem must be in training code or fine-tuning process")

print("=" * 80)

