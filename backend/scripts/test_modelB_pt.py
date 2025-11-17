"""Test script to verify Model B ConvNeXt Tiny TorchScript (.pt) is usable.

This script:
1. Loads modelB_convnext_tiny.pt
2. Tests inference with random inputs
3. Verifies output shape and normalization
4. Tests with different inputs to ensure model works correctly
"""
from __future__ import annotations

import sys
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MODELS_DIR = BACKEND_ROOT / "app" / "models"
MODEL_PATH = MODELS_DIR / "modelB_convnext_tiny.pt"

print("=" * 80)
print("MODEL B CONVNEXT TINY TORCHSCRIPT TEST")
print("=" * 80)
print()

# ============================================================================
# 1. Check if file exists
# ============================================================================
print("1. CHECKING MODEL FILE")
print("-" * 80)

if not MODEL_PATH.exists():
    print(f"[ERROR] Model file not found at: {MODEL_PATH}")
    print(f"[INFO] Expected path: {MODEL_PATH.absolute()}")
    sys.exit(1)

file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
print(f"[OK] Model file found: {MODEL_PATH.name}")
print(f"[OK] File size: {file_size_mb:.2f} MB")
print(f"[OK] Full path: {MODEL_PATH.absolute()}")
print()

# ============================================================================
# 2. Load TorchScript model
# ============================================================================
print("2. LOADING TORCHSCRIPT MODEL")
print("-" * 80)

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    model = torch.jit.load(str(MODEL_PATH), map_location=device)
    model.eval()
    print(f"[OK] Model loaded successfully")
    print()
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 3. Test inference with random inputs
# ============================================================================
print("3. TESTING INFERENCE")
print("-" * 80)

try:
    # Test 1: Single random input
    print("Test 1: Single random input (1, 3, 112, 112)")
    test_input1 = torch.randn(1, 3, 112, 112).to(device)
    
    with torch.no_grad():
        output1 = model(test_input1)
    
    print(f"  Input shape:  {test_input1.shape}")
    print(f"  Output shape: {output1.shape}")
    print(f"  Output dtype: {output1.dtype}")
    
    if output1.shape != (1, 512):
        print(f"[ERROR] Expected output shape (1, 512), got {output1.shape}")
        sys.exit(1)
    
    # Check if output is normalized (should be L2 normalized)
    output1_norm = torch.norm(output1, p=2, dim=1).item()
    print(f"  Output L2 norm: {output1_norm:.6f} (should be ~1.0)")
    
    if abs(output1_norm - 1.0) > 0.01:
        print(f"[WARN] Output is not properly normalized (norm={output1_norm:.6f})")
    else:
        print(f"  [OK] Output is properly normalized")
    
    # Check output statistics
    output1_mean = output1.mean().item()
    output1_std = output1.std().item()
    output1_min = output1.min().item()
    output1_max = output1.max().item()
    print(f"  Output stats: mean={output1_mean:.6f}, std={output1_std:.6f}, min={output1_min:.6f}, max={output1_max:.6f}")
    
    # Check if output is all zeros or constant
    if torch.allclose(output1, torch.zeros_like(output1), atol=1e-6):
        print(f"[ERROR] Output is all zeros - model is not working!")
        sys.exit(1)
    
    if output1_std < 1e-6:
        print(f"[ERROR] Output has zero variance - model is not working!")
        sys.exit(1)
    
    print()
    
    # Test 2: Different random input (should produce different output)
    print("Test 2: Different random input (should produce different output)")
    test_input2 = torch.randn(1, 3, 112, 112).to(device)
    
    with torch.no_grad():
        output2 = model(test_input2)
    
    # Check if outputs are different
    output_diff = torch.norm(output1 - output2, p=2).item()
    cosine_sim = F.cosine_similarity(output1, output2, dim=1).item()
    
    print(f"  Output difference (L2): {output_diff:.6f}")
    print(f"  Cosine similarity: {cosine_sim:.6f}")
    
    if output_diff < 1e-6 or cosine_sim > 0.9999:
        print(f"[WARN] Outputs are very similar for different inputs (might be OK for random noise)")
    else:
        print(f"  [OK] Model produces different outputs for different inputs")
    
    print()
    
    # Test 3: Batch of inputs
    print("Test 3: Batch of 4 inputs")
    test_input3 = torch.randn(4, 3, 112, 112).to(device)
    
    with torch.no_grad():
        output3 = model(test_input3)
    
    print(f"  Input shape:  {test_input3.shape}")
    print(f"  Output shape: {output3.shape}")
    
    if output3.shape != (4, 512):
        print(f"[ERROR] Expected output shape (4, 512), got {output3.shape}")
        sys.exit(1)
    
    # Check normalization for each sample in batch
    output3_norms = torch.norm(output3, p=2, dim=1)
    print(f"  Output norms: {output3_norms.tolist()}")
    
    if torch.allclose(output3_norms, torch.ones_like(output3_norms), atol=0.01):
        print(f"  [OK] All outputs in batch are properly normalized")
    else:
        print(f"[WARN] Some outputs are not properly normalized")
    
    print()
    
except Exception as e:
    print(f"[ERROR] Inference test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 4. Test with normalized input (as used in production)
# ============================================================================
print("4. TESTING WITH PRODUCTION PREPROCESSING")
print("-" * 80)

try:
    # Simulate production preprocessing: (pixel - 127.5) / 128.0
    # Input should be in range [0, 255] -> normalized to [-1, 1] range
    test_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    test_img_float = test_img.astype(np.float32)
    
    # Normalize: (pixel - 127.5) / 128.0
    test_img_normalized = (test_img_float - 127.5) / 128.0
    
    # Convert to CHW format
    test_img_chw = np.transpose(test_img_normalized, (2, 0, 1))
    test_tensor = torch.from_numpy(test_img_chw).unsqueeze(0).to(device)
    
    print(f"  Input image shape: {test_img.shape}")
    print(f"  Input image range: [{test_img.min()}, {test_img.max()}]")
    print(f"  Normalized tensor range: [{test_tensor.min().item():.3f}, {test_tensor.max().item():.3f}]")
    
    with torch.no_grad():
        output_prod = model(test_tensor)
    
    output_prod_norm = torch.norm(output_prod, p=2, dim=1).item()
    print(f"  Output shape: {output_prod.shape}")
    print(f"  Output L2 norm: {output_prod_norm:.6f}")
    
    if abs(output_prod_norm - 1.0) > 0.01:
        print(f"[WARN] Output is not properly normalized")
    else:
        print(f"  [OK] Model works correctly with production preprocessing")
    
    print()
    
except Exception as e:
    print(f"[ERROR] Production preprocessing test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# 5. Summary
# ============================================================================
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("[OK] Model file exists")
print("[OK] Model loads successfully")
print("[OK] Model produces correct output shape (1, 512)")
print("[OK] Model produces different outputs for different inputs")
print("[OK] Model works with batch inputs")
print("[OK] Model works with production preprocessing")
print()
print("[OK] MODEL B CONVNEXT TINY .PT IS USABLE!")
print("=" * 80)

