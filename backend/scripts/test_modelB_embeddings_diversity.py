"""Test Model B ConvNeXt embeddings diversity - Check if model produces different embeddings for different inputs.

This script tests if the model is collapsed (all embeddings similar) or working correctly.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import cv2
import torch

# Ensure project root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MODELS_DIR = BACKEND_ROOT / "app" / "models"
MODEL_PATH = MODELS_DIR / "modelB_convnext_tiny.pt"

print("=" * 80)
print("MODEL B CONVNEXT EMBEDDINGS DIVERSITY TEST")
print("=" * 80)
print()

# ============================================================================
# 1. Load Model
# ============================================================================
print("1. LOADING MODEL")
print("-" * 80)

if not MODEL_PATH.exists():
    print(f"[ERROR] Model not found at: {MODEL_PATH}")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

try:
    model = torch.jit.load(str(MODEL_PATH), map_location=device)
    model.eval()
    print(f"[OK] Model loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# ============================================================================
# 2. Test with Random Inputs
# ============================================================================
print("\n2. TESTING WITH RANDOM INPUTS")
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
    print(f"          Mean similarity {mean_sim:.4f} > 0.9 indicates model collapse")
elif mean_sim > 0.7:
    print(f"  [WARN] Model may be collapsed - Mean similarity {mean_sim:.4f} > 0.7")
else:
    print(f"  [OK] Model produces diverse embeddings for random inputs")

# ============================================================================
# 3. Test with Production Preprocessing
# ============================================================================
print("\n3. TESTING WITH PRODUCTION PREPROCESSING")
print("-" * 80)
print("Creating 10 different images with ImageNet normalization...")

embeddings_prod = []
for i in range(10):
    # Create random image [0, 255]
    img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    img_rgb = img.astype(np.float32)
    
    # ImageNet normalization (Model B ConvNeXt)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img_normalized = (img_rgb / 255.0 - mean) / std
    
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
    print(f"          Mean similarity {mean_sim_prod:.4f} > 0.9 indicates model collapse")
elif mean_sim_prod > 0.7:
    print(f"  [WARN] Model may be collapsed - Mean similarity {mean_sim_prod:.4f} > 0.7")
else:
    print(f"  [OK] Model produces diverse embeddings with production preprocessing")

# ============================================================================
# 4. Test with Same Input (Should be Identical)
# ============================================================================
print("\n4. TESTING WITH SAME INPUT (Should be Identical)")
print("-" * 80)

test_input = torch.randn(1, 3, 112, 112).to(device)

with torch.no_grad():
    output1 = model(test_input)
    output2 = model(test_input)
    
    embedding1 = output1.cpu().numpy().flatten()
    embedding2 = output2.cpu().numpy().flatten()
    
    diff = np.linalg.norm(embedding1 - embedding2)
    sim = np.dot(embedding1, embedding2)

print(f"  Same input, run 1 vs run 2:")
print(f"  Difference (L2):  {diff:.8f} (should be ~0.0)")
print(f"  Similarity:       {sim:.8f} (should be ~1.0)")

if diff < 1e-6 and sim > 0.9999:
    print(f"  [OK] Model is deterministic (same input -> same output)")
else:
    print(f"  [WARN] Model may not be deterministic")

# ============================================================================
# 5. Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nRandom inputs diversity:")
print(f"  Mean similarity: {mean_sim:.4f} ({mean_sim*100:.2f}%)")
if mean_sim > 0.9:
    print(f"  Status: [ERROR] Model COLLAPSED")
elif mean_sim > 0.7:
    print(f"  Status: [WARN] Model may be collapsed")
else:
    print(f"  Status: [OK] Model produces diverse embeddings")

print(f"\nProduction preprocessing diversity:")
print(f"  Mean similarity: {mean_sim_prod:.4f} ({mean_sim_prod*100:.2f}%)")
if mean_sim_prod > 0.9:
    print(f"  Status: [ERROR] Model COLLAPSED")
elif mean_sim_prod > 0.7:
    print(f"  Status: [WARN] Model may be collapsed")
else:
    print(f"  Status: [OK] Model produces diverse embeddings")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if mean_sim > 0.9 or mean_sim_prod > 0.9:
    print("[ERROR] MODEL COLLAPSED")
    print("  -> Model outputs similar embeddings for all inputs")
    print("  -> Cannot distinguish different people")
    print("  -> Need to retrain model or check preprocessing")
elif mean_sim > 0.7 or mean_sim_prod > 0.7:
    print("[WARN] MODEL MAY BE COLLAPSED")
    print("  -> Model outputs somewhat similar embeddings")
    print("  -> May have trouble distinguishing different people")
    print("  -> Need to investigate further")
else:
    print("[OK] MODEL PRODUCES DIVERSE EMBEDDINGS")
    print("  -> Model outputs different embeddings for different inputs")
    print("  -> Should be able to distinguish different people")
    print("  -> If discrimination still poor, check preprocessing or dataset")

print("=" * 80)

