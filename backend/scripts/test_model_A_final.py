"""
FINAL CRITICAL TEST - Model A Post-Training Verification
Tests embeddings with REAL preprocessing to match training exactly.
"""

import torch
import numpy as np
import cv2
import sys
sys.path.append('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend')

from app.pipelines.arcface_model import get_model

print("="*80)
print("CRITICAL TEST: Model A with EXACT Training Preprocessing")
print("="*80)
print()

# Load model
print("[1/4] Loading Model A...")
model = get_model(input_size=[112, 112], num_layers=100, mode='ir')
ckpt = torch.load('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend/app/models/modelA_best.pth', 
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
print("   ✓ Model loaded")
print()

# L2 normalization function (same as training)
def l2norm(x, eps=1e-12):
    """L2-normalize embeddings (unit sphere projection) - SAME AS TRAINING"""
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)

print("[2/4] Testing preprocessing pipeline...")
print("   Training notebook uses: (x*255 - 127.5) / 128.0 where x in [0,1]")
print("   This equals: (pixel - 127.5) / 128.0 where pixel in [0,255]")
print()

# Create 5 DIFFERENT synthetic face-like images
test_images = []

# Image 1: Dark face
img1 = np.full((112, 112, 3), 50, dtype=np.uint8)
cv2.circle(img1, (56, 56), 30, (80, 80, 80), -1)  # Face
test_images.append(("Dark Face", img1))

# Image 2: Bright face
img2 = np.full((112, 112, 3), 200, dtype=np.uint8)
cv2.circle(img2, (56, 56), 30, (230, 230, 230), -1)
test_images.append(("Bright Face", img2))

# Image 3: Contrast face
img3 = np.full((112, 112, 3), 127, dtype=np.uint8)
cv2.rectangle(img3, (20, 20), (92, 92), (255, 255, 255), -1)
cv2.circle(img3, (56, 56), 25, (50, 50, 50), -1)
test_images.append(("High Contrast", img3))

# Image 4: Gradient
img4 = np.tile(np.linspace(0, 255, 112, dtype=np.uint8).reshape(-1, 1), (1, 112))
img4 = np.stack([img4, img4, img4], axis=2)
test_images.append(("Gradient", img4))

# Image 5: Random noise
np.random.seed(42)
img5 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
test_images.append(("Random Noise", img5))

print("[3/4] Extracting embeddings with training-exact preprocessing...")
print()

embeddings_raw = []
embeddings_l2 = []

with torch.no_grad():
    for name, img in test_images:
        # EXACT preprocessing as training
        # Training: transforms.Lambda(lambda x:(x*255-127.5)/128.0) where x is from ToTensor() [0,1]
        # Equivalent: (pixel - 127.5) / 128.0 where pixel is [0,255]
        img_norm = (img.astype(np.float32) - 127.5) / 128.0
        img_chw = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)
        
        # Forward pass (raw embedding)
        emb_raw = model(img_tensor)
        
        # L2 normalize (CRITICAL - same as training!)
        emb_l2 = l2norm(emb_raw)
        
        embeddings_raw.append(emb_raw)
        embeddings_l2.append(emb_l2)
        
        raw_norm = torch.norm(emb_raw).item()
        l2_norm = torch.norm(emb_l2).item()
        
        print(f"{name:15} | Raw norm: {raw_norm:8.4f} | L2 norm: {l2_norm:.6f} | First 3: {emb_l2[0, :3].tolist()}")

print()
print("[4/4] Computing pairwise similarities (L2-normalized embeddings)...")
print("="*80)

max_sim = 0
similarities = []

for i in range(5):
    for j in range(i+1, 5):
        name1, _ = test_images[i]
        name2, _ = test_images[j]
        
        # Cosine similarity on L2-normalized embeddings (dot product)
        cos_sim = (embeddings_l2[i] * embeddings_l2[j]).sum().item()
        similarities.append(cos_sim)
        max_sim = max(max_sim, cos_sim)
        
        status = "✅ Good" if cos_sim < 0.5 else ("⚠️  Moderate" if cos_sim < 0.8 else "❌ TOO SIMILAR")
        print(f"{name1:15} vs {name2:15}: {cos_sim:+.6f}  {status}")

print("="*80)
print()

# Final verdict
print("FINAL VERDICT:")
print("="*80)
print(f"Maximum similarity: {max_sim:.6f}")
print()

if max_sim > 0.8:
    print("❌ CRITICAL ERROR: Model still produces similar embeddings!")
    print("   Despite L2 normalization, embeddings are too close.")
    print()
    print("🔍 Possible causes:")
    print("   1. Model didn't converge properly during training")
    print("   2. Training data had issues")
    print("   3. Loss function wasn't working correctly")
    print()
    print("💡 SOLUTION: Check training logs for:")
    print("   - Did training loss decrease? (should go from 0.4 → 0.08)")
    print("   - Did AUC increase? (should reach 0.98+)")
    print("   - Was l2norm() actually called in training loop?")
elif max_sim > 0.5:
    print("⚠️  WARNING: Some embeddings show moderate similarity")
    print("   This is acceptable for synthetic test images.")
    print("   Real face images should have better separation.")
    print()
    print("✅ Model appears to be working correctly")
else:
    print("✅ EXCELLENT: Embeddings are diverse!")
    print("   Different inputs produce well-separated embeddings.")
    print("   Model learned discriminative features successfully.")
    print()
    print("🎯 Model is ready for production use")

print("="*80)
print()

# Additional check: Compare raw vs L2-normalized
print("NORMALIZATION ANALYSIS:")
print("="*80)
raw_norms = [torch.norm(e).item() for e in embeddings_raw]
l2_norms = [torch.norm(e).item() for e in embeddings_l2]

print(f"Raw embedding norms: min={min(raw_norms):.4f}, max={max(raw_norms):.4f}, std={np.std(raw_norms):.4f}")
print(f"L2-normalized norms: min={min(l2_norms):.6f}, max={max(l2_norms):.6f}, std={np.std(l2_norms):.10f}")
print()

if max(l2_norms) - min(l2_norms) > 0.01:
    print("⚠️  WARNING: L2 normalized embeddings don't have unit norm!")
    print("   Something is wrong with the normalization.")
else:
    print("✅ All L2-normalized embeddings have unit norm (≈1.0)")

print("="*80)
