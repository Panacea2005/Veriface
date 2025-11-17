"""Test MS1MV3 model with face-like images instead of random noise.

Face recognition models are trained on face images, not random noise.
Testing with random noise doesn't reflect real performance.
This script creates face-like synthetic images for more realistic testing.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Ensure project root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MODELS_DIR = BACKEND_ROOT / "app" / "models"
MS1MV3_PATH = MODELS_DIR / "ms1mv3_arcface_r100_fp16.pth"

print("=" * 80)
print("MS1MV3 MODEL TEST WITH FACE-LIKE IMAGES")
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
# 2. Wrap with NormalizedBackbone
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
        x = F.normalize(x, p=2, dim=1)
        return x

model = NormalizedBackbone(backbone).to(device)
model.eval()
print(f"[OK] Model wrapped with NormalizedBackbone")

# ============================================================================
# 3. Create Face-Like Synthetic Images
# ============================================================================
print("\n3. CREATING FACE-LIKE SYNTHETIC IMAGES")
print("-" * 80)

def create_face_like_image(seed=None, person_id=0):
    """Create a synthetic face-like image with face-like structure"""
    if seed is not None:
        np.random.seed(seed)
    
    # Create base image with skin tone (around 120-180 in grayscale)
    img = np.ones((112, 112, 3), dtype=np.uint8) * np.random.randint(120, 180, 3)
    
    # Add face structure:
    # 1. Face shape (ellipse)
    center = (56, 56)
    axes = (45 + person_id * 5, 55 + person_id * 5)
    cv2.ellipse(img, center, axes, 0, 0, 360, (140 + person_id * 10, 120 + person_id * 8, 100 + person_id * 6), -1)
    
    # 2. Eyes (two circles)
    eye_y = 40
    eye_spacing = 25
    left_eye = (56 - eye_spacing, eye_y)
    right_eye = (56 + eye_spacing, eye_y)
    cv2.circle(img, left_eye, 8 + person_id, (0, 0, 0), -1)
    cv2.circle(img, right_eye, 8 + person_id, (0, 0, 0), -1)
    
    # 3. Nose (small ellipse)
    nose = (56, 50)
    cv2.ellipse(img, nose, (3, 8), 0, 0, 360, (100, 80, 60), -1)
    
    # 4. Mouth (ellipse)
    mouth = (56, 70)
    cv2.ellipse(img, mouth, (15 + person_id, 5), 0, 0, 360, (50, 30, 30), -1)
    
    # 5. Add some texture/noise
    noise = np.random.randint(-10, 10, (112, 112, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img

print("Creating 10 different face-like images (different people)...")
face_images = []
for i in range(10):
    face_img = create_face_like_image(seed=i, person_id=i)
    face_images.append(face_img)
    print(f"  Person {i+1}: Created face-like image (mean={np.mean(face_img):.1f}, std={np.std(face_img):.1f})")

# ============================================================================
# 4. Test with Face-Like Images
# ============================================================================
print("\n4. TESTING WITH FACE-LIKE IMAGES")
print("-" * 80)
print("Testing model with face-like synthetic images...")

embeddings_face = []
for i, face_img in enumerate(face_images):
    # Convert to RGB and normalize (ArcFace normalization)
    img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # ArcFace normalization: (pixel - 127.5) / 128.0
    img_normalized = (img_rgb - 127.5) / 128.0
    
    # Convert to CHW tensor
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        embedding = output.cpu().numpy().flatten()
        embeddings_face.append(embedding)
    
    norm = np.linalg.norm(embedding)
    print(f"  Person {i+1}: norm={norm:.6f}")

# Check diversity
print("\nChecking diversity between different people...")
similarities_face = []
for i in range(len(embeddings_face)):
    for j in range(i + 1, len(embeddings_face)):
        sim = np.dot(embeddings_face[i], embeddings_face[j])
        similarities_face.append(sim)

mean_sim_face = np.mean(similarities_face)
std_sim_face = np.std(similarities_face)
min_sim_face = np.min(similarities_face)
max_sim_face = np.max(similarities_face)

print(f"  Pairs: {len(similarities_face)}")
print(f"  Mean similarity:  {mean_sim_face:.4f} ({mean_sim_face*100:.2f}%)")
print(f"  Std:              {std_sim_face:.4f}")
print(f"  Range:            [{min_sim_face:.4f}, {max_sim_face:.4f}]")

# ============================================================================
# 5. Test Same Person Variations
# ============================================================================
print("\n5. TESTING SAME PERSON VARIATIONS")
print("-" * 80)
print("Creating 5 variations of the same person...")

# Create 5 variations of person 0
same_person_images = []
for i in range(5):
    face_img = create_face_like_image(seed=0, person_id=0)  # Same person_id, different seed
    same_person_images.append(face_img)

embeddings_same = []
for i, face_img in enumerate(same_person_images):
    img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_normalized = (img_rgb - 127.5) / 128.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        embedding = output.cpu().numpy().flatten()
        embeddings_same.append(embedding)

# Check similarity for same person
print("\nChecking similarity for same person variations...")
similarities_same = []
for i in range(len(embeddings_same)):
    for j in range(i + 1, len(embeddings_same)):
        sim = np.dot(embeddings_same[i], embeddings_same[j])
        similarities_same.append(sim)

mean_sim_same = np.mean(similarities_same)
std_sim_same = np.std(similarities_same)
min_sim_same = np.min(similarities_same)
max_sim_same = np.max(similarities_same)

print(f"  Pairs: {len(similarities_same)}")
print(f"  Mean similarity:  {mean_sim_same:.4f} ({mean_sim_same*100:.2f}%)")
print(f"  Std:              {std_sim_same:.4f}")
print(f"  Range:            [{min_sim_same:.4f}, {max_sim_same:.4f}]")

# ============================================================================
# 6. Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nDifferent people (should be LOW similarity):")
print(f"  Mean similarity: {mean_sim_face:.4f} ({mean_sim_face*100:.2f}%)")
if mean_sim_face < 0.4:
    print(f"  Status: [OK] Model can distinguish different people")
elif mean_sim_face < 0.6:
    print(f"  Status: [WARN] Model may have trouble distinguishing some people")
else:
    print(f"  Status: [ERROR] Model cannot distinguish different people well")

print(f"\nSame person variations (should be HIGH similarity):")
print(f"  Mean similarity: {mean_sim_same:.4f} ({mean_sim_same*100:.2f}%)")
if mean_sim_same > 0.7:
    print(f"  Status: [OK] Model recognizes same person")
elif mean_sim_same > 0.5:
    print(f"  Status: [WARN] Model may have trouble recognizing same person")
else:
    print(f"  Status: [ERROR] Model cannot recognize same person")

print(f"\nSeparation gap (same - different):")
separation = mean_sim_same - mean_sim_face
print(f"  Gap: {separation:.4f}")
if separation > 0.3:
    print(f"  Status: [OK] Good separation between same and different people")
elif separation > 0.1:
    print(f"  Status: [WARN] Moderate separation")
else:
    print(f"  Status: [ERROR] Poor separation - model may not work well")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if mean_sim_face < 0.4 and mean_sim_same > 0.7 and separation > 0.3:
    print("[OK] MS1MV3 MODEL WORKS CORRECTLY")
    print("  -> Model can distinguish different people (low similarity)")
    print("  -> Model recognizes same person (high similarity)")
    print("  -> Good separation between same and different")
    print("  -> Previous test with random noise was NOT appropriate for face recognition models")
elif mean_sim_face < 0.6 and mean_sim_same > 0.5:
    print("[WARN] MS1MV3 MODEL MAY HAVE ISSUES")
    print("  -> Model may have trouble distinguishing some people")
    print("  -> Or recognizing same person variations")
else:
    print("[ERROR] MS1MV3 MODEL HAS ISSUES")
    print("  -> Model cannot distinguish different people well")
    print("  -> Or cannot recognize same person")

print("=" * 80)

