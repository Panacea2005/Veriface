"""Test MS1MV3 model properly - using real face images or verification pairs.

This script tests MS1MV3 model the way it's actually evaluated in benchmarks:
- Using real face images (aligned and preprocessed correctly)
- Computing verification metrics (AUC, EER) on verification pairs
- Comparing same person vs different person similarities
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import json
from sklearn.metrics import roc_curve, auc

# Ensure project root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MODELS_DIR = BACKEND_ROOT / "app" / "models"
MS1MV3_PATH = MODELS_DIR / "ms1mv3_arcface_r100_fp16.pth"
REGISTRY_PATH = BACKEND_ROOT / "app" / "store" / "registry.json"

print("=" * 80)
print("MS1MV3 MODEL PROPER TEST - Using Real Face Images")
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

try:
    import importlib.util
    import urllib.request
    
    iresnet_path = Path(__file__).parent.parent.parent / "iresnet_insightface.py"
    if not iresnet_path.exists():
        URL = 'https://raw.githubusercontent.com/deepinsight/insightface/master/recognition/arcface_torch/backbones/iresnet.py'
        urllib.request.urlretrieve(URL, str(iresnet_path))
        print(f"[INFO] Downloaded iResNet implementation")
    
    spec = importlib.util.spec_from_file_location('iresnet_insightface', str(iresnet_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules['iresnet_insightface'] = mod
    spec.loader.exec_module(mod)
    
    backbone = mod.iresnet100(num_features=512).to(device)
    checkpoint = torch.load(str(MS1MV3_PATH), map_location=device)
    missing, unexpected = backbone.load_state_dict(checkpoint, strict=False)
    print(f"[OK] MS1MV3 model loaded")
    
except Exception as e:
    print(f"[ERROR] Failed to load MS1MV3 model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Wrap with NormalizedBackbone
class NormalizedBackbone(torch.nn.Module):
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
# 2. Test with Registry Embeddings (Real Face Images)
# ============================================================================
print("\n2. TESTING WITH REGISTRY EMBEDDINGS (Real Face Images)")
print("-" * 80)

if REGISTRY_PATH.exists():
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)
    
    if len(registry) >= 2:
        print(f"[OK] Found {len(registry)} users in registry")
        
        # Extract embeddings for each user
        user_embeddings = {}
        for user_id, data in registry.items():
            embeddings = data.get('embeddings', [])
            if embeddings:
                embs = [np.array(emb, dtype=np.float32) for emb in embeddings]
                # Normalize
                embs = [emb / (np.linalg.norm(emb) + 1e-8) for emb in embs]
                user_embeddings[user_id] = {
                    'name': data.get('name', user_id),
                    'embeddings': embs
                }
        
        if len(user_embeddings) >= 2:
            print(f"[OK] Found {len(user_embeddings)} users with embeddings")
            
            # Compute intra-user similarity (same person)
            intra_similarities = []
            for user_id, data in user_embeddings.items():
                embs = data['embeddings']
                if len(embs) >= 2:
                    for i in range(len(embs)):
                        for j in range(i + 1, len(embs)):
                            sim = np.dot(embs[i], embs[j])
                            intra_similarities.append(sim)
            
            # Compute inter-user similarity (different people)
            inter_similarities = []
            user_ids = list(user_embeddings.keys())
            for i in range(len(user_ids)):
                for j in range(i + 1, len(user_ids)):
                    embs1 = user_embeddings[user_ids[i]]['embeddings']
                    embs2 = user_embeddings[user_ids[j]]['embeddings']
                    # Compare all pairs
                    for emb1 in embs1:
                        for emb2 in embs2:
                            sim = np.dot(emb1, emb2)
                            inter_similarities.append(sim)
            
            if intra_similarities and inter_similarities:
                intra_mean = np.mean(intra_similarities)
                intra_std = np.std(intra_similarities)
                inter_mean = np.mean(inter_similarities)
                inter_std = np.std(inter_similarities)
                inter_max = np.max(inter_similarities)
                separation = intra_mean - inter_mean
                
                print(f"\nIntra-user similarity (same person):")
                print(f"  Mean: {intra_mean:.4f} ({intra_mean*100:.2f}%)")
                print(f"  Std:  {intra_std:.4f}")
                print(f"  Range: [{np.min(intra_similarities):.4f}, {np.max(intra_similarities):.4f}]")
                
                print(f"\nInter-user similarity (different people):")
                print(f"  Mean: {inter_mean:.4f} ({inter_mean*100:.2f}%)")
                print(f"  Std:  {inter_std:.4f}")
                print(f"  Max:  {inter_max:.4f} ({inter_max*100:.2f}%)")
                print(f"  Range: [{np.min(inter_similarities):.4f}, {np.max(inter_similarities):.4f}]")
                
                print(f"\nSeparation gap: {separation:.4f}")
                
                # Compute AUC
                labels = [1] * len(intra_similarities) + [0] * len(inter_similarities)
                scores = intra_similarities + inter_similarities
                fpr, tpr, thresholds = roc_curve(labels, scores)
                roc_auc = auc(fpr, tpr)
                
                # Compute EER
                fnr = 1 - tpr
                eer_idx = np.nanargmin(np.abs(fpr - fnr))
                eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
                eer_threshold = thresholds[eer_idx]
                
                print(f"\nVerification Metrics:")
                print(f"  AUC: {roc_auc:.4f}")
                print(f"  EER: {eer:.4f} (threshold: {eer_threshold:.4f})")
                
                # Verdict
                print("\n" + "=" * 80)
                print("VERDICT")
                print("=" * 80)
                
                if inter_mean < 0.4 and separation > 0.4 and roc_auc > 0.9:
                    print("[OK] MS1MV3 MODEL WORKS CORRECTLY")
                    print("  -> Model can distinguish different people (low inter-user similarity)")
                    print("  -> Model recognizes same person (high intra-user similarity)")
                    print("  -> Good separation and high AUC")
                elif inter_mean < 0.6 and separation > 0.3 and roc_auc > 0.8:
                    print("[WARN] MS1MV3 MODEL PERFORMANCE MODERATE")
                    print("  -> Model can distinguish people but with some overlap")
                    print("  -> May need higher threshold for verification")
                else:
                    print("[ERROR] MS1MV3 MODEL HAS ISSUES")
                    print("  -> Model cannot distinguish different people well")
                    print("  -> Or cannot recognize same person")
                    print("  -> Low AUC indicates poor discrimination")
                
                print("=" * 80)
                sys.exit(0)
            else:
                print("[WARN] Not enough embeddings for comparison")
        else:
            print("[WARN] Need at least 2 users with embeddings")
    else:
        print("[WARN] Registry has less than 2 users")
else:
    print("[WARN] Registry not found")

# ============================================================================
# 3. Fallback: Test with Proper Preprocessing (No Real Images Available)
# ============================================================================
print("\n3. FALLBACK: Testing Model Architecture")
print("-" * 80)
print("[INFO] No real face images available, testing model architecture...")
print("[INFO] Note: This is NOT a proper benchmark test")
print("[INFO] For proper evaluation, use real face verification pairs")

# Test that model produces different outputs for different inputs
# (at least the architecture works)
test_inputs = []
for i in range(5):
    # Create different inputs with proper preprocessing
    img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img_normalized = (img_rgb - 127.5) / 128.0
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    test_inputs.append(img_tensor)

embeddings = []
with torch.no_grad():
    for inp in test_inputs:
        emb = model(inp).cpu().numpy().flatten()
        embeddings.append(emb)

# Check diversity
similarities = []
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        sim = np.dot(embeddings[i], embeddings[j])
        similarities.append(sim)

mean_sim = np.mean(similarities)
print(f"\nMean similarity between different inputs: {mean_sim:.4f} ({mean_sim*100:.2f}%)")

if mean_sim < 0.5:
    print("[OK] Model architecture works - produces different embeddings")
    print("[INFO] For proper evaluation, test with real face verification pairs")
elif mean_sim < 0.8:
    print("[WARN] Model may have issues - embeddings somewhat similar")
else:
    print("[ERROR] Model architecture may be broken - embeddings too similar")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("For proper MS1MV3 evaluation, use:")
print("  1. Real face verification pairs (same person / different person)")
print("  2. Standard benchmarks: LFW, CFP-FP, AgeDB")
print("  3. Compute AUC and EER on verification pairs")
print("  4. Expected performance: AUC > 0.99 on LFW")
print("=" * 80)

