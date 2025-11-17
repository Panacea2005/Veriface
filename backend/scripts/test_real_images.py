"""
Test Model B with REAL registered user images
Compare pretrained vs trained model performance
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
import json
from PIL import Image
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("="*80)
print("REAL IMAGE TEST: Pretrained vs Trained Model B")
print("="*80)

# Load registry to get real user data
registry_path = "app/store/registry.json"
with open(registry_path, 'r') as f:
    registry = json.load(f)

print(f"\nRegistered users: {list(registry.keys())[:10]}")

# Check if we have SWS00001 and SWS00002
if 'SWS00001' not in registry or 'SWS00002' not in registry:
    print("❌ Need SWS00001 and SWS00002 for testing")
    print("Available users:", list(registry.keys())[:20])
    sys.exit(1)

print("\nTest users:")
print(f"  SWS00001: {registry['SWS00001']['name']} - {len(registry['SWS00001']['embeddings'])} embeddings")
print(f"  SWS00002: {registry['SWS00002']['name']} - {len(registry['SWS00002']['embeddings'])} embeddings")

# ============================================================================
# Load models
# ============================================================================
print("\n" + "="*80)
print("LOADING MODELS")
print("="*80)

from app.pipelines.arcface_model import get_model

# Model architecture
model = get_model(input_size=[112, 112], num_layers=100, mode='ir')
print(f"\nModel architecture: {type(model).__name__}")
print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

# Load pretrained weights
pretrained_path = "app/models/ms1mv3_arcface_r100_fp16.pth"
print(f"\n[1] Loading PRETRAINED: {pretrained_path}")
pretrained_ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
model.load_state_dict(pretrained_ckpt, strict=False)
model.eval()

# Test with dummy input
with torch.no_grad():
    dummy = torch.randn(1, 3, 112, 112)
    out = model(dummy)
    print(f"  Output shape: {out.shape}")
    print(f"  Output norm: {out.norm(dim=1).item():.6f} (should be ~1.0 for L2 normalized)")

model_pretrained = model

# Load trained Model B
trained_path = "app/models/modelB_best.pth"
print(f"\n[2] Loading TRAINED: {trained_path}")
model_trained = get_model(input_size=[112, 112], num_layers=100, mode='ir')
trained_ckpt = torch.load(trained_path, map_location='cpu', weights_only=False)

if 'backbone' in trained_ckpt:
    state_dict = trained_ckpt['backbone']
    print(f"  Using 'backbone' key from checkpoint")
    print(f"  Backbone params: {sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)):,}")
else:
    state_dict = trained_ckpt

# Load state dict
missing, unexpected = model_trained.load_state_dict(state_dict, strict=False)
print(f"  Missing keys: {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")
if unexpected:
    print(f"  Sample unexpected: {unexpected[:5]}")

model_trained.eval()

# Test output
with torch.no_grad():
    out = model_trained(dummy)
    print(f"  Output shape: {out.shape}")
    print(f"  Output norm: {out.norm(dim=1).item():.6f}")

# ============================================================================
# Extract stored embeddings
# ============================================================================
print("\n" + "="*80)
print("STORED EMBEDDINGS (from OLD preprocessing)")
print("="*80)

user1_embeddings = torch.tensor(registry['SWS00001']['embeddings'], dtype=torch.float32)
user2_embeddings = torch.tensor(registry['SWS00002']['embeddings'], dtype=torch.float32)

print(f"\nSWS00001: {user1_embeddings.shape}")
print(f"  Norm: {user1_embeddings.norm(dim=1).mean():.6f} ± {user1_embeddings.norm(dim=1).std():.6f}")

print(f"\nSWS00002: {user2_embeddings.shape}")
print(f"  Norm: {user2_embeddings.norm(dim=1).mean():.6f} ± {user2_embeddings.norm(dim=1).std():.6f}")

# Compute cross-similarity with stored embeddings
print("\n" + "="*80)
print("CROSS-SIMILARITY (OLD embeddings)")
print("="*80)

# SWS00001 vs SWS00001 (same person)
sim_11 = (user1_embeddings @ user1_embeddings.T).fill_diagonal_(0)
sim_11_scores = sim_11[sim_11 > 0]
print(f"\nSWS00001 vs SWS00001 (same person):")
print(f"  Mean: {sim_11_scores.mean():.4f}")
print(f"  Std:  {sim_11_scores.std():.4f}")
print(f"  Range: [{sim_11_scores.min():.4f}, {sim_11_scores.max():.4f}]")

# SWS00002 vs SWS00002 (same person)
sim_22 = (user2_embeddings @ user2_embeddings.T).fill_diagonal_(0)
sim_22_scores = sim_22[sim_22 > 0]
print(f"\nSWS00002 vs SWS00002 (same person):")
print(f"  Mean: {sim_22_scores.mean():.4f}")
print(f"  Std:  {sim_22_scores.std():.4f}")
print(f"  Range: [{sim_22_scores.min():.4f}, {sim_22_scores.max():.4f}]")

# SWS00001 vs SWS00002 (different people)
sim_12 = user1_embeddings @ user2_embeddings.T
print(f"\nSWS00001 vs SWS00002 (DIFFERENT people):")
print(f"  Mean: {sim_12.mean():.4f}")
print(f"  Std:  {sim_12.std():.4f}")
print(f"  Range: [{sim_12.min():.4f}, {sim_12.max():.4f}]")
print(f"  Max (worst case): {sim_12.max():.4f} <-- Should be < 0.4 threshold")

separation = sim_11_scores.mean() - sim_12.mean()
print(f"\nSeparation: {separation:.4f}")
if separation > 0.20:
    print("  ✅ GOOD discrimination")
elif separation > 0.10:
    print("  ⚠️  MODERATE discrimination (may cause false positives)")
else:
    print("  ❌ POOR discrimination (cannot distinguish users)")

# ============================================================================
# Diagnosis
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if separation < 0.15:
    print("\n❌ CRITICAL: Stored embeddings show poor discrimination")
    print("\nPOSSIBLE CAUSES:")
    print("  1. OLD preprocessing (/127.5) was used when registering")
    print("  2. Need to DELETE registry and RE-REGISTER both users")
    print("  3. New preprocessing (/128.0) will create better embeddings")
    
    print("\n🔧 FIX:")
    print("  cd \"d:\\Swinburne\\COS30082 - Applied Machine Learning\\Project\\Veriface\\backend\\app\\store\"")
    print("  Copy-Item registry.json registry.json.backup")
    print("  '{}'| Out-File -FilePath registry.json -Encoding utf8")
    print("  # Then restart backend and re-register SWS00001 and SWS00002")
else:
    print("\n✅ Stored embeddings show good discrimination")
    print("   Issue may be in query-time preprocessing or threshold setting")

print("\n" + "="*80)
