"""Test Model B .pth checkpoint - Check if it's using head/logits instead of backbone embeddings.

This script tests the .pth checkpoint to see if it's correctly using backbone embeddings (512-d)
instead of head logits (num_classes-d).
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
CHECKPOINT_PATH = MODELS_DIR / "modelB_best.pth"

print("=" * 80)
print("MODEL B .PTH CHECKPOINT ANALYSIS")
print("=" * 80)
print()

# ============================================================================
# 1. Load Checkpoint
# ============================================================================
print("1. LOADING CHECKPOINT")
print("-" * 80)

if not CHECKPOINT_PATH.exists():
    print(f"[ERROR] Checkpoint not found at: {CHECKPOINT_PATH}")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

try:
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)
    print(f"[OK] Checkpoint loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load checkpoint: {e}")
    sys.exit(1)

# ============================================================================
# 2. Analyze Checkpoint Structure
# ============================================================================
print("\n2. CHECKPOINT STRUCTURE")
print("-" * 80)

if isinstance(checkpoint, dict):
    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'backbone' in checkpoint:
        backbone_state = checkpoint['backbone']
        print(f"\n[OK] Found 'backbone' key")
        print(f"  Backbone state_dict keys: {len(backbone_state)} keys")
        print(f"  First 5 keys: {list(backbone_state.keys())[:5]}")
    else:
        print(f"\n[ERROR] No 'backbone' key found!")
        print(f"  Available keys: {list(checkpoint.keys())}")
    
    if 'head' in checkpoint:
        head_state = checkpoint['head']
        print(f"\n[OK] Found 'head' key")
        print(f"  Head state_dict keys: {len(head_state)} keys")
        print(f"  First 5 keys: {list(head_state.keys())[:5]}")
        
        # Check head output dimension
        if 'weight' in head_state:
            head_weight = head_state['weight']
            print(f"  Head weight shape: {head_weight.shape}")
            print(f"  Head output dim (num_classes): {head_weight.shape[0]}")
            print(f"  Head input dim (embedding_dim): {head_weight.shape[1]}")
    else:
        print(f"\n[WARN] No 'head' key found (may be OK if only backbone is needed)")
    
    if 'epoch' in checkpoint:
        print(f"\n[INFO] Epoch: {checkpoint['epoch']}")
    if 'best_auc' in checkpoint:
        print(f"[INFO] Best AUC: {checkpoint['best_auc']}")
else:
    print(f"[ERROR] Checkpoint is not a dict, it's a {type(checkpoint)}")
    sys.exit(1)

# ============================================================================
# 3. Check if Backend is Loading Correctly
# ============================================================================
print("\n3. BACKEND LOADING LOGIC")
print("-" * 80)

# Simulate backend loading logic
if 'backbone' in checkpoint:
    backbone_state = checkpoint['backbone']
    print(f"[INFO] Backend would load 'backbone' key")
    print(f"[INFO] Backbone has {len(backbone_state)} parameters")
    
    # Check if backbone has normalization layer
    has_normalization = False
    for key in backbone_state.keys():
        if 'normalize' in key.lower() or 'norm' in key.lower():
            has_normalization = True
            print(f"[INFO] Found normalization-related key: {key}")
    
    if not has_normalization:
        print(f"[WARN] No normalization layer found in backbone state_dict")
        print(f"[WARN] Backbone may not have L2 normalization in forward()")
else:
    print(f"[ERROR] No 'backbone' key - backend cannot load this checkpoint!")

# ============================================================================
# 4. Test with Dummy Model (if possible)
# ============================================================================
print("\n4. CHECKPOINT ANALYSIS SUMMARY")
print("-" * 80)

print(f"\nCheckpoint structure:")
print(f"  - Has 'backbone': {'Yes' if 'backbone' in checkpoint else 'No'}")
print(f"  - Has 'head': {'Yes' if 'head' in checkpoint else 'No'}")

if 'backbone' in checkpoint and 'head' in checkpoint:
    backbone_state = checkpoint['backbone']
    head_state = checkpoint['head']
    
    if 'weight' in head_state:
        num_classes = head_state['weight'].shape[0]
        embedding_dim = head_state['weight'].shape[1]
        
        print(f"\nHead dimensions:")
        print(f"  - Input (embedding_dim): {embedding_dim}")
        print(f"  - Output (num_classes): {num_classes}")
        
        print(f"\n[INFO] Backend should use backbone output ({embedding_dim}-d embeddings)")
        print(f"[INFO] Backend should NOT use head output ({num_classes}-d logits)")
        
        if embedding_dim == 512:
            print(f"[OK] Embedding dimension is 512 (correct)")
        else:
            print(f"[WARN] Embedding dimension is {embedding_dim} (expected 512)")
        
        if num_classes != 512:
            print(f"[OK] Head output dimension is {num_classes} (different from embeddings - correct)")
        else:
            print(f"[ERROR] Head output dimension is {num_classes} (same as embeddings - may be wrong!)")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if 'backbone' in checkpoint:
    print("[OK] Checkpoint has 'backbone' key - backend can load it")
    if 'head' in checkpoint:
        head_state = checkpoint['head']
        if 'weight' in head_state:
            embedding_dim = head_state['weight'].shape[1]
            num_classes = head_state['weight'].shape[0]
            if embedding_dim == 512 and num_classes != 512:
                print("[OK] Checkpoint structure looks correct")
                print("  -> Backbone outputs 512-d embeddings")
                print("  -> Head outputs num_classes-d logits")
                print("  -> Backend should use backbone, not head")
            else:
                print("[WARN] Checkpoint structure may be incorrect")
                print(f"  -> Embedding dim: {embedding_dim}, Num classes: {num_classes}")
else:
    print("[ERROR] Checkpoint does not have 'backbone' key!")
    print("  -> Backend cannot load this checkpoint correctly")

print("=" * 80)

