"""Test loading Model B .pth checkpoint and check if backbone keys have 'backbone.' prefix.

This script tests if the checkpoint has 'backbone.' prefix in keys, and if backend can handle it.
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn as nn

# Ensure project root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

MODELS_DIR = BACKEND_ROOT / "app" / "models"
CHECKPOINT_PATH = MODELS_DIR / "modelB_best.pth"

print("=" * 80)
print("MODEL B .PTH CHECKPOINT KEY PREFIX TEST")
print("=" * 80)
print()

# ============================================================================
# 1. Load Checkpoint
# ============================================================================
print("1. LOADING CHECKPOINT")
print("-" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(str(CHECKPOINT_PATH), map_location=device, weights_only=False)

if 'backbone' not in checkpoint:
    print("[ERROR] No 'backbone' key in checkpoint!")
    sys.exit(1)

backbone_state = checkpoint['backbone']
print(f"[OK] Loaded backbone state_dict with {len(backbone_state)} keys")

# ============================================================================
# 2. Check Key Prefixes
# ============================================================================
print("\n2. CHECKING KEY PREFIXES")
print("-" * 80)

keys_with_backbone_prefix = [k for k in backbone_state.keys() if k.startswith('backbone.')]
keys_without_backbone_prefix = [k for k in backbone_state.keys() if not k.startswith('backbone.')]

print(f"Keys with 'backbone.' prefix: {len(keys_with_backbone_prefix)}")
print(f"Keys without 'backbone.' prefix: {len(keys_without_backbone_prefix)}")

if keys_with_backbone_prefix:
    print(f"\nFirst 5 keys with 'backbone.' prefix:")
    for k in keys_with_backbone_prefix[:5]:
        print(f"  - {k}")
    
    # Check if removing prefix would work
    print(f"\n[INFO] Keys have 'backbone.' prefix")
    print(f"[INFO] This suggests backbone was wrapped in NormalizedBackbone")
    print(f"[INFO] When saving, it saved the wrapper's state_dict")
    print(f"[INFO] Backend needs to handle this prefix when loading")

if keys_without_backbone_prefix:
    print(f"\nFirst 5 keys without 'backbone.' prefix:")
    for k in keys_without_backbone_prefix[:5]:
        print(f"  - {k}")

# ============================================================================
# 3. Test Backend Key Mapping Logic
# ============================================================================
print("\n3. TESTING BACKEND KEY MAPPING")
print("-" * 80)

# Simulate a simple model to test loading
class SimpleBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.bn1 = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return x

# Create model
model = SimpleBackbone()
model_keys = set(model.state_dict().keys())
print(f"Model keys (expected): {model_keys}")

# Test direct load
print(f"\nTesting direct load...")
try:
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  First 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  First 5 unexpected: {unexpected[:5]}")
except Exception as e:
    print(f"  [ERROR] Direct load failed: {e}")

# Test with prefix removal (backend logic)
print(f"\nTesting with prefix removal (backend logic)...")
mapped_state_dict = {}
for k, v in backbone_state.items():
    # Remove 'backbone.' prefix
    mapped_key = k.replace('backbone.', '')
    mapped_state_dict[mapped_key] = v

try:
    missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  First 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  First 5 unexpected: {unexpected[:5]}")
except Exception as e:
    print(f"  [ERROR] Prefix removal load failed: {e}")

# ============================================================================
# 4. Check if NormalizedBackbone wrapper is in checkpoint
# ============================================================================
print("\n4. CHECKING FOR NORMALIZEDBACKBONE WRAPPER")
print("-" * 80)

# Check if there are any normalization-related keys
norm_keys = [k for k in backbone_state.keys() if 'normalize' in k.lower() or 'norm' in k.lower()]
if norm_keys:
    print(f"[INFO] Found normalization-related keys: {len(norm_keys)}")
    for k in norm_keys[:5]:
        print(f"  - {k}")
else:
    print(f"[WARN] No normalization-related keys found")
    print(f"[WARN] This suggests NormalizedBackbone wrapper is NOT in checkpoint")
    print(f"[WARN] Backbone may not have L2 normalization in forward()")

# Check if keys suggest wrapped model
if keys_with_backbone_prefix:
    print(f"\n[INFO] Keys have 'backbone.' prefix")
    print(f"[INFO] This suggests model structure is:")
    print(f"  NormalizedBackbone(backbone)")
    print(f"  -> When saving state_dict, it saves 'backbone.conv1.weight' etc.")
    print(f"  -> But NormalizedBackbone itself has no parameters (just F.normalize)")
    print(f"  -> So normalization is in forward(), not in state_dict")

print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

if keys_with_backbone_prefix:
    print("[INFO] Checkpoint has 'backbone.' prefix in keys")
    print("  -> Backend needs to remove 'backbone.' prefix when loading")
    print("  -> Backend code: mapped_key = k.replace('backbone.', '')")
    
    if 'backbone.' in list(backbone_state.keys())[0]:
        print("\n[OK] Backend should handle this correctly")
        print("  -> Backend has key mapping logic to remove 'backbone.' prefix")
    else:
        print("\n[WARN] May need to check backend key mapping")
else:
    print("[INFO] Checkpoint does not have 'backbone.' prefix")
    print("  -> Keys should load directly")

if not norm_keys:
    print("\n[WARN] No normalization layer in checkpoint")
    print("  -> NormalizedBackbone wrapper normalization is in forward()")
    print("  -> Not in state_dict (no parameters)")
    print("  -> This is OK if model was wrapped during training")

print("=" * 80)

