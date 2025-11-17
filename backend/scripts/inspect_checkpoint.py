"""
Deep inspection of Model B checkpoint structure
"""

import torch
import sys

print("="*80)
print("MODEL B CHECKPOINT DEEP INSPECTION")
print("="*80)

path = "app/models/modelB_best.pth"
print(f"\nLoading: {path}")

ckpt = torch.load(path, map_location='cpu', weights_only=False)

print(f"\n[1] Top-level keys: {list(ckpt.keys())}")

for key, value in ckpt.items():
    print(f"\n[2] Key '{key}':")
    print(f"    Type: {type(value)}")
    
    if isinstance(value, dict):
        print(f"    Dict keys: {list(value.keys())[:10]}")
        if len(value) > 10:
            print(f"    ... and {len(value)-10} more keys")
        
        # Check if it's a state dict
        first_val = next(iter(value.values())) if value else None
        if first_val is not None:
            print(f"    First value type: {type(first_val)}")
            if isinstance(first_val, torch.Tensor):
                print(f"    First value shape: {first_val.shape}")
    
    elif isinstance(value, torch.Tensor):
        print(f"    Tensor shape: {value.shape}")
    
    elif isinstance(value, (int, float, str)):
        print(f"    Value: {value}")
    
    else:
        print(f"    Unknown type: {type(value)}")

# Try to extract actual model weights
print("\n" + "="*80)
print("EXTRACTING MODEL WEIGHTS")
print("="*80)

if 'backbone' in ckpt:
    backbone = ckpt['backbone']
    print(f"\n'backbone' type: {type(backbone)}")
    
    if isinstance(backbone, dict):
        print(f"Backbone is a state_dict with {len(backbone)} keys")
        print(f"Sample keys: {list(backbone.keys())[:5]}")
        
        # Count parameters
        total_params = sum(p.numel() for p in backbone.values() if isinstance(p, torch.Tensor))
        print(f"Total parameters: {total_params:,}")
    else:
        print(f"❌ Backbone is NOT a state_dict! It's: {type(backbone)}")

if 'head' in ckpt:
    head = ckpt['head']
    print(f"\n'head' type: {type(head)}")
    
    if isinstance(head, dict):
        print(f"Head is a state_dict with {len(head)} keys")
        print(f"Keys: {list(head.keys())}")
        
        for k, v in head.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
        
        total_params = sum(p.numel() for p in head.values() if isinstance(p, torch.Tensor))
        print(f"Total parameters: {total_params:,}")
    else:
        print(f"❌ Head is NOT a state_dict! It's: {type(head)}")

print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

issues = []

# Check backbone structure
if 'backbone' not in ckpt:
    issues.append("❌ No 'backbone' key in checkpoint")
elif not isinstance(ckpt['backbone'], dict):
    issues.append("❌ 'backbone' is not a state_dict")
elif len(ckpt['backbone']) == 0:
    issues.append("❌ 'backbone' state_dict is empty")
else:
    # Check for NormalizedBackbone wrapper
    has_wrapper = any('backbone.' in k for k in ckpt['backbone'].keys())
    if has_wrapper:
        print("✅ Backbone has NormalizedBackbone wrapper")
    else:
        issues.append("⚠️  Backbone missing NormalizedBackbone wrapper prefix")

# Check head structure  
if 'head' in ckpt:
    if isinstance(ckpt['head'], dict) and len(ckpt['head']) > 0:
        print("✅ Sub-Center ArcFace head present")
        # This is expected during training but should be removed for deployment
        issues.append("⚠️  Classification head should be REMOVED for deployment (use backbone only)")
    else:
        issues.append("❌ 'head' key exists but is not a valid state_dict")

if issues:
    print("\n🔧 ISSUES:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✅ Checkpoint structure looks good!")

print("\n" + "="*80)
