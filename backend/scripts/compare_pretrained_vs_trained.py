"""
Compare pretrained ArcFace model vs trained Model B
Analyze why training may degrade performance
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("="*80)
print("PRETRAINED vs TRAINED MODEL COMPARISON")
print("="*80)

# ============================================================================
# 1. Load both models
# ============================================================================
print("\n[1] Loading models...")

pretrained_path = "app/models/ms1mv3_arcface_r100_fp16.pth"
trained_path = "app/models/modelB_best.pth"

# Load pretrained model
print(f"\nPretrained: {pretrained_path}")
pretrained = torch.load(pretrained_path, map_location='cpu', weights_only=False)
print(f"  Keys: {list(pretrained.keys())[:5]}...")
print(f"  Total params: {sum(p.numel() for p in pretrained.values() if isinstance(p, torch.Tensor)):,}")

# Load trained model
print(f"\nTrained: {trained_path}")
trained_ckpt = torch.load(trained_path, map_location='cpu', weights_only=False)
print(f"  Checkpoint keys: {list(trained_ckpt.keys())}")

if 'model' in trained_ckpt:
    trained = trained_ckpt['model']
else:
    trained = trained_ckpt

print(f"  Total params: {sum(p.numel() for p in trained.values() if isinstance(p, torch.Tensor)):,}")

# ============================================================================
# 2. Compare architectures
# ============================================================================
print("\n[2] Architecture comparison...")

pretrained_keys = set(pretrained.keys())
trained_keys = set(trained.keys())

print(f"\nPretrained layers: {len(pretrained_keys)}")
print(f"Trained layers: {len(trained_keys)}")

# Check for wrapper
has_wrapper = any('backbone.' in k for k in trained_keys)
print(f"\nTrained model has NormalizedBackbone wrapper: {has_wrapper}")

# Keys only in pretrained
only_pretrained = pretrained_keys - trained_keys
if has_wrapper:
    # Remove 'backbone.' prefix from trained keys for comparison
    trained_unwrapped = {k.replace('backbone.', ''): k for k in trained_keys if 'backbone.' in k}
    only_pretrained = pretrained_keys - set(trained_unwrapped.keys())

if only_pretrained:
    print(f"\nLayers ONLY in pretrained ({len(only_pretrained)}):")
    for k in sorted(list(only_pretrained))[:10]:
        print(f"  - {k}")
    if len(only_pretrained) > 10:
        print(f"  ... and {len(only_pretrained)-10} more")

# Keys only in trained
only_trained = trained_keys - pretrained_keys
if has_wrapper:
    only_trained = {k for k in trained_keys if not k.startswith('backbone.')}

if only_trained:
    print(f"\nLayers ONLY in trained ({len(only_trained)}):")
    for k in sorted(list(only_trained))[:10]:
        print(f"  - {k}")

# ============================================================================
# 3. Compare weight statistics
# ============================================================================
print("\n[3] Weight statistics comparison...")

def get_weight_stats(model_dict, prefix=""):
    """Get statistics for all weight tensors"""
    stats = {}
    for k, v in model_dict.items():
        if isinstance(v, torch.Tensor) and len(v.shape) > 0:
            key = k.replace(prefix, '') if prefix else k
            stats[key] = {
                'mean': v.float().mean().item(),
                'std': v.float().std().item(),
                'min': v.float().min().item(),
                'max': v.float().max().item(),
                'norm': v.float().norm().item()
            }
    return stats

pretrained_stats = get_weight_stats(pretrained)
trained_stats = get_weight_stats(trained, prefix='backbone.')

# Find common layers
common_layers = set(pretrained_stats.keys()) & set(trained_stats.keys())
print(f"\nCommon layers: {len(common_layers)}")

# Compare first few layers (should be similar if frozen/pretrained)
print("\nFirst 5 conv/bn layers comparison:")
print(f"{'Layer':<40} {'Pretrained Norm':<18} {'Trained Norm':<18} {'Change %':<12}")
print("-"*90)

first_layers = [k for k in sorted(common_layers) if any(x in k for x in ['conv1', 'bn1', 'layer1.0'])][:5]
for layer in first_layers:
    pre_norm = pretrained_stats[layer]['norm']
    train_norm = trained_stats[layer]['norm']
    change = ((train_norm - pre_norm) / pre_norm) * 100 if pre_norm != 0 else 0
    print(f"{layer:<40} {pre_norm:<18.6f} {train_norm:<18.6f} {change:>11.2f}%")

# Compare last few layers (should change during training)
print("\nLast 5 layers comparison:")
print(f"{'Layer':<40} {'Pretrained Norm':<18} {'Trained Norm':<18} {'Change %':<12}")
print("-"*90)

last_layers = [k for k in sorted(common_layers) if any(x in k for x in ['layer4', 'fc', 'bn2'])][-5:]
for layer in last_layers:
    pre_norm = pretrained_stats[layer]['norm']
    train_norm = trained_stats[layer]['norm']
    change = ((train_norm - pre_norm) / pre_norm) * 100 if pre_norm != 0 else 0
    print(f"{layer:<40} {pre_norm:<18.6f} {train_norm:<18.6f} {change:>11.2f}%")

# ============================================================================
# 4. Test embedding quality with pretrained model
# ============================================================================
print("\n[4] Testing pretrained model embedding quality...")

# Load iResNet architecture
try:
    from app.pipelines.arcface_model import build_iresnet100_backbone
    
    # Load pretrained model
    print("\nLoading pretrained model into architecture...")
    model_pre = build_iresnet100_backbone()
    missing, unexpected = model_pre.load_state_dict(pretrained, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    
    model_pre.eval()
    
    # Create test images (simulated faces)
    print("\nGenerating test embeddings with pretrained model...")
    batch_size = 10
    
    # Same person (base + small noise)
    base_img = torch.randn(1, 3, 112, 112) * 0.5
    same_person = base_img + torch.randn(batch_size, 3, 112, 112) * 0.05
    
    # Different people (random)
    diff_person = torch.randn(batch_size, 3, 112, 112) * 0.5
    
    with torch.no_grad():
        # Same person embeddings
        emb_same = model_pre(same_person)
        emb_same = F.normalize(emb_same, p=2, dim=1)
        
        # Different person embeddings
        emb_diff = model_pre(diff_person)
        emb_diff = F.normalize(emb_diff, p=2, dim=1)
        
        # Compute similarities
        # Same person: all pairs within same_person batch
        sim_same = []
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                sim_same.append((emb_same[i] * emb_same[j]).sum().item())
        
        # Different people: pairs between same_person[0] and diff_person
        sim_diff = []
        for j in range(batch_size):
            sim_diff.append((emb_same[0] * emb_diff[j]).sum().item())
        
        sim_same = np.array(sim_same)
        sim_diff = np.array(sim_diff)
        
        print(f"\nPRETRAINED MODEL RESULTS:")
        print(f"  Same person similarity:  {sim_same.mean():.4f} ± {sim_same.std():.4f}")
        print(f"  Different people:        {sim_diff.mean():.4f} ± {sim_diff.std():.4f}")
        print(f"  Separation:              {sim_same.mean() - sim_diff.mean():.4f}")
        
        if sim_same.mean() - sim_diff.mean() > 0.15:
            print("  ✅ EXCELLENT discrimination")
        elif sim_same.mean() - sim_diff.mean() > 0.10:
            print("  ⚠️  MODERATE discrimination")
        else:
            print("  ❌ POOR discrimination")

except Exception as e:
    print(f"  ❌ Error loading pretrained model: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# 5. Check for Sub-Center ArcFace head in trained model
# ============================================================================
print("\n[5] Checking for classification head...")

# Look for fc layer (classification head)
fc_layers = {k: v for k, v in trained.items() if 'fc' in k.lower()}
if fc_layers:
    print(f"\nFound FC layers in trained model: {len(fc_layers)}")
    for k, v in fc_layers.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")
else:
    print("\nNo FC layers found in trained model")

# ============================================================================
# 6. Diagnosis
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

diagnosis = []

# Check 1: Wrapper presence
if has_wrapper:
    print("\n✅ Model has NormalizedBackbone wrapper (correct)")
else:
    print("\n❌ Model missing NormalizedBackbone wrapper!")
    diagnosis.append("Model should be wrapped with NormalizedBackbone for L2 normalization")

# Check 2: Weight changes
if len(first_layers) > 0:
    first_change = abs(((trained_stats[first_layers[0]]['norm'] - pretrained_stats[first_layers[0]]['norm']) 
                       / pretrained_stats[first_layers[0]]['norm']) * 100)
    if first_change < 5.0:
        print(f"✅ Early layers mostly preserved (changed {first_change:.2f}%)")
    else:
        print(f"⚠️  Early layers changed significantly ({first_change:.2f}%)")
        diagnosis.append("Early layers should be frozen or have minimal changes during fine-tuning")

# Check 3: Classification head
if fc_layers:
    fc_shape = next(iter(fc_layers.values())).shape
    print(f"⚠️  Model has classification head: {fc_shape}")
    if len(fc_shape) == 2 and fc_shape[0] == 512:
        print(f"   Output dim: {fc_shape[1]} (Sub-Center ArcFace head)")
        diagnosis.append("Classification head should be REMOVED before deployment")
        diagnosis.append("Backend should only use embeddings BEFORE the FC layer")
    
# Summary
if diagnosis:
    print("\n🔧 ISSUES FOUND:")
    for i, issue in enumerate(diagnosis, 1):
        print(f"  {i}. {issue}")
else:
    print("\n✅ No obvious architectural issues found")
    print("   Problem likely in:")
    print("   - Preprocessing pipeline mismatch (already fixed)")
    print("   - Stored embeddings using old preprocessing (need re-registration)")
    print("   - Training overfitting to specific dataset")

print("\n" + "="*80)
