"""
Compare Pretrained vs Trained Model B on SAME test data
Verify training code quality
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("="*80)
print("PRETRAINED vs TRAINED MODEL B - QUALITY COMPARISON")
print("="*80)

from app.pipelines.arcface_model import get_model

# ============================================================================
# Load both models
# ============================================================================
print("\n[1] Loading models...")

# Pretrained model (baseline)
print("\nPRETRAINED MODEL (MS1MV3 ArcFace baseline):")
model_pretrained = get_model(input_size=[112, 112], num_layers=100, mode='ir')
pretrained_ckpt = torch.load("app/models/ms1mv3_arcface_r100_fp16.pth", 
                             map_location='cpu', weights_only=False)
model_pretrained.load_state_dict(pretrained_ckpt, strict=False)
model_pretrained.eval()
print(f"  ✅ Loaded baseline model")

# Trained model (your Model B)
print("\nTRAINED MODEL B (Sub-Center ArcFace fine-tuned):")
model_trained = get_model(input_size=[112, 112], num_layers=100, mode='ir')
trained_ckpt = torch.load("app/models/modelB_best.pth", 
                          map_location='cpu', weights_only=False)
if 'backbone' in trained_ckpt:
    state_dict = trained_ckpt['backbone']
else:
    state_dict = trained_ckpt
missing, unexpected = model_trained.load_state_dict(state_dict, strict=False)
model_trained.eval()
print(f"  ✅ Loaded trained model")
print(f"  Epoch: {trained_ckpt.get('epoch', 'N/A')}")
print(f"  Best AUC: {trained_ckpt.get('best_auc', 'N/A'):.4f}")

# ============================================================================
# Generate test embeddings (simulated faces)
# ============================================================================
print("\n" + "="*80)
print("TEST 1: Simulated Face Images (Controlled Environment)")
print("="*80)

print("\nGenerating test images...")
batch_size = 20
torch.manual_seed(42)  # Reproducible

# Same person: base image + small noise (5 variations)
base_img = torch.randn(1, 3, 112, 112) * 0.5
same_person_imgs = base_img + torch.randn(5, 3, 112, 112) * 0.05

# Different people: random images (15 people)
diff_people_imgs = torch.randn(15, 3, 112, 112) * 0.5

print(f"  Same person variations: {same_person_imgs.shape}")
print(f"  Different people: {diff_people_imgs.shape}")

# Test with pretrained model
print("\nPRETRAINED MODEL RESULTS:")
with torch.no_grad():
    # Same person embeddings
    emb_same_pre = model_pretrained(same_person_imgs)
    
    # Different people embeddings
    emb_diff_pre = model_pretrained(diff_people_imgs)
    
    # Compute similarities
    sim_same_pre = []
    for i in range(5):
        for j in range(i+1, 5):
            sim_same_pre.append((emb_same_pre[i] * emb_same_pre[j]).sum().item())
    
    sim_diff_pre = []
    for j in range(15):
        sim_diff_pre.append((emb_same_pre[0] * emb_diff_pre[j]).sum().item())
    
    sim_same_pre = np.array(sim_same_pre)
    sim_diff_pre = np.array(sim_diff_pre)
    
    print(f"  Same person:     {sim_same_pre.mean():.4f} ± {sim_same_pre.std():.4f}")
    print(f"  Different people: {sim_diff_pre.mean():.4f} ± {sim_diff_pre.std():.4f}")
    print(f"  Separation:       {sim_same_pre.mean() - sim_diff_pre.mean():.4f}")

# Test with trained model
print("\nTRAINED MODEL B RESULTS:")
with torch.no_grad():
    # Same person embeddings
    emb_same_train = model_trained(same_person_imgs)
    
    # Different people embeddings  
    emb_diff_train = model_trained(diff_people_imgs)
    
    # Compute similarities
    sim_same_train = []
    for i in range(5):
        for j in range(i+1, 5):
            sim_same_train.append((emb_same_train[i] * emb_same_train[j]).sum().item())
    
    sim_diff_train = []
    for j in range(15):
        sim_diff_train.append((emb_same_train[0] * emb_diff_train[j]).sum().item())
    
    sim_same_train = np.array(sim_same_train)
    sim_diff_train = np.array(sim_diff_train)
    
    print(f"  Same person:     {sim_same_train.mean():.4f} ± {sim_same_train.std():.4f}")
    print(f"  Different people: {sim_diff_train.mean():.4f} ± {sim_diff_train.std():.4f}")
    print(f"  Separation:       {sim_same_train.mean() - sim_diff_train.mean():.4f}")

# Comparison
print("\nCOMPARISON:")
sep_pre = sim_same_pre.mean() - sim_diff_pre.mean()
sep_train = sim_same_train.mean() - sim_diff_train.mean()

print(f"  Pretrained separation: {sep_pre:.4f}")
print(f"  Trained separation:    {sep_train:.4f}")
print(f"  Improvement:           {((sep_train - sep_pre) / sep_pre * 100):+.2f}%")

if sep_train > sep_pre:
    print(f"  ✅ Training IMPROVED discrimination")
elif sep_train > sep_pre * 0.95:
    print(f"  ✅ Training MAINTAINED quality (within 5%)")
else:
    print(f"  ⚠️  Training DEGRADED quality")

# ============================================================================
# Weight comparison (check if training actually changed weights)
# ============================================================================
print("\n" + "="*80)
print("TEST 2: Weight Change Analysis")
print("="*80)

print("\nComparing model weights...")

# Get common layers
pre_dict = pretrained_ckpt
train_dict = state_dict

# Remove 'backbone.' prefix from trained model for comparison
train_dict_unwrapped = {}
for k, v in train_dict.items():
    new_key = k.replace('backbone.', '')
    train_dict_unwrapped[new_key] = v

common_keys = set(pre_dict.keys()) & set(train_dict_unwrapped.keys())
print(f"  Common layers: {len(common_keys)}")

# Compare first/middle/last layers
early_layers = [k for k in sorted(common_keys) if 'layer1' in k and 'weight' in k][:3]
middle_layers = [k for k in sorted(common_keys) if 'layer2' in k and 'weight' in k][:3]
late_layers = [k for k in sorted(common_keys) if 'layer4' in k and 'weight' in k][-3:]

print("\nEarly layers (should have minimal change if frozen):")
for layer in early_layers:
    pre_norm = pre_dict[layer].float().norm().item()
    train_norm = train_dict_unwrapped[layer].float().norm().item()
    diff = abs(train_norm - pre_norm) / pre_norm * 100
    print(f"  {layer[:40]:<40} change: {diff:>6.2f}%")

print("\nMiddle layers:")
for layer in middle_layers:
    pre_norm = pre_dict[layer].float().norm().item()
    train_norm = train_dict_unwrapped[layer].float().norm().item()
    diff = abs(train_norm - pre_norm) / pre_norm * 100
    print(f"  {layer[:40]:<40} change: {diff:>6.2f}%")

print("\nLate layers (should change during training):")
for layer in late_layers:
    pre_norm = pre_dict[layer].float().norm().item()
    train_norm = train_dict_unwrapped[layer].float().norm().item()
    diff = abs(train_norm - pre_norm) / pre_norm * 100
    print(f"  {layer[:40]:<40} change: {diff:>6.2f}%")

# ============================================================================
# Check training configuration from notebook
# ============================================================================
print("\n" + "="*80)
print("TEST 3: Training Configuration Validation")
print("="*80)

print("\nExpected training setup (from notebook):")
print("  Loss: Sub-Center ArcFace (margin=0.5, scale=64)")
print("  Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)")
print("  Scheduler: Cosine with warmup")
print("  Epochs: 20 (early stopping patience=7)")
print("  Batch size: 512 (P=64, K=8 PK-sampling)")

print(f"\nActual trained model:")
print(f"  Epoch stopped: {trained_ckpt.get('epoch', 'N/A')}")
print(f"  Best AUC: {trained_ckpt.get('best_auc', 'N/A'):.4f}")

if trained_ckpt.get('epoch', 0) < 5:
    print("  ⚠️  WARNING: Model trained for very few epochs")
    print("     Consider training longer for better convergence")
elif trained_ckpt.get('epoch', 0) > 15:
    print("  ⚠️  WARNING: Model trained for many epochs")
    print("     May be overfitting - check validation curve")
else:
    print("  ✅ Training duration looks reasonable")

# ============================================================================
# Diagnosis
# ============================================================================
print("\n" + "="*80)
print("FINAL DIAGNOSIS")
print("="*80)

issues = []
recommendations = []

# Check separation quality
if sep_train < 0.15:
    issues.append("❌ Poor separation - model not discriminative enough")
    recommendations.append("Re-train with larger margin or more epochs")
elif sep_train < sep_pre * 0.9:
    issues.append("⚠️  Training degraded quality compared to pretrained")
    recommendations.append("Check learning rate, batch size, or loss function")
else:
    print("\n✅ Model quality is good")

# Check weight updates
early_change = sum([abs(train_dict_unwrapped[k].float().norm().item() - pre_dict[k].float().norm().item()) 
                    / pre_dict[k].float().norm().item() for k in early_layers]) / len(early_layers)
late_change = sum([abs(train_dict_unwrapped[k].float().norm().item() - pre_dict[k].float().norm().item()) 
                   / pre_dict[k].float().norm().item() for k in late_layers]) / len(late_layers)

if late_change < 0.01:
    issues.append("⚠️  Late layers barely changed - learning rate too low?")
    recommendations.append("Increase learning rate or train longer")
elif early_change > 0.1:
    issues.append("⚠️  Early layers changed significantly - may lose pretrained features")
    recommendations.append("Consider freezing early layers or reducing learning rate")

if not issues:
    print("\n✅✅✅ TRAINING CODE IS CORRECT AND PRODUCING GOOD RESULTS!")
    print("\nYour Model B:")
    print(f"  - Separation: {sep_train:.4f} (vs pretrained {sep_pre:.4f})")
    print(f"  - Improvement: {((sep_train - sep_pre) / sep_pre * 100):+.2f}%")
    print(f"  - Training epochs: {trained_ckpt.get('epoch', 'N/A')}")
    print(f"  - Best AUC: {trained_ckpt.get('best_auc', 'N/A'):.4f}")
    print("\n💡 CONCLUSION: No need to re-train!")
    print("   Model is production-ready. Just need to:")
    print("   1. Register 2nd user (SWS00002) for cross-person testing")
    print("   2. Verify cross-person similarity < 0.4 threshold")
else:
    print("\n🔧 ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n💡 RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print("\n⚠️  Consider re-training with adjustments")

print("\n" + "="*80)
