"""
Analyze cross-person discrimination between SWS00001 and SWS00002
"""

import torch
import json
import numpy as np

print("="*80)
print("CROSS-PERSON DISCRIMINATION TEST")
print("="*80)

# Load registry
with open("app/store/registry.json", 'r') as f:
    registry = json.load(f)

if 'SWS00001' not in registry or 'SWS00002' not in registry:
    print("❌ Need both SWS00001 and SWS00002")
    print(f"Available users: {list(registry.keys())}")
    exit(1)

user1 = registry['SWS00001']
user2 = registry['SWS00002']

emb1 = torch.tensor(user1['embeddings'], dtype=torch.float32)
emb2 = torch.tensor(user2['embeddings'], dtype=torch.float32)

print(f"\nSWS00001 ({user1['name']}): {emb1.shape}")
print(f"SWS00002 ({user2['name']}): {emb2.shape}")

# ============================================================================
# Self-similarity (same person)
# ============================================================================
print("\n" + "="*80)
print("SELF-SIMILARITY (Same Person)")
print("="*80)

# User 1 self-similarity
sim_11 = emb1 @ emb1.T
mask = torch.ones_like(sim_11, dtype=torch.bool)
mask.fill_diagonal_(False)
sim_11_off = sim_11[mask]

print(f"\nSWS00001 self-similarity:")
print(f"  Mean: {sim_11_off.mean():.4f}")
print(f"  Std:  {sim_11_off.std():.4f}")
print(f"  Min:  {sim_11_off.min():.4f}")
print(f"  Max:  {sim_11_off.max():.4f}")

# User 2 self-similarity
sim_22 = emb2 @ emb2.T
mask = torch.ones_like(sim_22, dtype=torch.bool)
mask.fill_diagonal_(False)
sim_22_off = sim_22[mask]

print(f"\nSWS00002 self-similarity:")
print(f"  Mean: {sim_22_off.mean():.4f}")
print(f"  Std:  {sim_22_off.std():.4f}")
print(f"  Min:  {sim_22_off.min():.4f}")
print(f"  Max:  {sim_22_off.max():.4f}")

# ============================================================================
# Cross-similarity (different people) - THE CRITICAL TEST
# ============================================================================
print("\n" + "="*80)
print("CROSS-SIMILARITY (DIFFERENT PEOPLE) ← CRITICAL TEST")
print("="*80)

# All pairs between user1 and user2
sim_12 = emb1 @ emb2.T

print(f"\nSWS00001 vs SWS00002 (all 25 pairs):")
print(f"  Mean: {sim_12.mean():.4f}")
print(f"  Std:  {sim_12.std():.4f}")
print(f"  Min:  {sim_12.min():.4f}")
print(f"  Max:  {sim_12.max():.4f} ← WORST CASE (most similar)")

print(f"\nCross-similarity matrix (5x5):")
print(sim_12.numpy())

# ============================================================================
# Aggregation simulation (what backend does)
# ============================================================================
print("\n" + "="*80)
print("BACKEND AGGREGATION SIMULATION")
print("="*80)

print("\nIf User1 verifies against User2 database:")
print("  Method: MAX aggregation (take best match)")

# For each User1 embedding, find max similarity with any User2 embedding
max_scores = sim_12.max(dim=1).values

print(f"\n  User1 embedding 1 vs User2 (max): {max_scores[0]:.4f}")
print(f"  User1 embedding 2 vs User2 (max): {max_scores[1]:.4f}")
print(f"  User1 embedding 3 vs User2 (max): {max_scores[2]:.4f}")
print(f"  User1 embedding 4 vs User2 (max): {max_scores[3]:.4f}")
print(f"  User1 embedding 5 vs User2 (max): {max_scores[4]:.4f}")

final_score = max_scores.max().item()
print(f"\n  FINAL AGGREGATED SCORE: {final_score:.4f}")
print(f"  Threshold: 0.4")

if final_score > 0.4:
    print(f"  Result: ❌ FALSE POSITIVE (would accept wrong person!)")
else:
    print(f"  Result: ✅ CORRECT REJECTION")

# ============================================================================
# Diagnosis
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

# Expected discrimination
expected_self = 0.80  # Same person should be ~0.80
expected_cross = 0.30  # Different people should be ~0.30
expected_sep = expected_self - expected_cross  # Separation ~0.50

actual_self = (sim_11_off.mean() + sim_22_off.mean()) / 2
actual_cross = sim_12.mean()
actual_sep = actual_self - actual_cross

print(f"\nExpected performance:")
print(f"  Same person:     ~{expected_self:.2f}")
print(f"  Different people: ~{expected_cross:.2f}")
print(f"  Separation:       ~{expected_sep:.2f}")

print(f"\nActual performance:")
print(f"  Same person:     {actual_self:.4f}")
print(f"  Different people: {actual_cross:.4f}")
print(f"  Separation:       {actual_sep:.4f}")

print(f"\nDifference from expected:")
print(f"  Same person:     {(actual_self - expected_self):+.4f}")
print(f"  Different people: {(actual_cross - expected_cross):+.4f}")
print(f"  Separation:       {(actual_sep - expected_sep):+.4f}")

# ============================================================================
# Final verdict
# ============================================================================
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

issues = []

# Check separation
if actual_sep < 0.15:
    issues.append("❌ CRITICAL: Poor separation - cannot distinguish users")
elif actual_sep < 0.30:
    issues.append("⚠️  WARNING: Weak separation - may cause false positives")
else:
    print("\n✅ Good separation between users")

# Check cross-similarity absolute value
if actual_cross > 0.70:
    issues.append("❌ CRITICAL: Cross-similarity too high - embeddings too similar")
elif actual_cross > 0.50:
    issues.append("⚠️  WARNING: Cross-similarity high - need higher threshold")

# Check worst case (max)
if sim_12.max() > 0.75:
    issues.append(f"❌ CRITICAL: Worst case {sim_12.max():.4f} > 0.75 - definite false positive")
elif sim_12.max() > 0.60:
    issues.append(f"⚠️  WARNING: Worst case {sim_12.max():.4f} > 0.60 - risky threshold")

if issues:
    print("\n🚨 ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n💡 ROOT CAUSES:")
    print("  1. ❌ Model B trained poorly (overfitting/underfitting)")
    print("  2. ❌ Training data quality issues")
    print("  3. ❌ Loss function not discriminative enough")
    print("  4. ❌ Embeddings not from trained model (using pretrained?)")
    
    print("\n🔧 SOLUTIONS:")
    print("  1. Verify backend loads modelB_best.pth (not pretrained)")
    print("  2. Check if model checkpoint is corrupted")
    print("  3. Re-train Model B with stricter parameters:")
    print("     - Increase margin to 0.6-0.7")
    print("     - Increase scale to 80-100")
    print("     - Train for more epochs")
    print("     - Use larger batch size")
else:
    print("\n✅✅✅ EXCELLENT! Model discriminates perfectly!")
    print(f"   Can use threshold range: {sim_12.max():.2f} - {actual_self:.2f}")

print("\n" + "="*80)
