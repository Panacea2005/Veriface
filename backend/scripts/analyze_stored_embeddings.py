"""
Analyze stored embeddings for single user
Check if preprocessing was correct
"""

import torch
import json
import numpy as np

print("="*80)
print("STORED EMBEDDINGS ANALYSIS - SWS00001")
print("="*80)

# Load registry
with open("app/store/registry.json", 'r') as f:
    registry = json.load(f)

if 'SWS00001' not in registry:
    print("❌ SWS00001 not found")
    print(f"Available users: {list(registry.keys())}")
    exit(1)

user = registry['SWS00001']
embeddings = torch.tensor(user['embeddings'], dtype=torch.float32)

print(f"\nUser: {user['name']}")
print(f"Embeddings: {embeddings.shape}")

# ============================================================================
# Check embedding statistics
# ============================================================================
print("\n" + "="*80)
print("EMBEDDING STATISTICS")
print("="*80)

for i, emb in enumerate(embeddings, 1):
    norm = emb.norm().item()
    mean = emb.mean().item()
    std = emb.std().item()
    min_val = emb.min().item()
    max_val = emb.max().item()
    
    print(f"\nEmbedding {i}:")
    print(f"  Norm: {norm:.6f} (should be ~1.0 for L2 normalized)")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std:  {std:.6f}")
    print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")

# ============================================================================
# Self-similarity analysis
# ============================================================================
print("\n" + "="*80)
print("SELF-SIMILARITY (embeddings from same person)")
print("="*80)

# Compute pairwise cosine similarity
similarity_matrix = embeddings @ embeddings.T

print("\nPairwise cosine similarity matrix:")
print(similarity_matrix.numpy())

# Get off-diagonal elements (exclude self-similarity of 1.0)
mask = torch.ones_like(similarity_matrix, dtype=torch.bool)
mask.fill_diagonal_(False)
off_diag = similarity_matrix[mask]

print(f"\nOff-diagonal statistics (different poses of same person):")
print(f"  Mean: {off_diag.mean():.4f}")
print(f"  Std:  {off_diag.std():.4f}")
print(f"  Min:  {off_diag.min():.4f}")
print(f"  Max:  {off_diag.max():.4f}")

# Expected: 0.75-0.95 for same person different angles
if off_diag.min() < 0.70:
    print(f"\n⚠️  WARNING: Minimum similarity {off_diag.min():.4f} is quite low")
    print("  Some poses may be very different or preprocessing had issues")

if off_diag.mean() < 0.75:
    print(f"\n⚠️  WARNING: Average similarity {off_diag.mean():.4f} is low")
    print("  Expected: 0.80-0.90 for same person")
    print("  Possible causes:")
    print("    - Inconsistent preprocessing between registration and verification")
    print("    - Poor quality embeddings from model")
    print("    - Extreme pose variations")
elif off_diag.mean() > 0.95:
    print(f"\n⚠️  WARNING: Average similarity {off_diag.mean():.4f} is too high")
    print("  Embeddings may be too similar (lack of pose diversity)")
    print("  Or model is not discriminative enough")
else:
    print(f"\n✅ Average similarity {off_diag.mean():.4f} looks reasonable")

# ============================================================================
# Check for normalization issues
# ============================================================================
print("\n" + "="*80)
print("NORMALIZATION CHECK")
print("="*80)

norms = embeddings.norm(dim=1)
print(f"\nL2 norms of all embeddings:")
for i, norm in enumerate(norms, 1):
    status = "✅" if abs(norm.item() - 1.0) < 0.01 else "❌"
    print(f"  Embedding {i}: {norm.item():.6f} {status}")

if (norms - 1.0).abs().max() > 0.01:
    print(f"\n❌ CRITICAL: Embeddings NOT L2-normalized!")
    print("  This means:")
    print("    - Model doesn't have NormalizedBackbone wrapper, OR")
    print("    - Backend code strips normalization layer, OR")
    print("    - Stored embeddings were saved before normalization")
    print("\n  Backend should use model with NormalizedBackbone wrapper!")
else:
    print(f"\n✅ All embeddings are properly L2-normalized")

# ============================================================================
# Diagnosis
# ============================================================================
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

issues = []

# Check normalization
if (norms - 1.0).abs().max() > 0.01:
    issues.append("❌ Embeddings not L2-normalized (model architecture issue)")

# Check similarity range
if off_diag.mean() < 0.75:
    issues.append(f"❌ Low self-similarity ({off_diag.mean():.4f}) - preprocessing mismatch likely")
elif off_diag.mean() > 0.95:
    issues.append(f"⚠️  Very high self-similarity ({off_diag.mean():.4f}) - model may not be discriminative")

# Check similarity variation
if off_diag.std() > 0.15:
    issues.append(f"⚠️  High variation in self-similarity (std={off_diag.std():.4f}) - inconsistent preprocessing")

if not issues:
    print("\n✅ Stored embeddings look good!")
    print("  If verification still fails, check:")
    print("    - Threshold setting (currently 0.4)")
    print("    - Query-time preprocessing")
    print("    - Need second user to test cross-person discrimination")
else:
    print("\n🔧 ISSUES FOUND:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n💡 RECOMMENDED FIX:")
    print("  1. Verify backend uses NormalizedBackbone wrapper")
    print("  2. Verify preprocessing uses (x-127.5)/128.0 normalization")
    print("  3. DELETE registry and RE-REGISTER all users")
    print("  4. Restart backend to apply any code changes")

print("\n" + "="*80)
