"""Test Model B ConvNeXt discrimination - Can it distinguish different people?

This script analyzes embeddings in the registry to check if Model B can
discriminate between different people:
- Intra-user similarity (same person): Should be HIGH (0.7-0.9)
- Inter-user similarity (different people): Should be LOW (<0.4)
"""
from __future__ import annotations

import sys
from pathlib import Path
import json
import numpy as np

# Ensure project root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import REGISTRY_PATH

print("=" * 80)
print("MODEL B CONVNEXT DISCRIMINATION TEST")
print("=" * 80)
print()

# ============================================================================
# 1. Load Registry
# ============================================================================
print("1. LOADING REGISTRY")
print("-" * 80)

if not REGISTRY_PATH.exists():
    print(f"[ERROR] Registry not found at {REGISTRY_PATH}")
    sys.exit(1)

with open(REGISTRY_PATH, 'r') as f:
    registry = json.load(f)

if not registry:
    print("[ERROR] Registry is empty. Please register at least 2 users first.")
    sys.exit(1)

print(f"[OK] Found {len(registry)} users in registry")
for user_id, data in registry.items():
    num_emb = len(data.get('embeddings', []))
    name = data.get('name', user_id)
    print(f"  - {user_id} ({name}): {num_emb} embeddings")

if len(registry) < 2:
    print("\n[WARNING] Need at least 2 users to test cross-person discrimination.")
    print("          Will only analyze intra-user similarity.")
    print()

# ============================================================================
# 2. Convert to NumPy Arrays
# ============================================================================
print("\n2. PROCESSING EMBEDDINGS")
print("-" * 80)

user_embeddings = {}
for user_id, data in registry.items():
    embeddings = data.get('embeddings', [])
    if not embeddings:
        print(f"[WARN] {user_id} has no embeddings, skipping")
        continue
    
    # Convert to numpy arrays
    embs = [np.array(emb, dtype=np.float32) for emb in embeddings]
    
    # Check normalization
    norms = [np.linalg.norm(emb) for emb in embs]
    avg_norm = np.mean(norms)
    
    if abs(avg_norm - 1.0) > 0.01:
        print(f"[WARN] {user_id}: embeddings not normalized (avg norm={avg_norm:.4f})")
        # Normalize them
        embs = [emb / (np.linalg.norm(emb) + 1e-8) for emb in embs]
        print(f"       → Normalized embeddings")
    
    user_embeddings[user_id] = {
        'name': data.get('name', user_id),
        'embeddings': embs
    }

if len(user_embeddings) == 0:
    print("[ERROR] No valid embeddings found in registry")
    sys.exit(1)

print(f"[OK] Processed {len(user_embeddings)} users with valid embeddings")

# ============================================================================
# 3. Intra-User Similarity (Same Person, Different Angles)
# ============================================================================
print("\n3. INTRA-USER SIMILARITY (Same Person)")
print("-" * 80)
print("Expected: HIGH similarity (0.7-0.9) for same person, different angles")
print()

intra_similarities = {}
all_intra_sims = []

for user_id, data in user_embeddings.items():
    embs = data['embeddings']
    name = data['name']
    
    if len(embs) < 2:
        print(f"{user_id} ({name}): N/A (only {len(embs)} embedding)")
        continue
    
    # Compute all pairwise similarities
    sims = []
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            sim = np.dot(embs[i], embs[j])  # Cosine similarity (already normalized)
            sims.append(sim)
    
    intra_similarities[user_id] = sims
    all_intra_sims.extend(sims)
    
    mean_sim = np.mean(sims)
    std_sim = np.std(sims)
    min_sim = np.min(sims)
    max_sim = np.max(sims)
    
    print(f"{user_id} ({name}):")
    print(f"  Embeddings: {len(embs)}")
    print(f"  Pairs: {len(sims)}")
    print(f"  Mean:  {mean_sim:.4f} ({mean_sim*100:.2f}%)")
    print(f"  Std:   {std_sim:.4f}")
    print(f"  Range: [{min_sim:.4f}, {max_sim:.4f}]")
    
    # Evaluation
    if mean_sim >= 0.7:
        print(f"  Status: [OK] Good intra-user similarity")
    elif mean_sim >= 0.5:
        print(f"  Status: [WARN] Moderate intra-user similarity (may need more training)")
    else:
        print(f"  Status: [ERROR] Low intra-user similarity (model may not be working)")
    print()

# Overall intra-user statistics
if all_intra_sims:
    print("OVERALL INTRA-USER STATISTICS:")
    print(f"  Mean:  {np.mean(all_intra_sims):.4f} ({np.mean(all_intra_sims)*100:.2f}%)")
    print(f"  Std:   {np.std(all_intra_sims):.4f}")
    print(f"  Min:   {np.min(all_intra_sims):.4f} ({np.min(all_intra_sims)*100:.2f}%)")
    print(f"  Max:   {np.max(all_intra_sims):.4f} ({np.max(all_intra_sims)*100:.2f}%)")
    print()

# ============================================================================
# 4. Inter-User Similarity (Different People) - CRITICAL TEST
# ============================================================================
print("\n4. INTER-USER SIMILARITY (Different People) - CRITICAL TEST")
print("-" * 80)
print("Expected: LOW similarity (<0.4) for different people")
print()

if len(user_embeddings) < 2:
    print("[SKIP] Need at least 2 users for inter-user analysis")
else:
    user_ids = list(user_embeddings.keys())
    inter_similarities = {}
    all_inter_sims = []
    
    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            user1_id = user_ids[i]
            user2_id = user_ids[j]
            user1_name = user_embeddings[user1_id]['name']
            user2_name = user_embeddings[user2_id]['name']
            
            embs1 = user_embeddings[user1_id]['embeddings']
            embs2 = user_embeddings[user2_id]['embeddings']
            
            # Compute all cross-similarities
            sims = []
            for emb1 in embs1:
                for emb2 in embs2:
                    sim = np.dot(emb1, emb2)  # Cosine similarity
                    sims.append(sim)
            
            pair_name = f"{user1_id} vs {user2_id}"
            inter_similarities[pair_name] = sims
            all_inter_sims.extend(sims)
            
            mean_sim = np.mean(sims)
            std_sim = np.std(sims)
            min_sim = np.min(sims)
            max_sim = np.max(sims)
            
            print(f"{user1_id} ({user1_name}) vs {user2_id} ({user2_name}):")
            print(f"  Pairs: {len(sims)}")
            print(f"  Mean:  {mean_sim:.4f} ({mean_sim*100:.2f}%)")
            print(f"  Std:   {std_sim:.4f}")
            print(f"  Range: [{min_sim:.4f}, {max_sim:.4f}]")
            
            # Critical evaluation
            if max_sim < 0.4:
                print(f"  Status: [OK] Excellent discrimination (max < 0.4)")
            elif max_sim < 0.6:
                print(f"  Status: [WARN] Moderate discrimination (max < 0.6, may cause false matches)")
            else:
                print(f"  Status: [ERROR] Poor discrimination (max >= 0.6, HIGH RISK of false matches!)")
            print()
    
    # Overall inter-user statistics
    if all_inter_sims:
        print("OVERALL INTER-USER STATISTICS:")
        print(f"  Mean:  {np.mean(all_inter_sims):.4f} ({np.mean(all_inter_sims)*100:.2f}%)")
        print(f"  Std:   {np.std(all_inter_sims):.4f}")
        print(f"  Min:   {np.min(all_inter_sims):.4f} ({np.min(all_inter_sims)*100:.2f}%)")
        print(f"  Max:   {np.max(all_inter_sims):.4f} ({np.max(all_inter_sims)*100:.2f}%)")
        print()

# ============================================================================
# 5. Discrimination Analysis
# ============================================================================
print("\n5. DISCRIMINATION ANALYSIS")
print("-" * 80)

if len(user_embeddings) >= 2 and all_intra_sims and all_inter_sims:
    intra_mean = np.mean(all_intra_sims)
    inter_mean = np.mean(all_inter_sims)
    inter_max = np.max(all_inter_sims)
    
    # Separation gap
    separation = intra_mean - inter_mean
    min_separation = np.min(all_intra_sims) - inter_max
    
    print(f"Intra-user mean:  {intra_mean:.4f} ({intra_mean*100:.2f}%)")
    print(f"Inter-user mean:  {inter_mean:.4f} ({inter_mean*100:.2f}%)")
    print(f"Inter-user max:   {inter_max:.4f} ({inter_max*100:.2f}%)")
    print(f"Separation gap:   {separation:.4f} (mean difference)")
    print(f"Min separation:   {min_separation:.4f} (worst case)")
    print()
    
    # Verdict
    print("VERDICT:")
    print("-" * 80)
    
    if inter_max < 0.4 and separation > 0.4:
        print("[OK] EXCELLENT DISCRIMINATION")
        print("  -> Model B can clearly distinguish different people")
        print("  -> Inter-user similarity is low (<0.4)")
        print("  -> Good separation between same and different people")
        print("  -> System should work correctly with threshold ~0.4-0.5")
    elif inter_max < 0.6 and separation > 0.3:
        print("[WARN] MODERATE DISCRIMINATION")
        print("  -> Model B can distinguish different people, but with some overlap")
        print("  -> Inter-user similarity is moderate (<0.6)")
        print("  -> May need higher threshold (~0.6-0.7) to avoid false matches")
        print("  -> Consider retraining with more diverse data")
    else:
        print("[ERROR] POOR DISCRIMINATION")
        print("  -> Model B may NOT be able to distinguish different people well")
        print("  -> Inter-user similarity is too high (>=0.6)")
        print("  -> High risk of false matches")
        print("  -> Model may need retraining or different architecture")
    
    # Threshold recommendation
    print()
    print("THRESHOLD RECOMMENDATION:")
    print("-" * 80)
    if inter_max < 0.4:
        recommended_threshold = 0.4
        print(f"  Recommended threshold: {recommended_threshold:.2f}")
        print(f"  -> This will correctly accept same person (>{recommended_threshold:.2f})")
        print(f"  -> And reject different people (<{recommended_threshold:.2f})")
    elif inter_max < 0.6:
        recommended_threshold = 0.6
        print(f"  Recommended threshold: {recommended_threshold:.2f}")
        print(f"  -> Higher threshold needed due to moderate discrimination")
        print(f"  -> May have some false negatives (same person rejected)")
    else:
        print("  [WARN] Cannot recommend threshold - discrimination too poor")
        print("  -> Need to improve model first")
    
elif len(user_embeddings) < 2:
    print("[SKIP] Need at least 2 users for discrimination analysis")
    if all_intra_sims:
        intra_mean = np.mean(all_intra_sims)
        print(f"\nIntra-user similarity: {intra_mean:.4f} ({intra_mean*100:.2f}%)")
        if intra_mean >= 0.7:
            print("[OK] Good intra-user similarity (same person embeddings are similar)")
        else:
            print("[WARN] Intra-user similarity is low - model may need more training")
else:
    print("[ERROR] Cannot compute discrimination - missing data")

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)

