"""
Analyze embeddings in registry to understand similarity patterns.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Load registry
registry_path = Path("Veriface/backend/app/store/registry.json")
with open(registry_path) as f:
    registry = json.load(f)

print("="*70)
print("REGISTRY EMBEDDING ANALYSIS")
print("="*70)

# Convert to numpy arrays
users = {}
for user_id, embeddings in registry.items():
    users[user_id] = [np.array(emb, dtype=np.float32) for emb in embeddings]
    print(f"\n{user_id}: {len(embeddings)} embeddings")

# Cosine similarity function
def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    return float(np.dot(v1_norm, v2_norm))

# Analyze intra-user similarity (same person, different embeddings)
print("\n" + "="*70)
print("INTRA-USER SIMILARITY (Same Person)")
print("="*70)

intra_similarities = {}

for user_id, embeddings in users.items():
    sims = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            sims.append(sim)
    
    intra_similarities[user_id] = sims
    print(f"\n{user_id}:")
    print(f"  Pairs: {len(sims)}")
    print(f"  Min:   {min(sims):.4f} ({min(sims)*100:.2f}%)")
    print(f"  Max:   {max(sims):.4f} ({max(sims)*100:.2f}%)")
    print(f"  Mean:  {np.mean(sims):.4f} ({np.mean(sims)*100:.2f}%)")
    print(f"  Std:   {np.std(sims):.4f}")

# Analyze inter-user similarity (different people)
print("\n" + "="*70)
print("INTER-USER SIMILARITY (Different People)")
print("="*70)

user_ids = list(users.keys())
inter_similarities = {}

for i in range(len(user_ids)):
    for j in range(i+1, len(user_ids)):
        user1 = user_ids[i]
        user2 = user_ids[j]
        
        sims = []
        for emb1 in users[user1]:
            for emb2 in users[user2]:
                sim = cosine_similarity(emb1, emb2)
                sims.append(sim)
        
        pair_name = f"{user1} vs {user2}"
        inter_similarities[pair_name] = sims
        
        print(f"\n{pair_name}:")
        print(f"  Pairs: {len(sims)}")
        print(f"  Min:   {min(sims):.4f} ({min(sims)*100:.2f}%)")
        print(f"  Max:   {max(sims):.4f} ({max(sims)*100:.2f}%)")
        print(f"  Mean:  {np.mean(sims):.4f} ({np.mean(sims)*100:.2f}%)")
        print(f"  Std:   {np.std(sims):.4f}")

# Overall statistics
print("\n" + "="*70)
print("OVERALL STATISTICS")
print("="*70)

all_intra = [sim for sims in intra_similarities.values() for sim in sims]
all_inter = [sim for sims in inter_similarities.values() for sim in sims]

print(f"\nINTRA-USER (Same Person):")
print(f"  Total pairs: {len(all_intra)}")
print(f"  Min:   {min(all_intra):.4f} ({min(all_intra)*100:.2f}%)")
print(f"  Max:   {max(all_intra):.4f} ({max(all_intra)*100:.2f}%)")
print(f"  Mean:  {np.mean(all_intra):.4f} ({np.mean(all_intra)*100:.2f}%)")
print(f"  Std:   {np.std(all_intra):.4f}")

print(f"\nINTER-USER (Different People):")
print(f"  Total pairs: {len(all_inter)}")
print(f"  Min:   {min(all_inter):.4f} ({min(all_inter)*100:.2f}%)")
print(f"  Max:   {max(all_inter):.4f} ({max(all_inter)*100:.2f}%)")
print(f"  Mean:  {np.mean(all_inter):.4f} ({np.mean(all_inter)*100:.2f}%)")
print(f"  Std:   {np.std(all_inter):.4f}")

# Calculate separation
print(f"\nSEPARATION ANALYSIS:")
intra_min = min(all_intra)
inter_max = max(all_inter)
gap = intra_min - inter_max

print(f"  Intra-user MIN:  {intra_min:.4f} ({intra_min*100:.2f}%)")
print(f"  Inter-user MAX:  {inter_max:.4f} ({inter_max*100:.2f}%)")
print(f"  Gap:             {gap:.4f} ({gap*100:.2f}%)")

if gap > 0:
    print(f"  ✅ GOOD: Clear separation (gap > 0)")
    print(f"     Any threshold in [{inter_max:.4f}, {intra_min:.4f}] will work perfectly!")
    recommended_threshold = (inter_max + intra_min) / 2
    print(f"     Recommended: {recommended_threshold:.4f} ({recommended_threshold*100:.2f}%)")
else:
    print(f"  ⚠️ OVERLAP: Intra and inter distributions overlap!")
    print(f"     Overlap region: [{intra_min:.4f}, {inter_max:.4f}]")
    
    # Find optimal threshold (minimize error)
    # Try thresholds from min to max
    test_thresholds = np.linspace(min(all_intra + all_inter), 
                                   max(all_intra + all_inter), 100)
    errors = []
    
    for threshold in test_thresholds:
        # False rejects: intra-user pairs below threshold
        fr = sum(1 for sim in all_intra if sim < threshold)
        frr = fr / len(all_intra)
        
        # False accepts: inter-user pairs above threshold
        fa = sum(1 for sim in all_inter if sim >= threshold)
        far = fa / len(all_inter)
        
        total_error = frr + far
        errors.append(total_error)
    
    optimal_idx = np.argmin(errors)
    optimal_threshold = test_thresholds[optimal_idx]
    optimal_error = errors[optimal_idx]
    
    print(f"     Optimal threshold: {optimal_threshold:.4f} ({optimal_threshold*100:.2f}%)")
    print(f"     Total error rate: {optimal_error*100:.2f}%")

# Threshold recommendations
print("\n" + "="*70)
print("THRESHOLD RECOMMENDATIONS")
print("="*70)

thresholds = [0.20, 0.23, 0.25, 0.30, 0.35, 0.40]

for threshold in thresholds:
    # False rejects: intra-user pairs below threshold
    fr = sum(1 for sim in all_intra if sim < threshold)
    frr = fr / len(all_intra) if all_intra else 0
    
    # False accepts: inter-user pairs above threshold
    fa = sum(1 for sim in all_inter if sim >= threshold)
    far = fa / len(all_inter) if all_inter else 0
    
    print(f"\nThreshold {threshold:.2f} ({threshold*100:.0f}%):")
    print(f"  FRR (False Reject):  {frr*100:.2f}% ({fr}/{len(all_intra)} same-person pairs rejected)")
    print(f"  FAR (False Accept):  {far*100:.2f}% ({fa}/{len(all_inter)} different-person pairs accepted)")
    print(f"  Total Error:         {(frr + far)*100:.2f}%")
    
    if frr == 0 and far == 0:
        print(f"  ✅ PERFECT: No errors!")
    elif frr + far < 0.10:
        print(f"  ✅ EXCELLENT: <10% total error")
    elif frr + far < 0.20:
        print(f"  ✅ GOOD: <20% total error")
    else:
        print(f"  ⚠️ HIGH ERROR: >{(frr + far)*100:.0f}% total error")

# Plot distributions
print("\n" + "="*70)
print("GENERATING PLOTS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Similarity distributions
ax = axes[0, 0]
ax.hist(all_intra, bins=30, alpha=0.6, color='blue', label=f'Same Person (n={len(all_intra)})', density=True)
ax.hist(all_inter, bins=30, alpha=0.6, color='red', label=f'Different People (n={len(all_inter)})', density=True)
ax.axvline(0.23, color='orange', linestyle='--', linewidth=2, label='Training (0.23)')
ax.axvline(0.30, color='purple', linestyle='--', linewidth=2, label='Production (0.30)')
ax.set_xlabel('Cosine Similarity', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Similarity Distribution', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Box plot by user
ax = axes[0, 1]
data_to_plot = []
labels = []
for user_id, sims in intra_similarities.items():
    data_to_plot.append(sims)
    labels.append(user_id.split()[0])  # First name only

bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax.axhline(0.30, color='purple', linestyle='--', linewidth=2, label='Threshold 0.30')
ax.set_ylabel('Cosine Similarity', fontsize=11)
ax.set_title('Intra-User Similarity by Person', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Pairwise similarity heatmap
ax = axes[1, 0]

# Create matrix of mean similarities
n_users = len(user_ids)
sim_matrix = np.zeros((n_users, n_users))

for i in range(n_users):
    for j in range(n_users):
        if i == j:
            # Intra-user similarity
            sim_matrix[i, j] = np.mean(intra_similarities[user_ids[i]])
        else:
            # Inter-user similarity
            pair_name = f"{user_ids[i]} vs {user_ids[j]}" if i < j else f"{user_ids[j]} vs {user_ids[i]}"
            sim_matrix[i, j] = np.mean(inter_similarities[pair_name])

sns.heatmap(sim_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
            xticklabels=[uid.split()[0] for uid in user_ids],
            yticklabels=[uid.split()[0] for uid in user_ids],
            vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Similarity'})
ax.set_title('Mean Similarity Matrix', fontsize=12, fontweight='bold')

# Plot 4: Error rates vs threshold
ax = axes[1, 1]
test_thresholds = np.linspace(0, 1, 100)
frrs = []
fars = []

for threshold in test_thresholds:
    fr = sum(1 for sim in all_intra if sim < threshold)
    frr = fr / len(all_intra) if all_intra else 0
    
    fa = sum(1 for sim in all_inter if sim >= threshold)
    far = fa / len(all_inter) if all_inter else 0
    
    frrs.append(frr * 100)
    fars.append(far * 100)

ax.plot(test_thresholds, frrs, 'b-', linewidth=2, label='FRR (False Reject)')
ax.plot(test_thresholds, fars, 'r-', linewidth=2, label='FAR (False Accept)')
ax.axvline(0.23, color='orange', linestyle='--', linewidth=1.5, label='Training (0.23)')
ax.axvline(0.30, color='purple', linestyle='--', linewidth=1.5, label='Production (0.30)')
ax.set_xlabel('Threshold', fontsize=11)
ax.set_ylabel('Error Rate (%)', fontsize=11)
ax.set_title('FAR/FRR vs Threshold', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('registry_embedding_analysis.png', dpi=150, bbox_inches='tight')
print("\n✅ Plot saved as 'registry_embedding_analysis.png'")
plt.show()

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
