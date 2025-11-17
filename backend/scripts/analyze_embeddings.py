"""Analyze embeddings in registry to check if different people have distinguishable embeddings."""
import sys
from pathlib import Path
import json
import numpy as np

# Add backend to path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import REGISTRY_PATH

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)

def analyze_registry():
    """Analyze embeddings in registry."""
    if not REGISTRY_PATH.exists():
        print(f"[ERROR] Registry not found at {REGISTRY_PATH}")
        return
    
    # Load registry
    with open(REGISTRY_PATH, 'r') as f:
        registry = json.load(f)
    
    if not registry:
        print("[ERROR] Registry is empty")
        return
    
    print(f"[INFO] Found {len(registry)} users in registry")
    print("=" * 80)
    
    # Convert to numpy arrays
    user_embeddings = {}
    for user_id, data in registry.items():
        embeddings = data.get("embeddings", [])
        if embeddings:
            user_embeddings[user_id] = {
                "name": data.get("name", user_id),
                "embeddings": [np.array(emb, dtype=np.float32) for emb in embeddings]
            }
            print(f"  {user_id} ({data.get('name', user_id)}): {len(embeddings)} embeddings")
    
    if len(user_embeddings) < 1:
        print("[ERROR] No users found in registry")
        return
    
    if len(user_embeddings) < 2:
        print("[WARNING] Only 1 user found. Will analyze intra-user similarity only.")
        print("[WARNING] Inter-user comparison requires at least 2 users.")
    
    print("=" * 80)
    print("\n[ANALYSIS] Intra-user similarity (same person, different angles):")
    print("-" * 80)
    
    intra_similarities = []
    for user_id, data in user_embeddings.items():
        embeddings = data["embeddings"]
        name = data["name"]
        
        if len(embeddings) < 2:
            print(f"  {user_id} ({name}): Only 1 embedding, skipping intra-user comparison")
            continue
        
        # Compare all pairs within same user
        similarities = []
        similarity_matrix = []
        for i in range(len(embeddings)):
            row = []
            for j in range(len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                row.append(sim)
                if i < j:  # Only count each pair once
                    similarities.append(sim)
                    intra_similarities.append(sim)
            similarity_matrix.append(row)
        
        if similarities:
            avg_sim = np.mean(similarities)
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)
            std_sim = np.std(similarities)
            print(f"  {user_id} ({name}):")
            print(f"    Avg: {avg_sim:.4f}, Min: {min_sim:.4f}, Max: {max_sim:.4f}, Std: {std_sim:.4f}")
            print(f"    Range: [{min_sim:.4f}, {max_sim:.4f}]")
            
            # Show similarity matrix for 5-angle analysis
            if len(embeddings) == 5:
                print(f"    Similarity matrix (5 angles):")
                print(f"      Angle:   1       2       3       4       5")
                angles = ["Front", "Left", "Right", "Up", "Down"]
                for i in range(5):
                    row_str = f"      {angles[i]:6s}: "
                    for j in range(5):
                        row_str += f"{similarity_matrix[i][j]:.3f}  "
                    print(row_str)
                
                # Check if all angles are too similar
                non_diag_sims = [similarity_matrix[i][j] for i in range(5) for j in range(5) if i != j]
                if np.mean(non_diag_sims) > 0.85:
                    print(f"    [WARN] All 5 angles are very similar (avg: {np.mean(non_diag_sims):.3f})")
                    print(f"           Model may not be capturing angle differences well")
                elif np.mean(non_diag_sims) > 0.70:
                    print(f"    [INFO] Angles are moderately similar (avg: {np.mean(non_diag_sims):.3f})")
                else:
                    print(f"    [OK] Angles show good variation (avg: {np.mean(non_diag_sims):.3f})")
    
    print("\n[ANALYSIS] Inter-user similarity (different people):")
    print("-" * 80)
    
    inter_similarities = []
    user_ids = list(user_embeddings.keys())
    
    if len(user_ids) < 2:
        print("  [SKIP] Need at least 2 users for inter-user comparison")
    else:
        for i in range(len(user_ids)):
            for j in range(i + 1, len(user_ids)):
                user1_id = user_ids[i]
                user2_id = user_ids[j]
                user1_name = user_embeddings[user1_id]["name"]
                user2_name = user_embeddings[user2_id]["name"]
                
                embeddings1 = user_embeddings[user1_id]["embeddings"]
                embeddings2 = user_embeddings[user2_id]["embeddings"]
                
                # Compare all pairs between different users
                similarities = []
                for emb1 in embeddings1:
                    for emb2 in embeddings2:
                        sim = cosine_similarity(emb1, emb2)
                        similarities.append(sim)
                        inter_similarities.append(sim)
                
                if similarities:
                    avg_sim = np.mean(similarities)
                    min_sim = np.min(similarities)
                    max_sim = np.max(similarities)
                    std_sim = np.std(similarities)
                    print(f"  {user1_id} ({user1_name}) vs {user2_id} ({user2_name}):")
                    print(f"    Avg: {avg_sim:.4f}, Min: {min_sim:.4f}, Max: {max_sim:.4f}, Std: {std_sim:.4f}")
    
    print("\n" + "=" * 80)
    print("[SUMMARY] Overall Statistics:")
    print("-" * 80)
    
    if intra_similarities:
        intra_mean = np.mean(intra_similarities)
        intra_min = np.min(intra_similarities)
        intra_max = np.max(intra_similarities)
        intra_std = np.std(intra_similarities)
        print(f"Intra-user (same person):")
        print(f"  Mean: {intra_mean:.4f}")
        print(f"  Min:  {intra_min:.4f}")
        print(f"  Max:  {intra_max:.4f}")
        print(f"  Std:  {intra_std:.4f}")
        print(f"  Count: {len(intra_similarities)} pairs")
    
    if inter_similarities:
        inter_mean = np.mean(inter_similarities)
        inter_min = np.min(inter_similarities)
        inter_max = np.max(inter_similarities)
        inter_std = np.std(inter_similarities)
        print(f"\nInter-user (different people):")
        print(f"  Mean: {inter_mean:.4f}")
        print(f"  Min:  {inter_min:.4f}")
        print(f"  Max:  {inter_max:.4f}")
        print(f"  Std:  {inter_std:.4f}")
        print(f"  Count: {len(inter_similarities)} pairs")
    
    # Separation analysis
    if intra_similarities and inter_similarities:
        print("\n[SEPARATION ANALYSIS]:")
        print("-" * 80)
        
        # Find best threshold (where FAR = FRR approximately)
        all_scores = []
        all_labels = []
        for score in intra_similarities:
            all_scores.append(score)
            all_labels.append(1)  # Genuine (same person)
        for score in inter_similarities:
            all_scores.append(score)
            all_labels.append(0)  # Impostor (different person)
        
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Calculate FAR/FRR for different thresholds
        thresholds = np.arange(0.5, 1.0, 0.01)
        best_threshold = 0.75
        best_eer = 1.0
        
        print("Threshold | FAR (False Accept) | FRR (False Reject) | EER")
        print("-" * 80)
        
        for threshold in thresholds:
            # FAR: inter-user scores >= threshold (wrongly accepted)
            far = np.mean((all_scores >= threshold) & (all_labels == 0))
            # FRR: intra-user scores < threshold (wrongly rejected)
            frr = np.mean((all_scores < threshold) & (all_labels == 1))
            eer = (far + frr) / 2.0
            
            if abs(far - frr) < abs(best_eer - (far + frr) / 2.0):
                best_threshold = threshold
                best_eer = eer
            
            if threshold in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
                print(f"  {threshold:.2f}   |      {far*100:6.2f}%        |      {frr*100:6.2f}%        | {eer*100:.2f}%")
        
        print(f"\n[RECOMMENDATION] Best threshold (EER): {best_threshold:.2f} (EER: {best_eer*100:.2f}%)")
        
        # Check if there's good separation
        intra_max = np.max(intra_similarities)
        inter_min = np.min(inter_similarities)
        separation = intra_max - inter_min
        
        print(f"\n[SEPARATION]:")
        print(f"  Highest intra-user similarity: {intra_max:.4f}")
        print(f"  Lowest inter-user similarity:  {inter_min:.4f}")
        print(f"  Separation gap: {separation:.4f}")
        
        if separation > 0.1:
            print(f"  [OK] GOOD: Clear separation between same/different people")
        elif separation > 0.05:
            print(f"  [WARN] WARNING: Moderate separation, may have some confusion")
        else:
            print(f"  [BAD] BAD: Poor separation, embeddings may not be distinguishable")
        
        # Check overlap
        intra_95th = np.percentile(intra_similarities, 95)
        inter_5th = np.percentile(inter_similarities, 5)
        overlap = max(0, intra_95th - inter_5th)
        
        print(f"\n  Intra-user 95th percentile: {intra_95th:.4f}")
        print(f"  Inter-user 5th percentile:  {inter_5th:.4f}")
        print(f"  Overlap: {overlap:.4f}")
        
        if overlap < 0.05:
            print(f"  [OK] GOOD: Minimal overlap")
        elif overlap < 0.15:
            print(f"  [WARN] WARNING: Some overlap, threshold selection is important")
        else:
            print(f"  [BAD] BAD: Significant overlap, embeddings may not be reliable")

if __name__ == "__main__":
    analyze_registry()

