"""
Find Optimal Threshold for Model A Face Verification

This script:
1. Loads test pairs from LFW or your validation set
2. Computes embeddings for all pairs
3. Calculates cosine similarity scores
4. Finds optimal threshold using ROC curve analysis
5. Reports: AUC, EER, best threshold, accuracy at different thresholds
"""

import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sys
sys.path.append('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend')

from app.pipelines.embedding import EmbedModel
import cv2
from pathlib import Path

print("="*80)
print("THRESHOLD OPTIMIZATION FOR MODEL A")
print("="*80)
print()

# L2 normalization (same as training)
def l2norm(x, eps=1e-12):
    """L2-normalize embeddings"""
    if isinstance(x, np.ndarray):
        return x / (np.linalg.norm(x) + eps)
    return x / (x.norm(p=2, dim=1, keepdim=True) + eps)

print("[1/5] Loading Model A...")
embed_model = EmbedModel(model_type="A")
print("   [OK] Model loaded")
print()

# ==============================================================================
# OPTION 1: Use synthetic test pairs (if you don't have real test data)
# ==============================================================================
def create_synthetic_test_pairs():
    """Create synthetic test pairs for threshold tuning"""
    print("[2/5] Creating synthetic test pairs...")
    print("   ⚠️  Using synthetic data - threshold may not be optimal for real faces")
    print("   💡 Recommendation: Use real face pairs from your validation set")
    print()
    
    pairs = []
    labels = []
    
    # Same person pairs (should have HIGH similarity)
    for i in range(20):
        # Create 2 slightly different versions of same "face"
        base = np.random.randint(100, 150, (112, 112, 3), dtype=np.uint8)
        noise1 = np.random.randint(-10, 10, (112, 112, 3), dtype=np.int16)
        noise2 = np.random.randint(-10, 10, (112, 112, 3), dtype=np.int16)
        
        img1 = np.clip(base + noise1, 0, 255).astype(np.uint8)
        img2 = np.clip(base + noise2, 0, 255).astype(np.uint8)
        
        pairs.append((img1, img2))
        labels.append(1)  # Same person
    
    # Different person pairs (should have LOW similarity)
    for i in range(20):
        img1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
        pairs.append((img1, img2))
        labels.append(0)  # Different person
    
    print(f"   ✓ Created {len(pairs)} test pairs ({sum(labels)} same, {len(labels)-sum(labels)} different)")
    return pairs, labels

# ==============================================================================
# OPTION 2: Load real test pairs from file (RECOMMENDED)
# ==============================================================================
def load_real_test_pairs(pairs_file):
    """
    Load real test pairs from a text file with format:
    path1 path2 label
    
    Example:
    /path/to/person1_img1.jpg /path/to/person1_img2.jpg 1
    /path/to/person1_img1.jpg /path/to/person2_img1.jpg 0
    """
    print(f"[2/5] Loading real test pairs from {pairs_file}...")
    
    if not Path(pairs_file).exists():
        print(f"   ❌ File not found: {pairs_file}")
        print("   Falling back to synthetic pairs...")
        return create_synthetic_test_pairs()
    
    pairs = []
    labels = []
    
    with open(pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            
            path1, path2, label = parts[0], parts[1], int(parts[2])
            
            # Load images
            img1 = cv2.imread(path1)
            img2 = cv2.imread(path2)
            
            if img1 is None or img2 is None:
                continue
            
            # Resize to 112x112
            img1 = cv2.resize(img1, (112, 112))
            img2 = cv2.resize(img2, (112, 112))
            
            pairs.append((img1, img2))
            labels.append(label)
    
    print(f"   ✓ Loaded {len(pairs)} test pairs ({sum(labels)} same, {len(labels)-sum(labels)} different)")
    return pairs, labels

# ==============================================================================
# Choose data source
# ==============================================================================
# Try to load real pairs first, fallback to synthetic
test_pairs_file = "d:/Swinburne/COS30082 - Applied Machine Learning/Project/Test/test_pairs.txt"
pairs, labels = load_real_test_pairs(test_pairs_file)

# If no real data, use synthetic
if len(pairs) == 0:
    pairs, labels = create_synthetic_test_pairs()

print()

# ==============================================================================
# Compute similarities
# ==============================================================================
print("[3/5] Computing embeddings and similarities...")
similarities = []

for idx, (img1, img2) in enumerate(pairs):
    # Extract embeddings (already L2-normalized by embedding.py)
    emb1 = embed_model.extract(img1)
    emb2 = embed_model.extract(img2)
    
    # Cosine similarity (dot product of normalized vectors)
    sim = np.dot(emb1, emb2)
    similarities.append(sim)
    
    if (idx + 1) % 10 == 0:
        print(f"   Processed {idx + 1}/{len(pairs)} pairs...")

similarities = np.array(similarities)
labels = np.array(labels)

print(f"   ✓ Computed {len(similarities)} similarity scores")
print()

# ==============================================================================
# ROC Curve Analysis
# ==============================================================================
print("[4/5] Computing ROC curve and optimal threshold...")

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(labels, similarities)
roc_auc = auc(fpr, tpr)

# Find optimal threshold (Youden's J statistic: max(TPR - FPR))
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
best_tpr = tpr[best_idx]
best_fpr = fpr[best_idx]

# Equal Error Rate (EER) - where FPR = FNR
fnr = 1 - tpr
eer_idx = np.argmin(np.abs(fpr - fnr))
eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
eer_threshold = thresholds[eer_idx]

print(f"   ✓ ROC-AUC: {roc_auc:.4f}")
print()

# ==============================================================================
# Results
# ==============================================================================
print("[5/5] RESULTS:")
print("="*80)
print()

print("📊 ROC CURVE ANALYSIS:")
print("-" * 80)
print(f"AUC (Area Under Curve):     {roc_auc:.4f}")
print(f"Equal Error Rate (EER):     {eer:.4f} (at threshold {eer_threshold:.4f})")
print(f"Optimal Threshold (Youden): {best_threshold:.4f}")
print(f"  - True Positive Rate:     {best_tpr:.4f}")
print(f"  - False Positive Rate:    {best_fpr:.4f}")
print(f"  - Accuracy at threshold:  {(best_tpr * sum(labels) + (1-best_fpr) * (len(labels)-sum(labels))) / len(labels):.4f}")
print("-" * 80)
print()

print("📈 THRESHOLD RECOMMENDATIONS:")
print("-" * 80)

# Test different thresholds
test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, best_threshold]
test_thresholds = sorted(set(test_thresholds))

for thr in test_thresholds:
    predictions = (similarities >= thr).astype(int)
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    marker = " ← RECOMMENDED" if abs(thr - best_threshold) < 0.01 else ""
    print(f"Threshold {thr:.3f}: Acc={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}{marker}")

print("-" * 80)
print()

print("💡 RECOMMENDATIONS FOR PRODUCTION:")
print("-" * 80)
print(f"✅ Use threshold: {best_threshold:.3f} (optimal from ROC curve)")
print()
print("Adjust based on your requirements:")
print(f"  • Higher threshold (e.g., {min(best_threshold + 0.1, 0.9):.3f}): Fewer false positives, stricter matching")
print(f"  • Lower threshold (e.g., {max(best_threshold - 0.1, 0.1):.3f}): Fewer false negatives, more lenient matching")
print()
print("Update in: Veriface/backend/app/core/thresholds.yaml")
print("="*80)

# ==============================================================================
# Plot ROC Curve
# ==============================================================================
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
plt.scatter([best_fpr], [best_tpr], color='red', s=100, zorder=5, 
            label=f'Optimal threshold = {best_threshold:.3f}')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Model A Face Verification', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = "model_A_roc_curve.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n📊 ROC curve saved to: {output_path}")
plt.show()
