"""
Test Anti-Spoof Threshold with DeepFace MiniFASNet

This script helps you find the optimal anti-spoof threshold by:
1. Testing with REAL webcam faces
2. Testing with SPOOF attacks (photos, screens)
3. Computing scores and recommending threshold

IMPORTANT: You need to prepare test images:
- Real faces from webcam
- Printed photos of faces
- Face displayed on phone/screen
- Different lighting conditions
"""

import numpy as np
import cv2
import sys
sys.path.append('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend')

from app.pipelines.liveness import LivenessModel
from pathlib import Path
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

print("="*80)
print("ANTI-SPOOF THRESHOLD OPTIMIZATION")
print("Using DeepFace MiniFASNet (Silent-Face-Anti-Spoofing)")
print("="*80)
print()

# Initialize liveness model
print("[1/4] Loading DeepFace MiniFASNet...")
liveness_model = LivenessModel()
print("   [OK] Model loaded")
print()

# ==============================================================================
# Data Collection
# ==============================================================================
print("[2/4] Collecting test images...")
print()
print("INSTRUCTIONS:")
print("-" * 80)
print("1. Create folders:")
print("   - Test/real_faces/     <- Put webcam captures of REAL faces here")
print("   - Test/spoof_attacks/  <- Put printed photos, screen displays here")
print()
print("2. Capture test images:")
print("   Real faces: 20+ images from webcam with different:")
print("     - Lighting conditions (bright, dim, natural)")
print("     - Face angles (frontal, slight turns)")
print("     - Expressions (neutral, smile, serious)")
print()
print("   Spoof attacks: 20+ images with:")
print("     - Printed photos held in front of webcam")
print("     - Face displayed on phone screen")
print("     - Face displayed on laptop screen")
print("     - Different photo qualities (good, poor)")
print()
print("3. Run this script to find optimal threshold")
print("-" * 80)
print()

# Check if test folders exist
real_faces_dir = Path("d:/Swinburne/COS30082 - Applied Machine Learning/Project/Test/real_faces")
spoof_attacks_dir = Path("d:/Swinburne/COS30082 - Applied Machine Learning/Project/Test/spoof_attacks")

if not real_faces_dir.exists():
    real_faces_dir.mkdir(parents=True, exist_ok=True)
    print(f"[WARN] Created folder: {real_faces_dir}")
    print("       Please add real face images and run again")

if not spoof_attacks_dir.exists():
    spoof_attacks_dir.mkdir(parents=True, exist_ok=True)
    print(f"[WARN] Created folder: {spoof_attacks_dir}")
    print("       Please add spoof attack images and run again")

# Load test images
real_images = list(real_faces_dir.glob("*.jpg")) + list(real_faces_dir.glob("*.png"))
spoof_images = list(spoof_attacks_dir.glob("*.jpg")) + list(spoof_attacks_dir.glob("*.png"))

print(f"Found: {len(real_images)} real face images")
print(f"Found: {len(spoof_images)} spoof attack images")
print()

if len(real_images) < 5 or len(spoof_images) < 5:
    print("[ERROR] Need at least 5 images of each type!")
    print()
    print("QUICK START - Use webcam to capture test images:")
    print("-" * 80)
    print("Real faces:")
    print("  1. Open webcam")
    print("  2. Capture 20 photos of yourself")
    print("  3. Save to Test/real_faces/")
    print()
    print("Spoof attacks:")
    print("  1. Print a photo of yourself (or display on phone)")
    print("  2. Hold it in front of webcam")
    print("  3. Capture 20 photos")
    print("  4. Save to Test/spoof_attacks/")
    print("-" * 80)
    sys.exit(1)

# ==============================================================================
# Compute Anti-Spoof Scores
# ==============================================================================
print("[3/4] Computing anti-spoof scores...")
print()

real_scores = []
spoof_scores = []

print("Testing REAL faces...")
for idx, img_path in enumerate(real_images):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    result = liveness_model.predict(img)
    score = result.get('score', 0.0)
    real_scores.append(score)
    
    status = "PASS" if score >= 0.5 else "FAIL"
    print(f"  [{idx+1}/{len(real_images)}] {img_path.name}: score={score:.4f} [{status}]")

print()
print("Testing SPOOF attacks...")
for idx, img_path in enumerate(spoof_images):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    
    result = liveness_model.predict(img)
    score = result.get('score', 0.0)
    spoof_scores.append(score)
    
    status = "BLOCKED" if score < 0.5 else "LEAKED"
    print(f"  [{idx+1}/{len(spoof_images)}] {img_path.name}: score={score:.4f} [{status}]")

print()
print("-" * 80)

# ==============================================================================
# Statistical Analysis
# ==============================================================================
print("[4/4] RESULTS & RECOMMENDATIONS:")
print("="*80)
print()

real_scores = np.array(real_scores)
spoof_scores = np.array(spoof_scores)

print("[STATS] SCORE DISTRIBUTIONS:")
print("-" * 80)
print(f"Real Faces:")
print(f"  Min:    {real_scores.min():.4f}")
print(f"  Max:    {real_scores.max():.4f}")
print(f"  Mean:   {real_scores.mean():.4f}")
print(f"  Median: {np.median(real_scores):.4f}")
print(f"  Std:    {real_scores.std():.4f}")
print()
print(f"Spoof Attacks:")
print(f"  Min:    {spoof_scores.min():.4f}")
print(f"  Max:    {spoof_scores.max():.4f}")
print(f"  Mean:   {spoof_scores.mean():.4f}")
print(f"  Median: {np.median(spoof_scores):.4f}")
print(f"  Std:    {spoof_scores.std():.4f}")
print("-" * 80)
print()

# Compute ROC curve if we have enough samples
if len(real_scores) >= 10 and len(spoof_scores) >= 10:
    # Prepare labels (1=real, 0=spoof)
    scores = np.concatenate([real_scores, spoof_scores])
    labels = np.concatenate([np.ones(len(real_scores)), np.zeros(len(spoof_scores))])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (maximize TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[best_idx]
    
    # Equal Error Rate
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    eer_threshold = thresholds[eer_idx]
    
    print("[ANALYSIS] ROC CURVE ANALYSIS:")
    print("-" * 80)
    print(f"AUC (Area Under Curve):  {roc_auc:.4f}")
    print(f"Equal Error Rate (EER):  {eer:.4f} (at threshold {eer_threshold:.4f})")
    print(f"Optimal Threshold:       {optimal_threshold:.4f}")
    print(f"  - True Positive Rate:  {tpr[best_idx]:.4f} (real faces accepted)")
    print(f"  - False Positive Rate: {fpr[best_idx]:.4f} (spoofs leaked)")
    print("-" * 80)
    print()

# Test current threshold (0.5)
current_threshold = 0.5
real_pass_rate = np.sum(real_scores >= current_threshold) / len(real_scores)
spoof_block_rate = np.sum(spoof_scores < current_threshold) / len(spoof_scores)

print("[ANALYSIS] CURRENT THRESHOLD (0.5):")
print("-" * 80)
print(f"Real faces PASSED:     {real_pass_rate*100:.1f}% ({int(real_pass_rate*len(real_scores))}/{len(real_scores)})")
print(f"Spoofs BLOCKED:        {spoof_block_rate*100:.1f}% ({int(spoof_block_rate*len(spoof_scores))}/{len(spoof_scores)})")
print(f"False Positive Rate:   {(1-spoof_block_rate)*100:.1f}% (spoofs leaked)")
print(f"False Negative Rate:   {(1-real_pass_rate)*100:.1f}% (real faces rejected)")
print("-" * 80)
print()

# Recommendations
print("[TIP] THRESHOLD RECOMMENDATIONS:")
print("-" * 80)

test_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
if len(real_scores) >= 10 and len(spoof_scores) >= 10:
    test_thresholds.append(optimal_threshold)
test_thresholds = sorted(set(test_thresholds))

for thr in test_thresholds:
    real_pass = np.sum(real_scores >= thr) / len(real_scores)
    spoof_block = np.sum(spoof_scores < thr) / len(spoof_scores)
    
    marker = ""
    if len(real_scores) >= 10 and abs(thr - optimal_threshold) < 0.01:
        marker = " <- OPTIMAL"
    elif thr == 0.5:
        marker = " <- CURRENT"
    
    print(f"Threshold {thr:.2f}: Real pass {real_pass*100:.1f}%, Spoof block {spoof_block*100:.1f}%{marker}")

print("-" * 80)
print()

print("[PASS] FINAL RECOMMENDATIONS:")
print("="*80)

if len(real_scores) >= 10 and len(spoof_scores) >= 10:
    print(f"Recommended threshold: {optimal_threshold:.2f}")
    print()
    print("Update in: Veriface/backend/app/core/thresholds.yaml")
    print(f"  anti_spoof:")
    print(f"    threshold: {optimal_threshold:.2f}")
else:
    print("Recommended threshold: 0.50 (default, need more test data)")
    print()
    print("Collect more test images for better threshold tuning:")
    print("  - At least 20 real face images")
    print("  - At least 20 spoof attack images")

print()
print("Adjust based on your requirements:")
print("  - Higher threshold (0.6-0.7): Stricter, fewer spoofs leaked")
print("  - Lower threshold (0.3-0.4): More lenient, fewer real faces rejected")
print("="*80)

# Plot distribution
plt.figure(figsize=(12, 5))

# Subplot 1: Score distributions
plt.subplot(1, 2, 1)
plt.hist(real_scores, bins=20, alpha=0.7, label='Real Faces', color='green', edgecolor='black')
plt.hist(spoof_scores, bins=20, alpha=0.7, label='Spoof Attacks', color='red', edgecolor='black')
plt.axvline(x=0.5, color='blue', linestyle='--', linewidth=2, label='Current Threshold (0.5)')
if len(real_scores) >= 10 and len(spoof_scores) >= 10:
    plt.axvline(x=optimal_threshold, color='orange', linestyle='--', linewidth=2, label=f'Optimal ({optimal_threshold:.2f})')
plt.xlabel('Anti-Spoof Score', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Score Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: ROC curve
if len(real_scores) >= 10 and len(spoof_scores) >= 10:
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.scatter([fpr[best_idx]], [tpr[best_idx]], color='red', s=100, zorder=5, 
                label=f'Optimal threshold = {optimal_threshold:.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Spoof Leaked)', fontsize=12)
    plt.ylabel('True Positive Rate (Real Accepted)', fontsize=12)
    plt.title('ROC Curve - Anti-Spoof Detection', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
output_path = "antispoof_threshold_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n[PASS] Analysis plot saved to: {output_path}")
plt.show()
