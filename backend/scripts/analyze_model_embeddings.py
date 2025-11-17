"""Analyze model architecture and embedding quality.

This script:
1. Loads modelB_best.pth and inspects all layers
2. Tests embeddings on sample images from the same person
3. Tests embeddings on sample images from different people
4. Computes cosine similarity to verify embedding quality
"""
from __future__ import annotations

import sys
from pathlib import Path
import os

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.pipelines.arcface_model import get_model

# ============================================================================
# Configuration
# ============================================================================
MODELS_DIR = BACKEND_ROOT / "app" / "models"
MODEL_PATH = MODELS_DIR / "modelB_best.pth"

print("="*80)
print("MODEL B ARCHITECTURE & EMBEDDING QUALITY ANALYSIS")
print("="*80)
print()

# ============================================================================
# 1. Load Model and Inspect Architecture
# ============================================================================
print("1. LOADING MODEL B")
print("-"*80)

if not MODEL_PATH.exists():
    print(f"[ERROR] Model not found at: {MODEL_PATH}")
    sys.exit(1)

size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
print(f"Model file: {MODEL_PATH.name}")
print(f"File size:  {size_mb:.1f} MB")
print()

# Load model
model = get_model(input_size=[112, 112], num_layers=100, mode="ir")
checkpoint = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)

if isinstance(checkpoint, dict) and not ("state_dict" in checkpoint or "model" in checkpoint):
    state_dict = checkpoint
else:
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"[OK] Model loaded")
print(f"     Missing keys:    {len(missing)}")
print(f"     Unexpected keys: {len(unexpected)}")
print()

# ============================================================================
# 2. Inspect Model Architecture
# ============================================================================
print("2. MODEL ARCHITECTURE INSPECTION")
print("-"*80)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
print()

# List all modules
print("Model Structure (first 30 layers):")
print("-"*80)
for i, (name, module) in enumerate(model.named_modules()):
    if i >= 30:  # Limit output
        print("... (truncated, total modules:", len(list(model.named_modules())), ")")
        break
    module_type = module.__class__.__name__
    if hasattr(module, 'weight') and module.weight is not None:
        shape = tuple(module.weight.shape)
        print(f"{name:40s} {module_type:20s} {str(shape):30s}")
    else:
        print(f"{name:40s} {module_type:20s}")
print()

# Check for specific layers
print("Key Layers:")
print("-"*80)
for name, module in model.named_modules():
    if 'fc' in name.lower() or 'head' in name.lower() or 'output' in name.lower():
        module_type = module.__class__.__name__
        if hasattr(module, 'weight') and module.weight is not None:
            shape = tuple(module.weight.shape)
            print(f"{name:40s} {module_type:20s} {str(shape):30s}")
print()

# ============================================================================
# 3. Test Embedding Generation
# ============================================================================
print("3. EMBEDDING GENERATION TEST")
print("-"*80)

model.eval()

# Generate random test images
num_same_person = 5
num_diff_people = 10

print(f"Generating embeddings for {num_same_person} images (same person simulation)")
print(f"Generating embeddings for {num_diff_people} images (different people simulation)")
print()

with torch.no_grad():
    # Simulate same person: add small noise to base image
    base_image = torch.randn(1, 3, 112, 112)
    same_person_images = [base_image + torch.randn(1, 3, 112, 112) * 0.1 for _ in range(num_same_person)]
    
    # Simulate different people: completely random images
    diff_people_images = [torch.randn(1, 3, 112, 112) for _ in range(num_diff_people)]
    
    # Get embeddings
    same_embeddings = []
    for img in same_person_images:
        emb = model(img)
        same_embeddings.append(emb)
    
    diff_embeddings = []
    for img in diff_people_images:
        emb = model(img)
        diff_embeddings.append(emb)

print(f"[OK] Generated embeddings")
print(f"     Embedding shape: {same_embeddings[0].shape}")
print(f"     Embedding dim:   {same_embeddings[0].shape[1]}")
print()

# ============================================================================
# 4. Embedding Statistics
# ============================================================================
print("4. EMBEDDING STATISTICS")
print("-"*80)

# Check embedding norms
same_norms = [emb.norm(p=2, dim=1).item() for emb in same_embeddings]
diff_norms = [emb.norm(p=2, dim=1).item() for emb in diff_embeddings]

print("Embedding L2 Norms (should be ~1.0 if normalized):")
print(f"  Same person:      min={min(same_norms):.4f}, max={max(same_norms):.4f}, avg={np.mean(same_norms):.4f}")
print(f"  Different people: min={min(diff_norms):.4f}, max={max(diff_norms):.4f}, avg={np.mean(diff_norms):.4f}")
print()

# Check if embeddings are normalized
if abs(np.mean(same_norms) - 1.0) < 0.01 and abs(np.mean(diff_norms) - 1.0) < 0.01:
    print("[OK] Embeddings are L2-normalized (model includes normalization layer)")
else:
    print("[WARNING] Embeddings are NOT normalized - may need to add F.normalize()")
print()

# ============================================================================
# 5. Cosine Similarity Analysis
# ============================================================================
print("5. COSINE SIMILARITY ANALYSIS")
print("-"*80)

# Normalize embeddings for fair comparison
same_emb_norm = [F.normalize(emb, p=2, dim=1) for emb in same_embeddings]
diff_emb_norm = [F.normalize(emb, p=2, dim=1) for emb in diff_embeddings]

# Within same person similarities
same_person_sims = []
for i in range(len(same_emb_norm)):
    for j in range(i+1, len(same_emb_norm)):
        sim = (same_emb_norm[i] * same_emb_norm[j]).sum().item()
        same_person_sims.append(sim)

# Between different people similarities
diff_people_sims = []
for i in range(len(diff_emb_norm)):
    for j in range(i+1, len(diff_emb_norm)):
        sim = (diff_emb_norm[i] * diff_emb_norm[j]).sum().item()
        diff_people_sims.append(sim)

print("Cosine Similarity Statistics:")
print(f"  Same person pairs ({len(same_person_sims)} pairs):")
print(f"    min={min(same_person_sims):.4f}, max={max(same_person_sims):.4f}, avg={np.mean(same_person_sims):.4f}, std={np.std(same_person_sims):.4f}")
print()
print(f"  Different people pairs ({len(diff_people_sims)} pairs):")
print(f"    min={min(diff_people_sims):.4f}, max={max(diff_people_sims):.4f}, avg={np.mean(diff_people_sims):.4f}, std={np.std(diff_people_sims):.4f}")
print()

# ============================================================================
# 6. Discrimination Analysis
# ============================================================================
print("6. DISCRIMINATION ANALYSIS")
print("-"*80)

# Check if same-person similarities are higher than different-person
avg_same = np.mean(same_person_sims)
avg_diff = np.mean(diff_people_sims)
separation = avg_same - avg_diff

print(f"Average same-person similarity:      {avg_same:.4f}")
print(f"Average different-people similarity: {avg_diff:.4f}")
print(f"Separation (same - diff):            {separation:.4f}")
print()

if separation > 0.1:
    print("[OK] Good separation between same/different person embeddings")
    print("     Model can discriminate between identities")
elif separation > 0.0:
    print("[WARNING] Weak separation between same/different person embeddings")
    print("          Model may struggle with hard cases")
else:
    print("[ERROR] No separation - embeddings are random!")
    print("        Model is not learning meaningful features")
print()

# Check overlap
same_min = min(same_person_sims)
diff_max = max(diff_people_sims)
overlap = diff_max - same_min

if overlap > 0:
    print(f"[WARNING] Similarity ranges overlap by {overlap:.4f}")
    print(f"          Same-person min: {same_min:.4f}")
    print(f"          Diff-people max: {diff_max:.4f}")
    print(f"          Overlap indicates some ambiguous cases")
else:
    print(f"[OK] No overlap between same/different distributions")
    print(f"     Same-person min: {same_min:.4f}")
    print(f"     Diff-people max: {diff_max:.4f}")
print()

# ============================================================================
# 7. Distribution Visualization (Text-based)
# ============================================================================
print("7. SIMILARITY DISTRIBUTION (Text Histogram)")
print("-"*80)

def text_histogram(values, label, bins=20, width=50):
    """Create a simple text-based histogram"""
    hist, edges = np.histogram(values, bins=bins)
    max_count = max(hist)
    
    print(f"{label}:")
    for i in range(len(hist)):
        bar_len = int((hist[i] / max_count) * width)
        bar = '█' * bar_len
        print(f"  [{edges[i]:.2f} to {edges[i+1]:.2f}]: {bar} ({hist[i]})")
    print()

text_histogram(same_person_sims, "Same Person Similarities", bins=10, width=40)
text_histogram(diff_people_sims, "Different People Similarities", bins=10, width=40)

# ============================================================================
# 8. Summary
# ============================================================================
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Model:                {MODEL_PATH.name}")
print(f"Size:                 {size_mb:.1f} MB")
print(f"Parameters:           {total_params/1e6:.2f}M")
print(f"Embedding dim:        {same_embeddings[0].shape[1]}")
print(f"Normalized:           {'Yes' if abs(np.mean(same_norms) - 1.0) < 0.01 else 'No'}")
print(f"Same-person sim:      {avg_same:.4f} ± {np.std(same_person_sims):.4f}")
print(f"Diff-people sim:      {avg_diff:.4f} ± {np.std(diff_people_sims):.4f}")
print(f"Separation:           {separation:.4f}")
print(f"Discrimination:       {'Good' if separation > 0.1 else 'Weak' if separation > 0.0 else 'None'}")
print("="*80)

sys.exit(0)
