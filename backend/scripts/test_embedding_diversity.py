"""
Test if Model A produces DIVERSE embeddings for DIFFERENT images.
This is the critical test - if all embeddings are the same, the model is broken.
"""

import torch
import numpy as np
import sys
sys.path.append('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend')

from app.pipelines.arcface_model import get_model

# Load model
print("Loading Model A...")
model = get_model(input_size=[112, 112], num_layers=100, mode='ir')
ckpt = torch.load('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend/app/models/modelA_best.pth', 
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

print("="*80)
print("CRITICAL TEST: Embedding Diversity for Different Inputs")
print("="*80)

# Test with 5 VERY different random images
embeddings = []
with torch.no_grad():
    for i in range(5):
        # Create very different patterns
        if i == 0:
            x = torch.zeros(1, 3, 112, 112)  # Black
        elif i == 1:
            x = torch.ones(1, 3, 112, 112)  # White
        elif i == 2:
            x = torch.randn(1, 3, 112, 112) * 0.1  # Small noise
        elif i == 3:
            x = torch.randn(1, 3, 112, 112) * 10  # Large noise
        else:
            x = torch.randn(1, 3, 112, 112)  # Medium noise
        
        emb = model(x)
        embeddings.append(emb)
        
        print(f"\nImage {i+1}:")
        print(f"  Input: mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")
        print(f"  Embedding: norm={torch.norm(emb):.4f}, mean={emb.mean():.4f}, std={emb.std():.4f}")
        print(f"  First 10 values: {emb[0, :10].tolist()}")

print("\n" + "="*80)
print("PAIRWISE COSINE SIMILARITIES")
print("="*80)
print("(Should be LOW for different images, typically < 0.5)")
print()

max_sim = 0
for i in range(5):
    for j in range(i+1, 5):
        cos_sim = torch.nn.functional.cosine_similarity(embeddings[i], embeddings[j], dim=1).item()
        print(f"Image {i+1} vs Image {j+1}: {cos_sim:.6f}", end="")
        if cos_sim > 0.8:
            print(" ❌ TOO SIMILAR!")
        elif cos_sim > 0.5:
            print(" ⚠️ Somewhat similar")
        else:
            print(" ✅ Good diversity")
        max_sim = max(max_sim, cos_sim)

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if max_sim > 0.8:
    print("❌ CRITICAL ERROR: Model produces nearly identical embeddings!")
    print("   All different images map to the same point in embedding space.")
    print()
    print("🔍 Root Cause Analysis:")
    print("   This suggests the model did NOT learn discriminative features.")
    print()
    print("💡 Possible Reasons:")
    print("   1. Model collapsed during training (loss didn't decrease properly)")
    print("   2. Learning rate was too high (weights diverged)")
    print("   3. BatchNorm stats not updated during training")
    print("   4. Training data preprocessing was wrong")
    print("   5. Loss function not working correctly")
    print()
    print("🔧 SOLUTION:")
    print("   1. Check training loss curve - did it decrease?")
    print("   2. Check validation accuracy - was it improving?")
    print("   3. Re-train with correct settings:")
    print("      - Use model.train() during training")
    print("      - Ensure loss.backward() is called")
    print("      - Check learning rate schedule")
    print("      - Verify preprocessing matches notebook")
elif max_sim > 0.5:
    print("⚠️  WARNING: Embeddings show high similarity")
    print("   Model may not be well-trained or needs more epochs.")
else:
    print("✅ SUCCESS: Model produces diverse embeddings!")
    print("   Different images map to different points in embedding space.")

print("="*80)
