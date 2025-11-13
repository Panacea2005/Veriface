import torch
import numpy as np
import sys
sys.path.append('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend')

from app.pipelines.arcface_model import get_model

# Load model
model = get_model(input_size=[112, 112], num_layers=100, mode='ir')
ckpt = torch.load('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend/app/models/modelA_best.pth', 
                  map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

# Test with 5 different random inputs
print("="*80)
print("Testing Model A - 5 Different Random Inputs")
print("="*80)

embeddings = []
for i in range(5):
    x = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        y = model(x)
        y_np = y.cpu().numpy().flatten()
        embeddings.append(y_np)
        
    print(f"\nInput {i+1}:")
    print(f"  Output norm: {torch.norm(y).item():.6f}")
    print(f"  Output std: {torch.std(y).item():.6f}")
    print(f"  Output mean: {torch.mean(y).item():.6f}")
    print(f"  Output sample (first 5): {y_np[:5]}")

# Compute pairwise cosine similarities
print("\n" + "="*80)
print("Pairwise Cosine Similarities (should be LOW < 0.3 for different inputs)")
print("="*80)

from sklearn.metrics.pairwise import cosine_similarity
embeddings_matrix = np.array(embeddings)
cos_sim_matrix = cosine_similarity(embeddings_matrix)

for i in range(5):
    for j in range(i+1, 5):
        print(f"Input {i+1} vs Input {j+1}: {cos_sim_matrix[i, j]:.6f}")

print("\n" + "="*80)
if cos_sim_matrix[0, 1] > 0.9:
    print("❌ PROBLEM DETECTED: Embeddings are too similar (> 0.9)")
    print("   Model is NOT working correctly - outputs are nearly identical!")
else:
    print("✅ Model is working correctly - outputs are different")
print("="*80)
