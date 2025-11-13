import torch
import sys
sys.path.append('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend')

from app.pipelines.arcface_model import get_model

# Load model
model = get_model(input_size=[112, 112], num_layers=100, mode='ir')
ckpt = torch.load('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend/app/models/modelA_best.pth', 
                  map_location='cpu', weights_only=False)

print("="*80)
print("ANALYZING CHECKPOINT - What was saved?")
print("="*80)

print(f"\n1. Checkpoint keys: {list(ckpt.keys())}")
print(f"\n2. Epoch: {ckpt.get('epoch', 'N/A')}")
print(f"3. Best AUC: {ckpt.get('best_auc', 'N/A')}")

# Check if model weights have variance
print(f"\n4. Checking model weight variance:")
model_state = ckpt['model']

# Sample a few key layers
key_layers = [
    'conv1.weight',
    'layer1.0.conv1.weight',
    'layer2.0.conv1.weight',
    'layer3.0.conv1.weight',
    'layer4.0.conv1.weight',
    'fc.weight'
]

for key in key_layers:
    if key in model_state:
        weight = model_state[key]
        w_std = torch.std(weight).item()
        w_mean = torch.mean(weight).item()
        w_norm = torch.norm(weight).item()
        print(f"  {key:30s}: std={w_std:.6f}, mean={w_mean:.6f}, norm={w_norm:.6f}")
    else:
        print(f"  {key:30s}: NOT FOUND")

# Check if fc layer (embedding projection) has reasonable weights
if 'fc.weight' in model_state:
    fc_weight = model_state['fc.weight']
    print(f"\n5. FC layer (embedding projection):")
    print(f"  Shape: {fc_weight.shape}")
    print(f"  Std: {torch.std(fc_weight).item():.6f}")
    print(f"  First row sample: {fc_weight[0, :5].tolist()}")
    print(f"  Last row sample: {fc_weight[-1, :5].tolist()}")
    
    # Check if all rows are similar (indicates not trained)
    row_stds = torch.std(fc_weight, dim=1)
    print(f"  Row stds (should vary): min={row_stds.min().item():.6f}, max={row_stds.max().item():.6f}, mean={row_stds.mean().item():.6f}")

# Compare with pretrained weights
PRETRAIN_PATH = 'd:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend/app/models/ms1mv3_arcface_r100_fp16.pth'
try:
    pretrained = torch.load(PRETRAIN_PATH, map_location='cpu', weights_only=False)
    print(f"\n6. Comparing with pretrained weights:")
    
    # Check if they're the same (not trained)
    same_count = 0
    total_count = 0
    for key in ['conv1.weight', 'layer1.0.conv1.weight']:
        if key in model_state and key in pretrained:
            diff = torch.norm(model_state[key] - pretrained[key]).item()
            print(f"  {key:30s}: diff_norm={diff:.6f}")
            total_count += 1
            if diff < 1e-6:
                same_count += 1
    
    if same_count == total_count:
        print(f"\n  ❌ CRITICAL: Weights are IDENTICAL to pretrained!")
        print(f"     Model was NOT trained!")
    else:
        print(f"\n  ✅ Weights are different from pretrained (model was trained)")
        
except Exception as e:
    print(f"\n6. Could not load pretrained weights: {e}")

print("\n" + "="*80)
