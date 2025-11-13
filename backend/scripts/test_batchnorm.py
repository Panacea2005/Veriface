import torch
import numpy as np
import sys
sys.path.append('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend')

from app.pipelines.arcface_model import get_model

# Load model
model = get_model(input_size=[112, 112], num_layers=100, mode='ir')
ckpt = torch.load('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend/app/models/modelA_best.pth', 
                  map_location='cpu', weights_only=False)

print("="*80)
print("ANALYZING MODEL A - BatchNorm Statistics")
print("="*80)

# Check BatchNorm layers BEFORE loading weights
print("\n1. BatchNorm layers in fresh model:")
bn_count = 0
for name, m in model.named_modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        bn_count += 1
        if bn_count <= 3:  # Show first 3
            print(f"  {name}: track_running_stats={m.track_running_stats}, running_mean exists={m.running_mean is not None}")

print(f"  Total BatchNorm layers: {bn_count}")

# Load state dict
missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
print(f"\n2. Loading checkpoint:")
print(f"  Missing keys: {len(missing)}")
print(f"  Unexpected keys: {len(unexpected)}")

# Check BatchNorm stats AFTER loading
print(f"\n3. BatchNorm statistics AFTER loading:")
zero_mean_count = 0
for name, m in model.named_modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        if m.running_mean is not None:
            mean_norm = torch.norm(m.running_mean).item()
            if mean_norm < 1e-6:
                zero_mean_count += 1
                if zero_mean_count <= 3:
                    print(f"  ❌ {name}: running_mean norm = {mean_norm:.10f} (ZERO!)")
            elif zero_mean_count == 0:
                # Show first good one
                print(f"  ✅ {name}: running_mean norm = {mean_norm:.6f}")
                zero_mean_count = -1  # Flag to stop showing good ones

print(f"\n4. Summary:")
print(f"  Total BatchNorm layers: {bn_count}")
print(f"  Layers with zero running_mean: {zero_mean_count if zero_mean_count > 0 else 'checking...'}")

# Final check - are running stats being used?
model.eval()
print(f"\n5. Model eval mode check:")
for name, m in list(model.named_modules())[:5]:
    if isinstance(m, torch.nn.BatchNorm2d):
        print(f"  {name}: training={m.training}, track_running_stats={m.track_running_stats}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
if zero_mean_count > 0:
    print("❌ BatchNorm running stats are ZERO or not properly saved during training!")
    print("   This causes the model to output identical embeddings for all inputs.")
    print("\n💡 SOLUTION:")
    print("   1. Re-train the model ensuring BatchNorm stats are tracked")
    print("   2. Make sure model.train() is called during training")
    print("   3. Don't use torch.no_grad() during training forward pass")
else:
    print("✅ BatchNorm stats look normal")
print("="*80)
