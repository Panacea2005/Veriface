import torch
import sys
sys.path.append('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend')

from app.pipelines.arcface_model import get_model

# Load model
model = get_model(input_size=[112, 112], num_layers=100, mode='ir')
ckpt = torch.load('d:/Swinburne/COS30082 - Applied Machine Learning/Project/Veriface/backend/app/models/modelA_best.pth', 
                  map_location='cpu', weights_only=False)

print("="*80)
print("ANALYZING MISSING KEYS IN MODEL A")
print("="*80)

# Load and get missing keys
missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)

print(f"\n1. Missing keys count: {len(missing)}")
print(f"2. Unexpected keys count: {len(unexpected)}")

print(f"\n3. Sample of missing keys (first 20):")
for i, key in enumerate(missing[:20]):
    print(f"   {i+1}. {key}")

print(f"\n4. What types of layers are missing?")
missing_types = {}
for key in missing:
    # Extract layer type (last part of key)
    layer_type = key.split('.')[-1]
    missing_types[layer_type] = missing_types.get(layer_type, 0) + 1

for layer_type, count in sorted(missing_types.items(), key=lambda x: x[1], reverse=True):
    print(f"   {layer_type}: {count} missing")

print(f"\n5. Keys in checkpoint (first 30):")
ckpt_keys = list(ckpt['model'].keys())
for i, key in enumerate(ckpt_keys[:30]):
    print(f"   {i+1}. {key}")

print(f"\n6. Keys expected by model (first 30):")
model_keys = list(model.state_dict().keys())
for i, key in enumerate(model_keys[:30]):
    print(f"   {i+1}. {key}")

print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)

# Check if it's a key naming mismatch
if 'fc.weight' in ckpt_keys and 'fc.weight' in missing:
    print("❌ CRITICAL: The model has 'fc' layer but checkpoint is missing it!")
elif 'features.fc.weight' in ckpt_keys and 'fc.weight' in missing:
    print("❌ Key mismatch: checkpoint uses 'features.fc' but model expects 'fc'")
else:
    print("⚠️  Check the missing keys - likely BatchNorm num_batches_tracked")
    print("    This is usually OK for inference, but indicates a version mismatch")

# Check if the critical embedding layer is loaded
fc_weight_loaded = 'fc.weight' not in missing
print(f"\n✅ Final FC layer (embedding) loaded: {fc_weight_loaded}")
if not fc_weight_loaded:
    print("❌ CRITICAL ERROR: Embedding layer NOT loaded!")
    print("   This explains why all embeddings are identical!")

print("="*80)
