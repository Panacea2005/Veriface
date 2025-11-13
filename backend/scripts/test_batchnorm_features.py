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

print("="*80)
print("ANALYZING BatchNorm 'features' layer (final BN after FC)")
print("="*80)

# Check the final BatchNorm1d layer
if hasattr(model, 'features'):
    bn_features = model.features
    print(f"\n1. Features layer type: {type(bn_features)}")
    print(f"   Parameters:")
    print(f"     weight shape: {bn_features.weight.shape}")
    print(f"     weight: {bn_features.weight.data[:10]}")
    print(f"     bias shape: {bn_features.bias.shape}")
    print(f"     bias: {bn_features.bias.data[:10]}")
    print(f"     running_mean: {bn_features.running_mean[:10]}")
    print(f"     running_var: {bn_features.running_var[:10]}")
    
    # Check if running stats are being used
    print(f"\n2. BatchNorm settings:")
    print(f"     track_running_stats: {bn_features.track_running_stats}")
    print(f"     affine: {bn_features.affine}")
    print(f"     eps: {bn_features.eps}")
    print(f"     momentum: {bn_features.momentum}")
    
    # Critical check: are running_mean and running_var reasonable?
    mean_norm = torch.norm(bn_features.running_mean).item()
    var_mean = torch.mean(bn_features.running_var).item()
    var_std = torch.std(bn_features.running_var).item()
    
    print(f"\n3. Running statistics quality:")
    print(f"     running_mean norm: {mean_norm:.6f}")
    print(f"     running_var mean: {var_mean:.6f}")
    print(f"     running_var std: {var_std:.6f}")
    
    if mean_norm < 1e-3:
        print(f"     ❌ WARNING: running_mean is near zero!")
    if var_mean < 0.5 or var_mean > 2.0:
        print(f"     ❌ WARNING: running_var is abnormal (should be ~1.0)")
    
    # Test the model
    print(f"\n4. Testing model with features layer:")
    model.eval()
    
    x1 = torch.randn(1, 3, 112, 112)
    x2 = torch.randn(1, 3, 112, 112)
    
    with torch.no_grad():
        # Forward pass through whole model
        y1 = model(x1)
        y2 = model(x2)
        
        print(f"     Output 1: norm={torch.norm(y1).item():.6f}, mean={torch.mean(y1).item():.6f}, std={torch.std(y1).item():.6f}")
        print(f"     Output 2: norm={torch.norm(y2).item():.6f}, mean={torch.mean(y2).item():.6f}, std={torch.std(y2).item():.6f}")
        
        cos_sim = torch.nn.functional.cosine_similarity(y1, y2, dim=1).item()
        print(f"     Cosine similarity: {cos_sim:.6f}")
        
        if cos_sim > 0.9:
            print(f"     ❌ Model outputs are too similar!")
            
            # Debug: check intermediate outputs
            print(f"\n5. Debugging intermediate outputs:")
            
            # Hook to capture FC output (before features BN)
            fc_outputs = []
            def hook_fn(module, input, output):
                fc_outputs.append(output.clone())
            
            hook = model.fc.register_forward_hook(hook_fn)
            
            fc_outputs.clear()
            y1_test = model(x1)
            fc_out1 = fc_outputs[0]
            
            fc_outputs.clear()
            y2_test = model(x2)
            fc_out2 = fc_outputs[0]
            
            hook.remove()
            
            print(f"     FC output 1 (before BN): norm={torch.norm(fc_out1).item():.6f}, std={torch.std(fc_out1).item():.6f}")
            print(f"     FC output 2 (before BN): norm={torch.norm(fc_out2).item():.6f}, std={torch.std(fc_out2).item():.6f}")
            fc_cos = torch.nn.functional.cosine_similarity(fc_out1, fc_out2, dim=1).item()
            print(f"     FC outputs cosine similarity: {fc_cos:.6f}")
            
            if fc_cos < 0.5 and cos_sim > 0.9:
                print(f"\n     ❌ FOUND THE PROBLEM!")
                print(f"        FC outputs are DIFFERENT ({fc_cos:.3f})")
                print(f"        But final outputs are IDENTICAL ({cos_sim:.3f})")
                print(f"        => BatchNorm 'features' layer is DESTROYING the embeddings!")
                print(f"        => running_mean/running_var are NOT properly learned!")

print("\n" + "="*80)
