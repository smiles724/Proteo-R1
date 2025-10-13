"""
Verify that encoder weights are properly loaded from model.safetensors
"""

import torch
from proteinLLM_pllm import PLLM

print("Loading model...")
model = PLLM.from_pretrained("./pllm").to("cpu")

print("\nChecking protein encoder weights:")
sample_weight = None
for name, param in model.protein_encoder.named_parameters():
    if 'weight' in name and param.numel() > 100:  # Get a substantial weight tensor
        sample_weight = param
        weight_name = name
        break

if sample_weight is not None:
    print(f"  Parameter: {weight_name}")
    print(f"  Shape: {sample_weight.shape}")
    print(f"  Mean: {sample_weight.mean().item():.6f}")
    print(f"  Std: {sample_weight.std().item():.6f}")
    print(f"  Min: {sample_weight.min().item():.6f}")
    print(f"  Max: {sample_weight.max().item():.6f}")
    
    # Check if weights are non-trivial (not all zeros or random uniform)
    if abs(sample_weight.mean().item()) > 0.001 and sample_weight.std().item() > 0.01:
        print("  ✅ Weights appear to be properly loaded (non-trivial values)")
    else:
        print("  ⚠️  Weights might not be loaded correctly")

print("\nChecking structure encoder weights:")
sample_weight = None
for name, param in model.structure_encoder.named_parameters():
    if 'weight' in name and param.numel() > 100:
        sample_weight = param
        weight_name = name
        break

if sample_weight is not None:
    print(f"  Parameter: {weight_name}")
    print(f"  Shape: {sample_weight.shape}")
    print(f"  Mean: {sample_weight.mean().item():.6f}")
    print(f"  Std: {sample_weight.std().item():.6f}")
    print(f"  Min: {sample_weight.min().item():.6f}")
    print(f"  Max: {sample_weight.max().item():.6f}")
    
    if abs(sample_weight.mean().item()) > 0.001 and sample_weight.std().item() > 0.01:
        print("  ✅ Weights appear to be properly loaded (non-trivial values)")
    else:
        print("  ⚠️  Weights might not be loaded correctly")

# Now compare with safetensors directly
print("\n" + "="*80)
print("Comparing with safetensors file:")
print("="*80)

from safetensors import safe_open

with safe_open('pllm/model.safetensors', framework='pt', device='cpu') as f:
    # Get the same weight from safetensors
    keys = [k for k in f.keys() if k.startswith('protein_encoder.') and 'weight' in k]
    if keys:
        key = keys[0]
        tensor_from_file = f.get_tensor(key)
        
        # Get corresponding weight from model
        param_name = key.replace('protein_encoder.', '')
        model_tensor = dict(model.protein_encoder.named_parameters())[param_name]
        
        print(f"\nComparing: {key}")
        print(f"  From file - Mean: {tensor_from_file.mean().item():.6f}, Std: {tensor_from_file.std().item():.6f}")
        print(f"  From model - Mean: {model_tensor.mean().item():.6f}, Std: {model_tensor.std().item():.6f}")
        
        # Check if they're the same
        if torch.allclose(tensor_from_file, model_tensor, atol=1e-5):
            print("  ✅ Weights match perfectly!")
        else:
            print("  ❌ Weights don't match!")

print("\n" + "="*80)
print("✅ Verification complete!")
print("="*80)

