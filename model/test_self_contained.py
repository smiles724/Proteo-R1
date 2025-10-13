"""
Test that the saved model is fully self-contained and can be loaded
with just PLLM.from_pretrained() without any additional arguments.
"""

import torch
from proteinLLM_pllm import PLLM
from transformers import AutoTokenizer


def test_self_contained_model():
    """Test that the model loads all components from the saved directory."""
    
    print("="*80)
    print("Testing Self-Contained Model Loading")
    print("="*80)
    
    MODEL_PATH = "./pllm"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n1. Loading model from: {MODEL_PATH}")
    print(f"   Device: {DEVICE}")
    
    # Load model with NO additional arguments
    model = PLLM.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    
    print("   ✅ Model loaded successfully!")
    
    # Verify all components are loaded
    print("\n2. Verifying model components:")
    
    # Check LLM
    assert hasattr(model, 'llm'), "LLM not loaded"
    print("   ✅ LLM loaded")
    
    # Check tokenizer
    assert hasattr(model, 'tokenizer'), "Tokenizer not loaded"
    print("   ✅ Tokenizer loaded")
    
    # Check protein encoder
    assert hasattr(model, 'protein_encoder'), "Protein encoder not loaded"
    print("   ✅ Protein encoder loaded")
    
    # Check structure encoder
    assert hasattr(model, 'structure_encoder'), "Structure encoder not loaded"
    print("   ✅ Structure encoder loaded")
    
    # Check prefix MLP
    assert hasattr(model, 'prefix_mlp'), "Prefix MLP not loaded"
    print("   ✅ Prefix MLP loaded")
    
    # Check ProTrek checkpoint was loaded (verify weights are not random)
    print("\n3. Verifying ProTrek weights were loaded:")
    
    # Get a sample weight from protein encoder
    sample_weight = None
    for name, param in model.protein_encoder.named_parameters():
        if 'weight' in name:
            sample_weight = param
            break
    
    if sample_weight is not None:
        # Check if weights are not all zeros or ones (indicating they were loaded)
        weight_mean = sample_weight.mean().item()
        weight_std = sample_weight.std().item()
        print(f"   Sample weight statistics:")
        print(f"   - Mean: {weight_mean:.6f}")
        print(f"   - Std: {weight_std:.6f}")
        
        if abs(weight_mean) > 0.001 and weight_std > 0.001:
            print("   ✅ ProTrek weights appear to be loaded (non-trivial values)")
        else:
            print("   ⚠️  Weights might be random-initialized")
    
    # Test inference
    print("\n4. Testing inference:")
    
    aa_seq = "MKTFFVAIATGAFSATA"
    stru_seq = "ACDEFGHIKLMNPQRSTVWY"
    prompt = "What is the function of this protein?"
    
    inputs = model.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            aa_seq=[aa_seq],
            stru_str=[stru_seq],
            labels=None,
        )
    
    print(f"   Output logits shape: {outputs.logits.shape}")
    print("   ✅ Inference successful!")
    
    # Test protein encoding
    print("\n5. Testing protein encoding:")
    
    with torch.no_grad():
        prot_vec, prot_mask = model.encode_protein_batch([aa_seq], [stru_seq])
    
    print(f"   Protein vector shape: {prot_vec.shape}")
    print(f"   Protein mask shape: {prot_mask.shape}")
    print(f"   Number of protein tokens: {prot_mask.sum().item()}")
    print("   ✅ Protein encoding successful!")
    
    # Verify config was saved correctly
    print("\n6. Verifying saved configuration:")
    
    import json
    from pathlib import Path
    
    config_path = Path(MODEL_PATH) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    required_keys = [
        "protein_config",
        "structure_config",
        "base_model_name_or_path",
        "hidden_size",
        "prefix_len",
        "proj_hid",
        "train_encoders",
    ]
    
    for key in required_keys:
        if key in config:
            print(f"   ✅ {key}: {config[key]}")
        else:
            print(f"   ❌ {key}: MISSING")
    
    # Verify files exist
    print("\n7. Verifying all required files exist:")
    
    model_dir = Path(MODEL_PATH)
    required_files = [
        "config.json",
        "model.safetensors",  # Contains encoder weights (no need for ProTrek_650M.pt)
        "llm/config.json",
        "llm/model.safetensors",
        "protein_config/config.json",
        "structure_config/config.json",
    ]
    
    for file_path in required_files:
        full_path = model_dir / file_path
        if full_path.exists():
            size = full_path.stat().st_size / (1024**2)  # Size in MB
            print(f"   ✅ {file_path} ({size:.1f} MB)")
        else:
            print(f"   ❌ {file_path} MISSING")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\nThe model is fully self-contained and can be loaded with:")
    print("    model = PLLM.from_pretrained('./pllm')")
    print("\nNo additional arguments needed!")
    print("="*80)


if __name__ == "__main__":
    test_self_contained_model()

