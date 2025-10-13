"""
Simple inference script for PLLM model.
Loads a saved model and runs basic inference.
"""

import torch
from transformers import AutoTokenizer
from proteinLLM_pllm import PLLM


def main():
    # Configuration
    MODEL_PATH = "/mnt/efs/erran/rllm_v02/pllm"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {DEVICE}")
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load model
    model = PLLM.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/llm", use_fast=True)
    except:
        print("⚠️  Tokenizer not found in model dir, using default...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Example protein data
    aa_sequences = [
        "MKTFFVAIATGAFSATA",
        "MGDVEKGKKIFIMKCSQCHTVEK"
    ]
    
    # Structure sequences (3Di tokens - using AA alphabet as placeholder)
    structure_sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "ACDEFGHIKLMNP"
    ]
    
    # Prompts
    prompts = [
        "Explain the likely function of this protein based on its sequence.",
        "Which domain could this protein contain?"
    ]
    
    # Expected responses (for reference)
    expected_responses = [
        "It may be an enzyme with hydrolase activity.",
        "It likely contains a Rossmann-like fold."
    ]
    
    print("\n" + "="*80)
    print("Running Inference Examples")
    print("="*80)
    
    for i, (aa_seq, stru_seq, prompt) in enumerate(zip(aa_sequences, structure_sequences, prompts)):
        print(f"\n--- Example {i+1} ---")
        print(f"Amino Acid Sequence: {aa_seq}")
        print(f"Structure Sequence: {stru_seq}")
        print(f"Prompt: {prompt}")
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=False)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)
        
        # Forward pass (without labels for inference)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                aa_seq=[aa_seq],
                stru_str=[stru_seq],
                labels=None,  # No labels for inference
            )
        
        # Get predictions
        logits = outputs.logits
        predicted_token_ids = logits.argmax(dim=-1)
        
        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {logits.shape}")
        print(f"   Loss: {outputs.loss.item() if outputs.loss is not None else 'N/A'}")
        
        # Decode next token prediction
        next_token_id = logits[0, -1, :].argmax().item()
        next_token = tokenizer.decode([next_token_id])
        print(f"   Next token prediction: '{next_token}'")
    
    # Batch inference
    print("\n" + "="*80)
    print("Batch Inference")
    print("="*80)
    
    batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    batch_input_ids = batch_inputs["input_ids"].to(DEVICE)
    batch_attention_mask = batch_inputs["attention_mask"].to(DEVICE)
    
    with torch.no_grad():
        batch_outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            aa_seq=aa_sequences,
            stru_str=structure_sequences,
            labels=None,  # No labels for inference
        )
    
    print(f"✅ Batch inference successful!")
    print(f"   Batch size: {len(aa_sequences)}")
    print(f"   Output logits shape: {batch_outputs.logits.shape}")
    print(f"   Batch loss: {batch_outputs.loss.item() if batch_outputs.loss is not None else 'N/A'}")
    
    # Show protein encoding details
    print("\n" + "="*80)
    print("Protein Encoding Details")
    print("="*80)
    
    with torch.no_grad():
        prot_vec, prot_mask = model.encode_protein_batch(aa_sequences, structure_sequences)
    
    print(f"Protein vectors shape: {prot_vec.shape}")
    print(f"Protein mask shape: {prot_mask.shape}")
    print(f"Tokens per sequence: {prot_mask.sum(dim=1).tolist()}")
    print(f"Model hidden size: {model.hidden_size}")
    
    print("\n" + "="*80)
    print("✅ All inference tests completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()

