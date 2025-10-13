import os
import argparse
import torch
from transformers import AutoTokenizer
from proteinLLM_pllm import PLLM


def generate_text(model, tokenizer, input_ids, attention_mask, aa_seq, stru_str, max_new_tokens=100, temperature=0.7, top_p=0.9, device="cuda"):
    """
    Generate text using the PLLM model with protein context.
    
    Args:
        model: PLLM model
        tokenizer: Tokenizer
        input_ids: Input token IDs (prompt)
        attention_mask: Attention mask
        aa_seq: List of amino acid sequences
        stru_str: List of structure sequences (3Di tokens)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        device: Device to run on
    
    Returns:
        Generated text
    """
    model.eval()
    
    with torch.no_grad():
        # Encode protein context once
        prot_vec, prot_mask = model.encode_protein_batch(aa_seq, stru_str)
        
        # Generate tokens autoregressively
        generated_ids = input_ids.clone()
        current_attention_mask = attention_mask.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(
                input_ids=generated_ids,
                attention_mask=current_attention_mask,
                aa_seq=aa_seq,
                stru_str=stru_str,
                labels=None,  # No labels for generation
            )
            
            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((current_attention_mask.shape[0], 1), dtype=torch.long, device=device)
            ], dim=-1)
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load saved PLLM model and run inference")
    parser.add_argument("--model-path", default="./pllm", help="Path to saved PLLM model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run on")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter")
    args = parser.parse_args()
    
    DEVICE = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Load the saved model
    print(f"Loading model from {args.model_path}...")
    model = PLLM.from_pretrained(args.model_path).to(DEVICE)
    print("Model loaded successfully!")
    
    # Load tokenizer (assuming it's saved with the model or use the original)
    tokenizer_path = os.path.join(args.model_path, "llm")
    if os.path.exists(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    else:
        # Fallback to default tokenizer
        print("Tokenizer not found in model directory, using default...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model hidden size: {model.hidden_size}")
    
    # Example 1: Enzyme function prediction
    print("\n" + "="*80)
    print("Example 1: Enzyme Function Prediction")
    print("="*80)
    
    aa_seq = "MKTFFVAIATGAFSATA"
    stru_seq = "ACDEFGHIKLMNPQRSTVWY"  # 3Di structure tokens
    prompt = "Explain the likely function of this protein based on its sequence."
    
    print(f"Amino Acid Sequence: {aa_seq}")
    print(f"Structure Sequence: {stru_seq}")
    print(f"Prompt: {prompt}")
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    
    # Generate response
    print("\nGenerating response...")
    response = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        aa_seq=[aa_seq],
        stru_str=[stru_seq],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=DEVICE
    )
    
    print(f"\nResponse: {response}")
    
    # Example 2: Domain identification
    print("\n" + "="*80)
    print("Example 2: Domain Identification")
    print("="*80)
    
    aa_seq2 = "MGDVEKGKKIFIMKCSQCHTVEK"
    stru_seq2 = "ACDEFGHIKLMNP"
    prompt2 = "Which domain could this protein contain?"
    
    print(f"Amino Acid Sequence: {aa_seq2}")
    print(f"Structure Sequence: {stru_seq2}")
    print(f"Prompt: {prompt2}")
    
    inputs2 = tokenizer(prompt2, return_tensors="pt", padding=False, truncation=False)
    input_ids2 = inputs2["input_ids"].to(DEVICE)
    attention_mask2 = inputs2["attention_mask"].to(DEVICE)
    
    print("\nGenerating response...")
    response2 = generate_text(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids2,
        attention_mask=attention_mask2,
        aa_seq=[aa_seq2],
        stru_str=[stru_seq2],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=DEVICE
    )
    
    print(f"\nResponse: {response2}")
    
    # Example 3: Batch inference
    print("\n" + "="*80)
    print("Example 3: Batch Inference (Forward Pass Only)")
    print("="*80)
    
    aa_list = [aa_seq, aa_seq2]
    stru_list = [stru_seq, stru_seq2]
    prompts = [prompt, prompt2]
    
    # Tokenize batch
    batch_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    batch_input_ids = batch_inputs["input_ids"].to(DEVICE)
    batch_attention_mask = batch_inputs["attention_mask"].to(DEVICE)
    
    print(f"Batch size: {len(aa_list)}")
    print("Running forward pass...")
    
    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            aa_seq=aa_list,
            stru_str=stru_list,
            labels=None,  # No labels for inference
        )
    
    print(f"Output logits shape: {outputs.logits.shape}")
    print(f"Hidden states available: {outputs.hidden_states is not None}")
    
    # Show protein encoding
    print("\n" + "="*80)
    print("Protein Encoding Information")
    print("="*80)
    
    with torch.no_grad():
        prot_vec, prot_mask = model.encode_protein_batch(aa_list, stru_list)
        print(f"Protein vector shape: {prot_vec.shape}")
        print(f"Protein mask shape: {prot_mask.shape}")
        print(f"Number of protein tokens per sequence: {prot_mask.sum(dim=1).tolist()}")
    
    print("\n" + "="*80)
    print("Inference completed successfully!")
    print("="*80)

