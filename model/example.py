import argparse
import torch
from transformers import AutoTokenizer
from model.proteinLLM_pllm import PLLM


# Helper: build a tiny SFT batch (mask prompt + EOS with -100)
def make_batch(tok, ps, rs, max_len=512, device="cpu"):
    eos_id = tok.eos_token_id
    input_ids, attn, labels = [], [], []
    for p, r in zip(ps, rs):
        p_ids = tok.encode(p, add_special_tokens=False)
        r_ids = tok.encode(r, add_special_tokens=False)
        ids = p_ids + r_ids + [eos_id]
        la = [-100] * len(p_ids) + r_ids + [-100]  # mask prompt + EOS
        input_ids.append(ids)
        labels.append(la)
    maxL = min(max_len, max(len(x) for x in input_ids))

    def pad_to(x, pad_val): return x + [pad_val] * (maxL - len(x))

    input_ids = [pad_to(x, tok.pad_token_id) for x in input_ids]
    labels = [pad_to(x, -100) for x in labels]
    attn = [[1] * len(x) + [0] * (maxL - len(x)) for x in input_ids]
    return {"input_ids": torch.tensor(input_ids, dtype=torch.long, device=device), "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device), }


if __name__ == "__main__":
    # ---- CLI arguments with your requested defaults ----
    parser = argparse.ArgumentParser(description="BigProteinQwen demo run")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct", help="HF model id or local path for the LLM backbone.", )
    parser.add_argument("--protein-config", default="./ProTrek_650M/esm2_t33_650M_UR50D",
                        help="HF id or local path for the protein (sequence) encoder config/weights.", )
    parser.add_argument("--structure-config", default="./ProTrek_650M/foldseek_t30_150M", help="HF id or local path for the structure encoder config/weights.", )
    parser.add_argument("--protrek-ckpt", default="./ProTrek_650M/ProTrek_650M.pt", help="Path to ProTrek checkpoint containing slot weights.", )
    parser.add_argument("--dtype", default="auto", choices=["auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"], help="Model dtype for LLM load.", )
    parser.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"], help="Device override; 'auto' picks CUDA if available.", )
    parser.add_argument("--train-encoders", action="store_true", help="If set, encoders are trainable; otherwise they are frozen.", )
    parser.add_argument("--prot-slot", type=int, default=1, help="ProTrek slot id for protein encoder.", )
    parser.add_argument("--stru-slot", type=int, default=3, help="ProTrek slot id for structure encoder.", )
    args = parser.parse_args()

    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    DTYPE = args.dtype
    print(f"Device: {DEVICE} | dtype: {DTYPE}")

    # Tiny toy data (two items)
    aa_list = ["MKTFFVAIATGAFSATA", "MGDVEKGKKIFIMKCSQCHTVEK", ]
    # Use AA alphabet as stand-in 3Di tokens for a quick run
    stru_list = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNP", ]

    prompts = ["Explain the likely function of this protein based on its sequence.", "Which domain could this protein contain?", ]
    responses = ["It may be an enzyme with hydrolase activity.", "It likely contains a Rossmann-like fold.", ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build the big model
    model = PLLM(model_name=args.model_name, protein_config=args.protein_config, structure_config=args.structure_config, protrek_ckpt=args.protrek_ckpt, prot_slot=args.prot_slot,
                 stru_slot=args.stru_slot, train_encoders=bool(args.train_encoders), proj_hid=1024, dropout=0.10, dtype_str=DTYPE, ).to(DEVICE)
    print("hidden_size:", model.hidden_size)

    batch = make_batch(tokenizer, prompts, responses, device=DEVICE)

    # Forward once (no grad) to confirm wiring
    model.eval()
    with torch.no_grad():
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], aa_seq=aa_list, stru_str=stru_list, )
    print("Forward OK. loss =", float(out.loss))

    # Optional: show prefix shapes
    try:
        with torch.no_grad():
            prot_vec, prot_mask = model.encode_protein_batch(aa_list, stru_list)
            print("prot_vec (prefix tokens):", tuple(prot_vec.shape))  # e.g., (B, Lmax, 2048) if feature-concat path
            print("prot_mask:", tuple(prot_mask.shape))
    except Exception as e:
        print("Skipped prefix shape check:", repr(e))
