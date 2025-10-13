import os, argparse
from typing import List, Dict, Optional

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM

#import model.protein_encoder as protein_encoder_mod
#import model.structure_encoder as structure_encoder_mod
import protein_encoder as protein_encoder_mod
import structure_encoder as structure_encoder_mod

DTYPE_MAP = {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "auto": None,
             "default": None, None: None}


def resolve_dtype(dtype_str):
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    key = str(dtype_str).lower() if dtype_str is not None else None
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {dtype_str}. Use one of: fp32, fp16, bf16, auto.")
    return DTYPE_MAP[key]


# ------------------------
# Prefix projector: (B, L, 1024) -> (B, L, H)
# ------------------------
class PrefixProjector(nn.Module):
    def __init__(self, in_dim: int, mid_dim: int, out_hidden: int, dropout: float = 0.1, dtype=None, device=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_hidden, bias=True),
        )
        if dtype is not None or device is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        return self.net(x)              # (B, L, H)

class BigProteinQwen(nn.Module):
    def __init__(self, model_name: str, protein_config: str, structure_config: str, protrek_ckpt: str = None, prot_slot: int = 1, stru_slot: int = 3,
                 single_token_prefix: bool = False, prefix_len: int = 4, proj_hid: int = 1024, dropout: float = 0.1, train_encoders: bool = False, dtype_str: str = "auto", ):
        super().__init__()
        load_dtype = resolve_dtype(dtype_str)
        if load_dtype is None:
            load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=load_dtype, low_cpu_mem_usage=True, )
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = False
        self.llm.gradient_checkpointing_enable()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = self.llm.config.hidden_size
        self.prefix_len = 1 if single_token_prefix else prefix_len

        # Encoders (arch from configs; weights from ProTrek slots below)
        self.protein_encoder = protein_encoder_mod.ProteinEncoder(protein_config, out_dim=1024, load_pretrained=False)
        self.structure_encoder = structure_encoder_mod.StructureEncoder(structure_config, out_dim=1024, load_pretrained=False)

        # ---- ProTrek slot-based loading ----
        if protrek_ckpt and os.path.exists(protrek_ckpt):
            sd_raw = torch.load(protrek_ckpt, map_location="cpu")
            sd = sd_raw.get("model", sd_raw.get("state_dict", sd_raw))
            slots = {}
            for k, v in sd.items():
                head = k.split(".", 1)[0]
                if head.isdigit():
                    slots.setdefault(int(head), {})[k[len(head) + 1:]] = v

            def drop_extras(sub):
                return {k: v for k, v in sub.items() if "embeddings.position_ids" not in k}

            if prot_slot in slots:
                mp, up = self.protein_encoder.load_state_dict(drop_extras(slots[prot_slot]), strict=False)
                print(f"[ProteinEncoder] loaded from slot {prot_slot} | missing={len(mp)} unexpected={len(up)}")
            else:
                print(f"[ProteinEncoder] WARNING: slot {prot_slot} not found; skipping ckpt load.")

            if stru_slot in slots:
                ms, us = self.structure_encoder.load_state_dict(drop_extras(slots[stru_slot]), strict=False)
                print(f"[StructureEncoder] loaded from slot {stru_slot} | missing={len(ms)} unexpected={len(us)}")
            else:
                print(f"[StructureEncoder] WARNING: slot {stru_slot} not found; skipping ckpt load.")
        else:
            print("No ProTrek checkpoint provided or path not found; encoders stay random-init.")

        # projector (outputs directly to LLM hidden size, dtype-aligned to LLM)
        model_dtype = next(self.llm.parameters()).dtype
        self.prefix_mlp = PrefixProjector(1024, proj_hid, self.hidden_size, dropout, dtype=model_dtype)
        #self.proj_2048 = PrefixProjector(2048, proj_hid, self.hidden_size, dropout, dtype=model_dtype)

        # Freeze encoders if requested
        if not train_encoders:
            for p in self.protein_encoder.parameters():   p.requires_grad = False
            for p in self.structure_encoder.parameters(): p.requires_grad = False

    # -------- protein encoding to a uniform 2048-D vector ----------
    def encode_protein_batch(self, aa_list, stru_list):
        """
        Encode sequence/structure into token-level features and concatenate **along length**.
        Returns:
            fused: (B, L_seq + L_struct, 1024)
            mask:  (B, L_seq + L_struct)  (torch.bool)
        Notes:
            - Handles per-example missing inputs (None or ""), producing zero-length for that modality.
            - At least one modality list must be provided (not both None).
        """
        if aa_list is None and stru_list is None:
            raise ValueError("encode_protein_batch: both aa_list and stru_list are None; need at least one modality.")

        # Determine batch size B and sanitize lists to lists-of-str
        if aa_list is not None:
            assert isinstance(aa_list, (list, tuple)), "aa_list must be a list of strings or None."
            B = len(aa_list)
            aa_fixed = [s if (isinstance(s, str) and len(s) > 0) else "" for s in aa_list]
        else:
            aa_fixed = None

        if stru_list is not None:
            assert isinstance(stru_list, (list, tuple)), "stru_list must be a list of strings or None."
            if aa_list is None:
                B = len(stru_list)
            else:
                assert len(stru_list) == B, "aa_list and stru_list must have the same batch size."
            stru_fixed = [s if (isinstance(s, str) and len(s) > 0) else "" for s in stru_list]
        else:
            assert aa_list is not None, "Both modalities are None; cannot infer batch size."
            stru_fixed = None

        device = next(self.llm.parameters()).device
        model_dtype = next(self.llm.parameters()).dtype

        # Encode modalities using their public forward (already projects to out_dim=1024)
        seq_tok = seq_msk = None
        stru_tok = stru_msk = None

        if aa_fixed is not None:
            seq_tok, seq_msk, _ = self.protein_encoder(aa_fixed, device=device)  # (B, L_seq_max, 1024), (B, L_seq_max)
        if stru_fixed is not None:
            stru_tok, stru_msk, _ = self.structure_encoder(stru_fixed, device=device)  # (B, L_str_max, 1024), (B, L_str_max)

        # Concatenate along the token axis → width remains 1024
        if seq_tok is None and stru_tok is None:
            # Should not happen due to earlier checks; safe fallback
            fused = torch.zeros(B, 0, 1024, device=device, dtype=model_dtype)
            mask  = torch.zeros(B, 0, device=device, dtype=torch.bool)
        elif seq_tok is None:
            fused, mask = stru_tok.to(dtype=model_dtype), stru_msk.to(device=device)
        elif stru_tok is None:
            fused, mask = seq_tok.to(dtype=model_dtype), seq_msk.to(device=device)
        else:
            fused = torch.cat([seq_tok, stru_tok], dim=1).to(dtype=model_dtype)  # (B, L_seq + L_struct, 1024)
            mask  = torch.cat([seq_msk, stru_msk], dim=1).to(device=device)     # (B, L_seq + L_struct)

        # Ensure mask is boolean
        if mask.dtype != torch.bool:
            mask = mask.bool()

        return fused, mask


        # -------- prefix build ----------
    def build_prefix(self, protein_vec: torch.Tensor) -> torch.Tensor:
        """Project 1024-D protein tokens into LLM hidden space (length-fusion path)."""
        assert protein_vec.size(-1) == 1024, f"Expected 1024-D tokens, got {tuple(protein_vec.shape)}"
        pref = self.prefix_mlp(protein_vec)  # [B, T, H]
        return pref

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        aa_seq: List[Optional[str]],
        stru_str: List[Optional[str]],
    ):
        # Text embeddings (sharding-safe)
        text_embeds = self.llm.get_input_embeddings()(input_ids)       

        # Build protein prefix
        prot_vec, prot_mask = self.encode_protein_batch(aa_seq, stru_str) 
        pref = self.build_prefix(prot_vec)                                  
        # Cast protein mask to the same dtype as attention_mask (HF expects 1/0)
        attn_prefix = prot_mask.to(attention_mask.dtype)                    # [B, L']

        # Concat prefix + text
        inputs_embeds = torch.cat([pref, text_embeds], dim=1)               # [B, L'+T, H]
        attn = torch.cat([attn_prefix, attention_mask], dim=1)              # [B, L'+T]

        # Extend labels with -100 for prefix
        pad = torch.full((labels.size(0), pref.size(1)), -100, dtype=labels.dtype, device=labels.device)
        new_labels = torch.cat([pad, labels], dim=1)

        # Qwen adds RoPE & position_ids automatically when attention_mask is provided
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=new_labels, use_cache=False)
        return out
############################################################
# Simple demo: build model, run one forward pass with loss #
############################################################
if __name__ == "__main__":
    import os
    import argparse
    import torch
    from transformers import AutoTokenizer

    # ---- CLI arguments with your requested defaults ----
    parser = argparse.ArgumentParser(description="BigProteinQwen demo run")
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HF model id or local path for the LLM backbone.",
    )
    parser.add_argument(
        "--protein-config",
        default="/protrek/weights/ProTrek_35M/esm2_t12_35M_UR50D",
        help="HF id or local path for the protein (sequence) encoder config/weights.",
    )
    parser.add_argument(
        "--structure-config",
        default="/protrek/weights/ProTrek_35M/foldseek_t12_35M",
        help="HF id or local path for the structure encoder config/weights.",
    )
    parser.add_argument(
        "--protrek-ckpt",
        default="/protrek/weights/ProTrek_35M/ProTrek_35M.pt",
        help="Path to ProTrek checkpoint containing slot weights.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "float32", "fp16", "float16", "bf16", "bfloat16"],
        help="Model dtype for LLM load.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device override; 'auto' picks CUDA if available.",
    )
    parser.add_argument(
        "--train-encoders",
        action="store_true",
        help="If set, encoders are trainable; otherwise they are frozen.",
    )
    parser.add_argument(
        "--prot-slot",
        type=int,
        default=1,
        help="ProTrek slot id for protein encoder.",
    )
    parser.add_argument(
        "--stru-slot",
        type=int,
        default=3,
        help="ProTrek slot id for structure encoder.",
    )
    args = parser.parse_args()

    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    DTYPE = args.dtype
    print(f"Device: {DEVICE} | dtype: {DTYPE}")

    # Tiny toy data (two items)
    aa_list = [
        "MKTFFVAIATGAFSATA",
        "MGDVEKGKKIFIMKCSQCHTVEK",
    ]
    # Use AA alphabet as stand-in 3Di tokens for a quick run
    stru_list = [
        "ACDEFGHIKLMNPQRSTVWY",
        "ACDEFGHIKLMNP",
    ]

    prompts = [
        "Explain the likely function of this protein based on its sequence.",
        "Which domain could this protein contain?",
    ]
    responses = [
        "It may be an enzyme with hydrolase activity.",
        "It likely contains a Rossmann-like fold.",
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build the big model
    model = BigProteinQwen(
        model_name=args.model_name,
        protein_config=args.protein_config,
        structure_config=args.structure_config,
        protrek_ckpt=args.protrek_ckpt,
        prot_slot=args.prot_slot,
        stru_slot=args.stru_slot,
        train_encoders=bool(args.train_encoders),
        proj_hid=1024,
        dropout=0.10,
        dtype_str=DTYPE,
    ).to(DEVICE)
    print("hidden_size:", model.hidden_size)

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
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attn, dtype=torch.long, device=device),
            "labels": torch.tensor(labels, dtype=torch.long, device=device),
        }

    batch = make_batch(tokenizer, prompts, responses, device=DEVICE)

    # Forward once (no grad) to confirm wiring
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            aa_seq=aa_list,
            stru_str=stru_list,
        )
    print("Forward OK. loss =", float(out.loss))

    # Optional: show prefix shapes
    try:
        with torch.no_grad():
            prot_vec, prot_mask = model.encode_protein_batch(aa_list, stru_list)
            print("prot_vec (prefix tokens):", tuple(prot_vec.shape))  # e.g., (B, Lmax, 2048) if feature-concat path
            print("prot_mask:", tuple(prot_mask.shape))
    except Exception as e:
        print("Skipped prefix shape check:", repr(e))