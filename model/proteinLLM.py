import os
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

import model.protein_encoder as protein_encoder_mod
import model.structure_encoder as structure_encoder_mod

DTYPE_MAP = {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "auto": None,
             "default": None, None: None}


def resolve_dtype(dtype_str):
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    key = str(dtype_str).lower() if dtype_str is not None else None
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {dtype_str}. Use one of: fp32, fp16, bf16, auto.")
    return DTYPE_MAP[key]


class PLLM(nn.Module):
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
        self.structure_encoder = structure_encoder_mod.StructureEncoder(structure_config, out_dim=1024)

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
        self.projector = nn.Sequential(nn.Linear(2048, proj_hid, bias=True), nn.GELU(), nn.Dropout(dropout), nn.Linear(proj_hid, self.hidden_size, bias=True), )

        # Freeze encoders if requested
        if not train_encoders:
            for p in self.protein_encoder.parameters():   p.requires_grad = False
            for p in self.structure_encoder.parameters(): p.requires_grad = False

    # -------- protein encoding to a uniform 2048-D vector ----------
    def encode_protein_batch(self, aa_list: List[Optional[str]], stru_list: List[Optional[str]]) -> [torch.Tensor, torch.Tensor]:
        """
        Encode a batch of (aa_seq, stru_str) into a uniform 2048-D vector per example:
        - seq encoder -> 1024, stru encoder -> 1024
        - missing modality is padded with zeros
        - result shape: [B, 2048], dtype/device aligned to LLM
        """
        B = len(aa_list)
        device = next(self.llm.parameters()).device
        dtype = next(self.llm.parameters()).dtype

        # Collect present indices
        idx_seq = [i for i, a in enumerate(aa_list) if a is not None and len(a) > 0]
        idx_stru = [i for i, s in enumerate(stru_list) if s is not None and len(s) > 0]

        assert len(idx_seq) > 0 and len(idx_stru) > 0

        # Batch encode structures
        # Our ProteinEncoder returns (emb, mask, logits) with:
        #   emb:  (N, L_max_subbatch, 1024)  token-level reps for this sub-batch
        #   mask: (N, L_max_subbatch)        True on real residues (BOS/EOS removed)
        seqs = [aa_list[i] for i in idx_seq]
        emb, mask, _ = self.protein_encoder(seqs, get_mask_logits=False, device=device)

        # Allocate batch-shaped holders once we know sub-batch max length
        B = len(aa_list)
        L_max_seq = emb.size(1)
        seq_tok = torch.zeros(B, L_max_seq, 1024, device=device, dtype=dtype)
        seq_mask = torch.zeros(B, L_max_seq, device=device, dtype=torch.bool)

        # Place the sub-batch embeddings/masks at the right rows
        seq_tok[idx_seq, :L_max_seq, :] = emb.to(device=device, dtype=dtype)
        seq_mask[idx_seq, :L_max_seq] = mask.to(device=device)

        # Batch encode structures
        strs = [stru_list[i] for i in idx_stru]
        str_emb, str_mask, _ = self.structure_encoder(strs, batch_size=max(1, len(strs)), verbose=False)  # (N,L_2, 1024)

        B = len(stru_list)
        L_max_struct = str_emb.size(1)
        struct_tok = torch.zeros(B, L_max_struct, 1024, device=device, dtype=dtype)
        struct_mask = torch.zeros(B, L_max_struct, device=device, dtype=torch.bool)

        # Place the sub-batch embeddings/masks at the right rows
        struct_tok[idx_seq, :L_max_struct, :] = str_emb.to(device=device, dtype=dtype)
        struct_mask[idx_seq, :L_max_struct] = str_mask.to(device=device)

        return torch.cat([seq_tok, struct_tok], dim=-1), torch.cat([seq_mask, struct_mask], dim=-1)  # [B, L1 + L2, 2048], [B, L1 + L2]

    def forward(self, input_ids: torch.Tensor, aa_seq: List[Optional[str]], stru_str: List[Optional[str]], attention_mask: torch.Tensor, labels: torch.Tensor, ) -> torch.Tensor:
        # Text embeddings (sharding-safe)
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # Protein embeddings
        prot_vec, prot_mask = self.encode_protein_batch(aa_seq, stru_str)  # [B, L', 2048], [B, L']
        pref = self.projector(prot_vec)  # [B, L', H]

        # Concat prefix + text
        inputs_embeds = torch.cat([pref, text_embeds], dim=1)  # [B, L'+T, H]
        attn = torch.cat([prot_mask, attention_mask], dim=1)

        # Extend labels with -100 for prefix
        pad = torch.full((labels.size(0), pref.size(1)), -100, dtype=labels.dtype, device=labels.device)
        new_labels = torch.cat([pad, labels], dim=1)

        # Qwen add RoPE in the forward and will add position_id automatically!
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=new_labels, use_cache=False)
        return out
