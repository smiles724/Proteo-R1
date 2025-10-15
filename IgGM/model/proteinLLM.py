import os
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# Keep your original import style; adjust if your files aren't under a 'model' package
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


class PrefixProjector(nn.Module):
    """
    Simple per-token MLP: (B, L, D) -> (B, L, H)
    """

    def __init__(self, in_dim: int, mid_dim: int, out_hidden: int, dropout: float = 0.1, dtype=None, device=None):
        super().__init__()
        self.out_hidden = out_hidden
        self.net = nn.Sequential(nn.Linear(in_dim, mid_dim, bias=True), nn.GELU(), nn.Dropout(dropout), nn.Linear(mid_dim, out_hidden, bias=True), )
        if dtype is not None or device is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, protein_vec: torch.Tensor) -> torch.Tensor:
        # protein_vec: [B, L, D]
        return self.net(protein_vec)


class PLLM(nn.Module):
    def __init__(self, model_name: str, protein_config: str, structure_config: str, protrek_ckpt: str = None, prot_slot: int = 1, stru_slot: int = 3,
                 proj_hid: int = 1024, dropout: float = 0.1, train_encoders: bool = True, dtype_str: str = "auto", ):
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

        # Per-token projector: 1024 -> hidden size (dtype aligned to LLM)
        model_dtype = next(self.llm.parameters()).dtype
        self.prefix_mlp = PrefixProjector(in_dim=1024, mid_dim=proj_hid, out_hidden=self.hidden_size, dropout=dropout, dtype=model_dtype, )

        # Freeze encoders if requested
        if not train_encoders:
            for p in self.protein_encoder.parameters():
                p.requires_grad = False
            for p in self.structure_encoder.parameters():
                p.requires_grad = False

        # Extract QKV from specific layers
        self.extract_layer_idx = extract_layer_idx
        self.qkv_cache = {}  # Store QKV from hooks
        
        # Register hooks to capture QKV
        self._register_qkv_hooks()

    def _register_qkv_hooks(self):
        """Register forward hooks to capture QKV and attention mask"""
        def create_qkv_hook(layer_idx):
            def hook(module, input, output):
                if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                    hidden_states = input[0] if isinstance(input, tuple) else input
                    
                    # Compute QKV
                    q = module.q_proj(hidden_states)
                    k = module.k_proj(hidden_states)
                    v = module.v_proj(hidden_states)
                    
                    # Extract attention mask from input
                    attention_mask = None
                    if len(input) > 1:
                        attention_mask = input[1]  # Usually the second argument
                    
                    # Store in cache
                    self.qkv_cache[layer_idx] = {
                        'q': q.detach(),
                        'k': k.detach(), 
                        'v': v.detach(),
                        'attention_mask': attention_mask.detach() if attention_mask is not None else None,
                        'hidden_states': hidden_states.detach()
                    }
            return hook
        
        # Register hook on the specified layer
        if hasattr(self.llm, 'model') and hasattr(self.llm.model, 'layers'):
            target_layer = self.llm.model.layers[self.extract_layer_idx]
            if hasattr(target_layer, 'self_attn'):
                target_layer.self_attn.register_forward_hook(create_qkv_hook(self.extract_layer_idx))
    
    def get_qkv_from_layer(self, layer_idx: int = None):
        """Get QKV and attention mask from specified layer"""
        if layer_idx is None:
            layer_idx = self.extract_layer_idx
        
        qkv_data = self.qkv_cache.get(layer_idx, None)
        if qkv_data is not None:
            # Ensure attention mask is available
            if qkv_data.get('attention_mask') is None and hasattr(self, 'current_attention_mask'):
                qkv_data['attention_mask'] = self.current_attention_mask
            
            return {
                'q': qkv_data['q'],
                'k': qkv_data['k'],
                'v': qkv_data['v'],
                'm': qkv_data.get('attention_mask')  # 'm' for mask
            }
        return None
    
    def clear_qkv_cache(self):
        """Clear the QKV cache"""
        self.qkv_cache.clear()

    def encode_protein_batch(self, aa_list, stru_list):
        """
        Return token-level embeddings and mask:
          prot_tok:  (B, L_total, 1024)
          prot_mask: (B, L_total) boolean
        """
        B = len(aa_list)
        device = next(self.llm.parameters()).device
        dtype = next(self.llm.parameters()).dtype

        idx_seq = [i for i, a in enumerate(aa_list) if a is not None and len(a) > 0]
        idx_stru = [i for i, s in enumerate(stru_list) if s is not None and len(s) > 0]

        # sequences
        if len(idx_seq) > 0:
            seqs = [aa_list[i] for i in idx_seq]
            emb_s, m_s, _ = self.protein_encoder(seqs, get_mask_logits=False, device=device)  # (N, Ls, 1024)
            Ls = emb_s.size(1)
        else:
            emb_s = m_s = None
            Ls = 0
        seq_tok = torch.zeros(B, Ls, 1024, device=device, dtype=dtype)
        seq_mask = torch.zeros(B, Ls, device=device, dtype=torch.bool)
        if len(idx_seq) > 0:
            seq_tok[idx_seq, :Ls, :] = emb_s.to(device=device, dtype=dtype)
            seq_mask[idx_seq, :Ls] = m_s.to(device=device)

        # structures
        if len(idx_stru) > 0:
            strs = [stru_list[i] for i in idx_stru]
            emb_t, m_t, _ = self.structure_encoder.get_repr(strs, batch_size=max(1, len(strs)), verbose=False)  # (N, Lt, 1024)
            Lt = emb_t.size(1)
        else:
            emb_t = m_t = None
            Lt = 0
        struct_tok = torch.zeros(B, Lt, 1024, device=device, dtype=dtype)
        struct_mask = torch.zeros(B, Lt, device=device, dtype=torch.bool)
        if len(idx_stru) > 0:
            struct_tok[idx_stru, :Lt, :] = emb_t.to(device=device, dtype=dtype)
            struct_mask[idx_stru, :Lt] = m_t.to(device=device)

        # concatenate along sequence axis   # todo: special tokens, prompt before each modality
        prot_tok = torch.cat([seq_tok, struct_tok], dim=1)  # (B, L', 1024)
        prot_mask = torch.cat([seq_mask, struct_mask], dim=1)  # (B, L')
        return prot_tok, prot_mask

    def build_prefix(self, protein_tokens: torch.Tensor) -> torch.Tensor:
        # Per-token MLP to LLM hidden size
        return self.prefix_mlp(protein_tokens)  # (B, L', H)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: torch.Tensor, aa_seq: List[Optional[str]], 
                stru_str: List[Optional[str]], extract_qkv: bool = False) -> torch.Tensor:        # Text embeddings
        
        # Clear previous cache
        if extract_qkv:
            self.clear_qkv_cache()
            
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # Token-level protein prefix
        prot_tok, prot_mask = self.encode_protein_batch(aa_seq, stru_str)  # (B, L', 1024), (B, L')
        pref = self.build_prefix(prot_tok)  # (B, L', H)

        # Concat prefix + text and merge masks
        inputs_embeds = torch.cat([pref, text_embeds], dim=1)  # (B, L'+T, H)
        attn = torch.cat([prot_mask.to(attention_mask.dtype), attention_mask], dim=1)

        # Extend labels with -100 for prefix
        pad = torch.full((labels.size(0), pref.size(1)), -100, dtype=labels.dtype, device=labels.device)
        new_labels = torch.cat([pad, labels], dim=1)

        # Qwen adds position ids internally; no manual position_ids needed
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=new_labels, use_cache=False)
        return out


if __name__ == '__main__':
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    protein_config = 'esm2_t33_650M_UR50D'
    structure_config = "foldseek_t30_150M"
    protrek_ckpt = "ProTrek_35M/ProTrek_35M.pt"
    model = PLLM(model_name=model_name, protein_config=protein_config, structure_config=structure_config, protrek_ckpt=protrek_ckpt, prot_slot=1,
                 stru_slot=3, proj_hid=2048, dropout=0.1, dtype_str="fp32", )




