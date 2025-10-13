import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from safetensors import safe_open as _pllm_safe_open
    from safetensors.torch import save_file as _pllm_save_safetensors
except Exception as _e:
    _pllm_safe_open = None
    _pllm_save_safetensors = None

# Keep your original import style; adjust if your files aren't under a 'model' package
import protein_encoder as protein_encoder_mod
import structure_encoder as structure_encoder_mod

DTYPE_MAP = {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "auto": None,
             "default": None, None: None}

# ================== Hugging Face-style save/load helpers for PLLM ==================


def _as_local_path(path_or_repo: str) -> Path:
    p = Path(path_or_repo)
    if p.exists():
        return p
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(path_or_repo))
    except Exception:
        return p


def _resolve_rel(base_dir: Path, maybe_path):
    if maybe_path is None:
        return None
    s = str(maybe_path)
    if os.path.isabs(s):
        return s
    return str(base_dir / s)


def _maybe_copy_into(dirpath: Path, src_path, dst_name):
    if not src_path:
        return None
    try:
        src_path_obj = Path(src_path)
        
        # If it's a directory, copy the entire directory
        if src_path_obj.is_dir():
            dst = dirpath / dst_name.replace('.json', '')  # Remove .json extension for directories
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src_path_obj, dst)
            return dst_name.replace('.json', '')  # Return just the directory name (relative to dirpath)
        # If it's a file, copy the file
        elif src_path_obj.is_file():
            dst = dirpath / dst_name
            shutil.copyfile(src_path, dst)
            return dst_name  # store relative path
        else:
            # If path doesn't exist, store absolute path
            return str(src_path_obj.absolute())
    except Exception as e:
        # Fallback: store absolute path
        return str(Path(src_path).absolute())


def _pllm_gather_custom_state_dict(model: "PLLM"):
    sd = {}
    if hasattr(model, "protein_encoder"):
        sd.update({f"protein_encoder.{k}": v.detach().cpu() for k, v in model.protein_encoder.state_dict().items()})
    if hasattr(model, "structure_encoder"):
        sd.update({f"structure_encoder.{k}": v.detach().cpu() for k, v in model.structure_encoder.state_dict().items()})
    if hasattr(model, "prefix_mlp"):
        sd.update({f"prefix_mlp.{k}": v.detach().cpu() for k, v in model.prefix_mlp.state_dict().items()})
    return sd


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
                 single_token_prefix: bool = False, prefix_len: int = 4, proj_hid: int = 1024, dropout: float = 0.1, train_encoders: bool = True, dtype_str: str = "auto", ):
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
        
        # Store config paths for serialization
        self.protein_config = protein_config
        self.structure_config = structure_config
        self.train_encoders = train_encoders
        self.proj_hid = proj_hid
        # Note: We don't store protrek_ckpt because we save encoder weights directly

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

        # concatenate along sequence axis
        prot_tok = torch.cat([seq_tok, struct_tok], dim=1)  # (B, L', 1024)
        prot_mask = torch.cat([seq_mask, struct_mask], dim=1)  # (B, L')
        return prot_tok, prot_mask

    def build_prefix(self, protein_tokens: torch.Tensor) -> torch.Tensor:
        # Per-token MLP to LLM hidden size
        return self.prefix_mlp(protein_tokens)  # (B, L', H)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor], aa_seq: List[Optional[str]], stru_str: List[Optional[str]], ) -> torch.Tensor:
        # Text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        # Token-level protein prefix
        prot_tok, prot_mask = self.encode_protein_batch(aa_seq, stru_str)  # (B, L', 1024), (B, L')
        pref = self.build_prefix(prot_tok)  # (B, L', H)

        # Concat prefix + text and merge masks
        inputs_embeds = torch.cat([pref, text_embeds], dim=1)  # (B, L'+T, H)
        attn = torch.cat([prot_mask.to(attention_mask.dtype), attention_mask], dim=1)

        # Extend labels with -100 for prefix (if labels provided)
        new_labels = None
        if labels is not None:
            pad = torch.full((labels.size(0), pref.size(1)), -100, dtype=labels.dtype, device=labels.device)
            new_labels = torch.cat([pad, labels], dim=1)

        # Qwen adds position ids internally; no manual position_ids needed
        out = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=new_labels, use_cache=False)
        return out

    def _export_config(self):
        """Export config purely based on stored attributes (no module inspection)."""
        return {"_class_name": "PLLM", "model_type": "protein_llm_wrapper", "base_model_name_or_path": getattr(self.llm, "name_or_path", None),
                "hidden_size": int(getattr(self, "hidden_size", 0)), "prefix_len": int(getattr(self, "prefix_len", 1)),
                "proj_hid": int(getattr(self, "proj_hid", 1024) if hasattr(self, "proj_hid") else int(getattr(getattr(self, "prefix_mlp", None), "out_hidden", 1024))),
                "single_token_prefix": bool(getattr(self, "prefix_len", 1) == 1),
                "protein_config": getattr(self, "protein_config", getattr(getattr(self, "protein_encoder", None), "config_path", None)),
                "structure_config": getattr(self, "structure_config", getattr(getattr(self, "structure_encoder", None), "config_path", None)),  # rely on stored attribute
                "train_encoders": bool(getattr(self, "train_encoders", True)), }

    @classmethod
    def _construct_from_config(cls, cfg: dict, base_path: str = None):
        """Rebuild the PLLM wrapper from serialized config.json."""
        base_dir = Path(base_path) if base_path is not None else None

        def _res(x):
            if x is None or base_dir is None:
                return x
            return _resolve_rel(base_dir, x)

        # Note: We don't pass protrek_ckpt here because encoder weights are loaded from model.safetensors
        return cls(model_name=cfg.get("base_model_name_or_path", None), protein_config=_res(cfg.get("protein_config", None)),
                   structure_config=_res(cfg.get("structure_config", None)), protrek_ckpt=None,  # Weights loaded from safetensors
                   single_token_prefix=cfg.get("single_token_prefix", False), prefix_len=int(cfg.get("prefix_len", 4)),
                   proj_hid=int(cfg.get("proj_hid", 1024)), train_encoders=bool(cfg.get("train_encoders", True)), dtype_str="auto", )

    def save_pretrained(self, save_directory: str):
        """Save PLLM in Hugging Face format."""
        p = Path(save_directory)
        p.mkdir(parents=True, exist_ok=True)

        # Save base LLM + tokenizer
        (p / "llm").mkdir(exist_ok=True)
        try:
            self.llm.save_pretrained(p / "llm", safe_serialization=True)
        except TypeError:
            self.llm.save_pretrained(p / "llm")
        self.tokenizer.save_pretrained(p / "llm")

        # Save encoders & prefix
        sd = _pllm_gather_custom_state_dict(self)
        if _pllm_save_safetensors is None:
            raise RuntimeError("safetensors is required to save custom PLLM weights. Please `pip install safetensors`.")
        _pllm_save_safetensors(sd, str(p / "model.safetensors"))

        # Copy configs for portability
        cfg = self._export_config()
        prot_rel = _maybe_copy_into(p, cfg.get("protein_config", None), "protein_config.json")
        stru_rel = _maybe_copy_into(p, cfg.get("structure_config", None), "structure_config.json")
        
        if prot_rel is not None:
            cfg["protein_config"] = prot_rel
        if stru_rel is not None:
            cfg["structure_config"] = stru_rel
        # Note: We don't copy protrek_ckpt - encoder weights are already in model.safetensors

        with open(p / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        if not (p / "README.md").exists():
            (p / "README.md").write_text("# PLLM\n\nHugging Face–style package for a protein-augmented LLM wrapper.\n")

    @classmethod
    def from_pretrained(cls, load_path: str, **kwargs):
        """Load PLLM from either a local directory or a Hugging Face Hub repo."""
        p = _as_local_path(load_path)
        cfg = json.loads((p / "config.json").read_text())

        # Rebuild wrapper
        model = cls._construct_from_config(cfg, base_path=str(p))

        # Load base LLM + tokenizer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model.llm = AutoModelForCausalLM.from_pretrained(p / "llm", low_cpu_mem_usage=True, **kwargs)
        model.tokenizer = AutoTokenizer.from_pretrained(p / "llm", use_fast=True)
        if getattr(model.tokenizer, "pad_token", None) is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token

        # Load custom safetensors
        if _pllm_safe_open is None:
            raise RuntimeError("safetensors is required to load custom PLLM weights. Please `pip install safetensors`.")
        custom_tensors = {}
        with _pllm_safe_open(p / "model.safetensors", framework="pt", device="cpu") as f:
            for k in f.keys():
                custom_tensors[k] = f.get_tensor(k)

        # Split back into submodules
        pe = {k.split("protein_encoder.", 1)[1]: v for k, v in custom_tensors.items() if k.startswith("protein_encoder.")}
        se = {k.split("structure_encoder.", 1)[1]: v for k, v in custom_tensors.items() if k.startswith("structure_encoder.")}
        pm = {k.split("prefix_mlp.", 1)[1]: v for k, v in custom_tensors.items() if k.startswith("prefix_mlp.")}

        if getattr(model, "protein_encoder", None) is not None and pe:
            model.protein_encoder.load_state_dict(pe, strict=False)
        if getattr(model, "structure_encoder", None) is not None and se:
            model.structure_encoder.load_state_dict(se, strict=False)
        if getattr(model, "prefix_mlp", None) is not None and pm:
            model.prefix_mlp.load_state_dict(pm, strict=False)
        
        # Ensure dtype consistency across all components
        llm_dtype = next(model.llm.parameters()).dtype
        if getattr(model, "protein_encoder", None) is not None:
            model.protein_encoder = model.protein_encoder.to(dtype=llm_dtype)
        if getattr(model, "structure_encoder", None) is not None:
            model.structure_encoder = model.structure_encoder.to(dtype=llm_dtype)
        if getattr(model, "prefix_mlp", None) is not None:
            model.prefix_mlp = model.prefix_mlp.to(dtype=llm_dtype)

        return model
