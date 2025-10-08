#!/usr/bin/env python3
# (see file header in previous cell for full description)
import os, sys, json, time, math, argparse, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    ShardedStateDictConfig,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from inspect import signature

from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


import protein_encoder as protein_encoder_mod
import structure_encoder as structure_encoder_mod

DTYPE_MAP = {
    "fp32": torch.float32, "float32": torch.float32,
    "fp16": torch.float16, "float16": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "auto": None, "default": None, None: None
}
def resolve_dtype(dtype_str):
    if isinstance(dtype_str, torch.dtype):
        return dtype_str
    key = str(dtype_str).lower() if dtype_str is not None else None
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype {dtype_str}. Use one of: fp32, fp16, bf16, auto.")
    return DTYPE_MAP[key]

def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def is_main() -> bool:
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

@dataclass
class CollateCfg: tokenizer: Any; max_len: int

class JsonlStream(IterableDataset):
    def __init__(self, path: str): super().__init__(); self.path = path
    def __iter__(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        with open(self.path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip(): continue
                if (i % world) != rank: continue
                ex = json.loads(line)
                if "prompt" not in ex or "response" not in ex: continue
                yield {"prompt": ex["prompt"], "response": ex["response"],
                       "aa_seq": ex.get("aa_seq"), "stru_str": ex.get("stru_str")}

class PadAndMaskCollator:
    def __init__(self, cfg: CollateCfg): self.tok = cfg.tokenizer; self.max_len = cfg.max_len
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts  = [b["prompt"]  for b in batch]
        replies  = [b["response"] for b in batch]
        aa_list  = [b.get("aa_seq") for b in batch]
        stru_list= [b.get("stru_str") for b in batch]
        t_prompt = self.tok(prompts, add_special_tokens=False)
        t_reply  = self.tok(replies, add_special_tokens=False)
        input_ids, labels = [], []
        for p_ids, r_ids in zip(t_prompt["input_ids"], t_reply["input_ids"]):
            ids = p_ids + [self.tok.eos_token_id] + r_ids + [self.tok.eos_token_id]
            Lp  = len(p_ids) + 1
            lab = [-100]*Lp + r_ids + [self.tok.eos_token_id]
            input_ids.append(ids[:self.max_len]); labels.append(lab[:self.max_len])
        enc = self.tok.pad({"input_ids": input_ids}, padding="max_length", max_length=self.max_len, return_tensors="pt")
        ids, attn = enc.input_ids, enc.attention_mask
        maxT = ids.shape[1]
        padded_labels = torch.full((len(labels), maxT), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            L = min(len(lab), maxT); padded_labels[i, :L] = torch.tensor(lab[:L], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn, "labels": padded_labels,
                "aa_seq": aa_list, "stru_str": stru_list}

class PrefixProjector(nn.Module):
    def __init__(self, in_dim: int, mid_dim: int, out_hidden: int, out_tokens: int, dropout: float=0.1, dtype=None):
        super().__init__(); self.out_tokens = out_tokens; self.out_hidden = out_hidden
        self.net = nn.Sequential(nn.Linear(in_dim, mid_dim, bias=True), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mid_dim, out_hidden*out_tokens, bias=True))
        if dtype is not None: self.to(dtype=dtype)
    def forward(self, protein_vec: torch.Tensor) -> torch.Tensor:
        x = self.net(protein_vec); B = x.size(0); T = self.out_tokens; H = self.out_hidden
        return x.view(B, T, H)

class BigProteinQwen(nn.Module):
    def __init__(self, model_name: str, protein_config: str, structure_config: str, protrek_ckpt: str=None,
                 prot_slot: int=1, stru_slot: int=3, single_token_prefix: bool=False, prefix_len: int=4,
                 proj_hid: int=1024, dropout: float=0.1, train_encoders: bool=False, load_dtype=torch.bfloat16):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=load_dtype, low_cpu_mem_usage=True)
        if hasattr(self.llm.config, "use_cache"): self.llm.config.use_cache = False
        self.llm.gradient_checkpointing_enable()

        try:
            if hasattr(self.llm.config, "tie_word_embeddings"):
                self.llm.config.tie_word_embeddings = False
            lm = getattr(self.llm, "lm_head", None)
            emb = self.llm.get_input_embeddings()
            if lm is not None and emb is not None and hasattr(lm, "weight") and hasattr(emb, "weight"):
                if lm.weight.data_ptr() == emb.weight.data_ptr():
                    with torch.no_grad():
                        lm.weight = nn.Parameter(lm.weight.detach().clone())
                    if hasattr(self.llm, "_tie_or_clone_weights"): self.llm._tie_or_clone_weights = lambda *a, **k: None
                    if hasattr(self.llm, "tie_weights"):           self.llm.tie_weights           = lambda *a, **k: None
                    print("[tie-fix] Untied lm_head <-> embed_tokens")
        except Exception as e:
            print("[tie-fix] skipped:", e)

        self.hidden_size = self.llm.config.hidden_size
        self.prefix_len = 1 if single_token_prefix else prefix_len

        self.protein_encoder   = protein_encoder_mod.ProteinEncoder(protein_config, out_dim=1024, load_pretrained=False)
        self.structure_encoder = structure_encoder_mod.StructureEncoder(structure_config, out_dim=1024, load_pretrained=False)

        if protrek_ckpt and os.path.exists(protrek_ckpt):
            sd_raw = torch.load(protrek_ckpt, map_location="cpu")
            sd = sd_raw.get("model", sd_raw.get("state_dict", sd_raw))
            slots = {}
            for k, v in sd.items():
                head = k.split(".", 1)[0]
                if head.isdigit(): slots.setdefault(int(head), {})[k[len(head)+1:]] = v
            def drop_extras(sub): return {k: v for k, v in sub.items() if "embeddings.position_ids" not in k}
            if prot_slot in slots:
                mp, up = self.protein_encoder.load_state_dict(drop_extras(slots[prot_slot]), strict=False)
                print(f"[ProteinEncoder] loaded from slot {prot_slot} | missing={len(mp)} unexpected={len(up)}")
            if stru_slot in slots:
                ms, us = self.structure_encoder.load_state_dict(drop_extras(slots[stru_slot]), strict=False)
                print(f"[StructureEncoder] loaded from slot {stru_slot} | missing={len(ms)} unexpected={len(us)}")
        else:
            print("No ProTrek checkpoint provided or path not found; encoders stay random-init.")

        model_dtype = next(self.llm.parameters()).dtype
        self.proj_2048 = PrefixProjector(2048, proj_hid, self.hidden_size, self.prefix_len, dropout, dtype=model_dtype)
        self.protein_encoder.to(model_dtype); self.structure_encoder.to(model_dtype)
        if not train_encoders:
            for p in self.protein_encoder.parameters():   p.requires_grad = False
            for p in self.structure_encoder.parameters(): p.requires_grad = False

    def encode_protein_batch(self, aa_list: List[Optional[str]], stru_list: List[Optional[str]]) -> torch.Tensor:
        B = len(aa_list); device = next(self.llm.parameters()).device; dtype = next(self.llm.parameters()).dtype
        seq_out  = torch.zeros(B, 1024, device=device, dtype=dtype)
        stru_out = torch.zeros(B, 1024, device=device, dtype=dtype)
        idx_seq  = [i for i, a in enumerate(aa_list)   if a is not None and len(a) > 0]
        idx_stru = [i for i, s in enumerate(stru_list) if s is not None and len(s) > 0]
        if len(idx_seq) > 0:
            seqs = [aa_list[i] for i in idx_seq]; emb = self.protein_encoder.get_repr(seqs, batch_size=max(1, len(seqs)), verbose=False)
            emb = torch.as_tensor(emb, device=device, dtype=dtype); seq_out[idx_seq, :] = emb
        if len(idx_stru) > 0:
            strs = [stru_list[i] for i in idx_stru]; emb = self.structure_encoder.get_repr(strs, batch_size=max(1, len(strs)), verbose=False)
            emb = torch.as_tensor(emb, device=device, dtype=dtype); stru_out[idx_stru, :] = emb
        return torch.cat([seq_out, stru_out], dim=-1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor,
                aa_seq: List[Optional[str]], stru_str: List[Optional[str]]) -> torch.Tensor:
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        pref = self.proj_2048(self.encode_protein_batch(aa_seq, stru_str))
        inputs_embeds  = torch.cat([pref, text_embeds], dim=1)
        if not inputs_embeds.requires_grad:
            inputs_embeds.requires_grad_()
        pref_attn      = torch.ones(inputs_embeds.size(0), pref.size(1), dtype=attention_mask.dtype, device=attention_mask.device)
        attn           = torch.cat([pref_attn, attention_mask], dim=1)
        pad            = torch.full((labels.size(0), pref.size(1)), -100, dtype=labels.dtype, device=labels.device)
        new_labels     = torch.cat([pad, labels], dim=1)
        return self.llm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=new_labels, use_cache=False)

def human_time(s: float) -> str:
    m, s = divmod(int(s), 60); h, m = divmod(m, 60)
    return (f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s"))

def is_main():
    return (not dist.is_initialized()) or (dist.get_rank() == 0)

def train(args):
    # dist init
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    set_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    train_ds = JsonlStream(args.train_file)
    collate  = PadAndMaskCollator(CollateCfg(tokenizer=tok, max_len=args.max_len))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate)

    load_dtype = resolve_dtype(args.dtype)
    if load_dtype is None:
        load_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    big = BigProteinQwen(
        model_name=args.model_name,
        protein_config=args.protein_config,
        structure_config=args.structure_config,
        protrek_ckpt=args.protrek_ckpt,
        prot_slot=args.prot_slot, stru_slot=args.stru_slot,
        single_token_prefix=args.single_token_prefix, prefix_len=args.prefix_len,
        proj_hid=args.proj_hid, dropout=args.dropout,
        train_encoders=args.train_encoders, load_dtype=load_dtype
    ).to(device)
    # Required with HF models when using activation checkpointing
    big.llm.config.use_cache = False

    # Enable activation checkpointing with the non-reentrant path
    big.llm.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # Let embeddings (or provided inputs_embeds) participate in autograd
    if hasattr(big.llm, "enable_input_require_grads"):
        big.llm.enable_input_require_grads()

    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        layer_cls = {Qwen2DecoderLayer}
        auto_wrap_policy = transformer_auto_wrap_policy(layer_cls)
    except Exception:
        auto_wrap_policy = None

    mp = None
    if load_dtype in (torch.bfloat16, torch.float16):
        mp = MixedPrecision(param_dtype=load_dtype, reduce_dtype=load_dtype, buffer_dtype=load_dtype)

    for i, block in enumerate(big.llm.model.layers):
        big.llm.model.layers[i] = checkpoint_wrapper(
            block, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
    fsdp_model = FSDP(
        big,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp,
        use_orig_params=True,
        forward_prefetch=False,
        #reshard_after_forward=False,
        backward_prefetch=BackwardPrefetch.BACKWARD_POST,
        limit_all_gathers=True,
        sync_module_states=True,
        device_id=device,
    )

    params = [p for p in fsdp_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = args.steps_per_epoch or max(1, 1000 // max(1, args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * args.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup, num_training_steps=total_steps)

    wandb = None
    if args.wandb and is_main():
        try:
            import wandb as _wandb
            if args.wandb_mode:
                os.environ["WANDB_MODE"] = "disabled" if args.wandb_mode=="disabled" else args.wandb_mode
            _wandb.init(project=args.wandb_project or "bioreasoner",
                        name=args.wandb_run_name, entity=args.wandb_entity,
                        config=vars(args),
                        tags=[t.strip() for t in args.wandb_tags.split(",")] if args.wandb_tags else None)
            wandb = _wandb
            print("[wandb] initialized.")
        except Exception as e:
            print("[wandb] init failed:", e); wandb = None

    fsdp_model.train()
    global_step = 0; t0 = time.time(); it = 0
    for epoch in range(args.epochs):
        for batch in train_loader:
            for k in ("input_ids","attention_mask","labels"):
                batch[k] = batch[k].to(device, non_blocking=True)
            out = fsdp_model(**batch)
            loss = out.loss / max(1, args.accum_steps)
            loss.backward()
            it += 1
            if it % args.accum_steps == 0:
                try:
                    FSDP.clip_grad_norm_(fsdp_model.parameters(), max_norm=1.0)
                except Exception:
                    torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); global_step += 1

                if is_main() and args.log_every and (global_step % args.log_every == 0):
                    dt = time.time() - t0
                    print(f"[epoch {epoch+1}] step {global_step} | loss={loss.item()*args.accum_steps:.4f} | {dt:.1f}s")
                    if wandb is not None:
                        try:
                            lr_val = scheduler.get_last_lr()[0] if scheduler and scheduler.optimizer.param_groups else None
                            wandb.log({"train/loss": float(loss.item()*args.accum_steps), "train/step": int(global_step),
                                       "train/epoch": int(epoch+1), "train/lr": float(lr_val) if lr_val is not None else None}, step=int(global_step))
                        except Exception as _e: print("[wandb] log failed:", _e)

            if global_step >= (epoch+1)*steps_per_epoch:
                break
        if dist.is_initialized(): dist.barrier()

    if is_main(): os.makedirs(args.save_dir, exist_ok=True)
    if dist.is_initialized(): dist.barrier()

    with FSDP.state_dict_type(fsdp_model, StateDictType.SHARDED_STATE_DICT, ShardedStateDictConfig(offload_to_cpu=True)):
        state = fsdp_model.state_dict()
    if is_main():
        final_path = os.path.join(args.save_dir, "model_sharded.pt")
        torch.save(state, final_path); print("Saved FINAL (sharded):", final_path)
        if wandb is not None:
            try:
                wandb.summary["final/step"] = int(global_step); wandb.summary["final/epoch"] = int(args.epochs); wandb.summary["final/save_path"] = final_path; wandb.finish()
            except Exception as _e: print("[wandb] summary/finish failed:", _e)

    if dist.is_initialized(): dist.destroy_process_group()

def parse_args():
    p = argparse.ArgumentParser("Vanilla-FSDP Protein-conditioned SFT (no Accelerate)")
    p.add_argument("--train-file", type=str, required=True)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--steps-per-epoch", type=int, default=0)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--protein-config", type=str, required=True)
    p.add_argument("--structure-config", type=str, required=True)
    p.add_argument("--protrek-ckpt", type=str, default=None)
    p.add_argument("--prot-slot", type=int, default=1)
    p.add_argument("--stru-slot", type=int, default=3)
    p.add_argument("--single-token-prefix", action="store_true")
    p.add_argument("--prefix-len", type=int, default=4)
    p.add_argument("--proj-hid", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--train-encoders", action="store_true")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--save-dir", type=str, default="./runs_fsdp")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default=None, choices=["online","offline","disabled",None])
    p.add_argument("--wandb-tags", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bf16", choices=["auto","fp32","fp16","bf16","float32","float16","bfloat16","default"])
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
