#!/usr/bin/env python
# coding: utf-8
"""
SFT for BigProteinQwen (multi-GPU via Accelerate) with a clear train(args) entry point
and optional Weights & Biases tracking.

Dataset JSONL: prompt, response, aa_seq, stru_str
Loss: only response tokens (prompt/EOS masked with -100)
Saves HF from_pretrained-style checkpoint:
  - sharded safetensors (model-00001-of-000NN.safetensors via --max-shard-size)
  - tokenizer files
  - preprocessor_config.json

Run (multi-GPU):
  accelerate launch sft_accelerate.py \
    --dataset /path/to/train.jsonl \
    --output_dir ./checkpoints/bigprotein-sft \
    --model_name Qwen/Qwen2.5-0.5B-Instruct \
    --protein_config facebook/esm2_t12_35M_UR50D \
    --structure_config facebook/esm2_t12_35M_UR50D \
    --num_epochs 1 --batch_size 4 --lr 5e-5 --max_length 1024 \
    --max-shard-size 1GB --mixed_precision bf16 \
    --wandb --wandb_project bigprotein --wandb_run_name exp1 --wandb_mode online
"""

import os
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    PretrainedConfig,
    PreTrainedModel,
)

# Your modules (must be colocated or in PYTHONPATH)
import proteinLLM as bigmod  # defines BigProteinQwen


# -------------------------------
# Config + HF wrapper for saving
# -------------------------------
class BigProteinConfig(PretrainedConfig):
    model_type = "bigprotein-qwen"

    def __init__(
        self,
        llm_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        protein_config: str = "facebook/esm2_t12_35M_UR50D",
        structure_config: str = "facebook/esm2_t12_35M_UR50D",
        proj_hid: int = 1024,
        dropout: float = 0.1,
        train_encoders: bool = False,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_name = llm_name
        self.protein_config = protein_config
        self.structure_config = structure_config
        self.proj_hid = proj_hid
        self.dropout = dropout
        self.train_encoders = train_encoders
        self.extra_kwargs = extra_kwargs or {}


class BigProteinForCausalLM(PreTrainedModel):
    """Thin wrapper so save_pretrained/from_pretrained handle ALL weights (LLM+encoders+projector)."""
    config_class = BigProteinConfig

    def __init__(self, config: BigProteinConfig):
        super().__init__(config)
        self.model = bigmod.BigProteinQwen(
            model_name=config.llm_name,
            protein_config=config.protein_config,
            structure_config=config.structure_config,
            protrek_ckpt=config.extra_kwargs.get("protrek_ckpt"),
            prot_slot=config.extra_kwargs.get("prot_slot", 1),
            stru_slot=config.extra_kwargs.get("stru_slot", 3),
            single_token_prefix=config.extra_kwargs.get("single_token_prefix", False),
            prefix_len=config.extra_kwargs.get("prefix_len", 4),
            proj_hid=config.proj_hid,
            dropout=config.dropout,
            train_encoders=config.train_encoders,
            dtype_str=config.extra_kwargs.get("dtype", "auto"),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        aa_seq: List[Optional[str]],
        stru_str: List[Optional[str]],
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            aa_seq=aa_seq,
            stru_str=stru_str,
        )


# ---------------------------
# Dataset & Collator
# ---------------------------
class JsonlSFTDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "prompt" in obj and "response" in obj:
                    self.samples.append({
                        "prompt": obj["prompt"],
                        "response": obj["response"],
                        "aa_seq": obj.get("aa_seq"),
                        "stru_str": obj.get("stru_str"),
                    })
        if len(self.samples) == 0:
            raise ValueError(f"No usable records found in {path}.")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


@dataclass
class CollateCfg:
    tokenizer: Any
    max_length: int = 1024
    append_eos: bool = True


class PadAndMaskCollator:
    """Mask prompt (and EOS) with -100 so loss computes only on response tokens."""
    def __init__(self, cfg: CollateCfg):
        self.cfg = cfg
        self.tok = cfg.tokenizer
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        eos_id = self.tok.eos_token_id
        inps, labels = [], []
        aa_list, stru_list = [], []

        for ex in batch:
            p = ex["prompt"] or ""
            r = ex["response"] or ""
            p_ids = self.tok.encode(p, add_special_tokens=False)
            r_ids = self.tok.encode(r, add_special_tokens=False)
            ids = p_ids + r_ids
            la  = [-100] * len(p_ids) + r_ids
            if self.cfg.append_eos:
                ids += [eos_id]; la += [-100]   # mask EOS too

            # truncate
            ids = ids[:self.cfg.max_length]
            la  = la[:self.cfg.max_length]

            inps.append(ids)
            labels.append(la)
            aa_list.append(ex.get("aa_seq"))
            stru_list.append(ex.get("stru_str"))

        maxL = min(self.cfg.max_length, max(len(x) for x in inps))

        def pad_to(x, pad_val): return x + [pad_val] * (maxL - len(x))
        input_ids = torch.tensor([pad_to(x, self.tok.pad_token_id) for x in inps], dtype=torch.long)
        labels    = torch.tensor([pad_to(x, -100) for x in labels], dtype=torch.long)
        attention = torch.tensor([[1]*len(x) + [0]*(maxL-len(x)) for x in inps], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "aa_seq": aa_list,
            "stru_str": stru_list,
        }


# ---------------------------
# Argparse and helpers
# ---------------------------
def get_args():
    p = argparse.ArgumentParser(description="SFT for BigProteinQwen with Accelerate")
    # Data
    p.add_argument("--dataset", type=str, required=True, help="Path to train JSONL.")
    p.add_argument("--valset", type=str, default=None, help="Optional validation JSONL.")
    p.add_argument("--output_dir", type=str, required=True)
    # Models
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--protein_config", type=str, default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--structure_config", type=str, default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--protrek_ckpt", type=str, default=None)
    p.add_argument("--prot_slot", type=int, default=1)
    p.add_argument("--stru_slot", type=int, default=3)
    p.add_argument("--train_encoders", action="store_true", help="If set, encoders are trainable.")
    # Train
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no","fp16","bf16"])
    # Logging
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging via Accelerate.")
    p.add_argument("--wandb_project", type=str, default="bigprotein-sft")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default=None, choices=[None, "online", "offline", "disabled"])
    # Save
    p.add_argument("--save_every_steps", type=int, default=0, help="0 = only final save.")
    p.add_argument("--max-shard-size", type=str, default="1GB",
                   help='HF shard size like "1GB" → files named "model-00001-of-000NN.safetensors"')
    return p.parse_args()


def save_preprocessor_config(out_dir: str, cfg: CollateCfg):
    pre = {
        "feature_extractor_type": "BigProteinPreprocessor",
        "task": "protein-text-conditional-generation",
        "text_field": "prompt",
        "response_field": "response",
        "sequence_field": "aa_seq",
        "structure_field": "stru_str",
        "mask_prompt": True,
        "append_eos_to_response": cfg.append_eos,
        "max_length": cfg.max_length,
    }
    with open(os.path.join(out_dir, "preprocessor_config.json"), "w", encoding="utf-8") as f:
        json.dump(pre, f, indent=2)


# ---------------------------
# Train entry point (W&B supported)
# ---------------------------
def train(args) -> None:
    # pick logger backend dynamically
    log_with = None
    if args.wandb:
        try:
            import wandb  # noqa: F401
            log_with = "wandb"
        except Exception as e:
            print(f"[warn] W&B not available ({e}); continuing without W&B.")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=None if args.mixed_precision=="no" else args.mixed_precision,
        log_with=log_with,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize W&B tracker if requested
    if log_with == "wandb":
        init_kwargs = {"wandb": {}}
        if args.wandb_run_name: init_kwargs["wandb"]["name"] = args.wandb_run_name
        if args.wandb_entity:   init_kwargs["wandb"]["entity"] = args.wandb_entity
        if args.wandb_mode:     init_kwargs["wandb"]["mode"] = args.wandb_mode  # "online"/"offline"/"disabled"
        accelerator.init_trackers(args.wandb_project, config=vars(args), init_kwargs=init_kwargs)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Config + Model
    bp_config = BigProteinConfig(
        llm_name=args.model_name,
        protein_config=args.protein_config,
        structure_config=args.structure_config,
        proj_hid=1024,
        dropout=0.10,
        train_encoders=args.train_encoders,
        extra_kwargs={
            "protrek_ckpt": args.protrek_ckpt,
            "prot_slot": args.prot_slot,
            "stru_slot": args.stru_slot,
            "dtype": "auto",
            "single_token_prefix": False,
            "prefix_len": 4,
        },
    )
    model = BigProteinForCausalLM(bp_config)

    # Data
    train_ds = JsonlSFTDataset(args.dataset)
    val_ds   = JsonlSFTDataset(args.valset) if args.valset else None
    collate  = PadAndMaskCollator(CollateCfg(tokenizer, args.max_length, append_eos=True))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate) if val_ds else None

    # Optimizer / Scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Prepare for multi-GPU
    model, optim, train_loader, sched = accelerator.prepare(model, optim, train_loader, sched)
    if val_loader:
        val_loader = accelerator.prepare(val_loader)

    model.train()
    global_step = 0
    t0 = time.time()

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_loader, start=1):
            with accelerator.accumulate(model):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    aa_seq=batch["aa_seq"],
                    stru_str=batch["stru_str"],
                )
                loss = out.loss
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.clip_grad and args.clip_grad > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad)
                optim.step(); sched.step(); optim.zero_grad(set_to_none=True)

            global_step += 1
            if accelerator.is_main_process and (global_step % args.log_interval == 0 or step == 1):
                elapsed = time.time() - t0
                lr = sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") else args.lr
                print(f"[epoch {epoch+1}] step {global_step}/{total_steps}  loss={loss.item():.4f}  lr={lr:.3e}  elapsed={elapsed:.1f}s")
            # Log to trackers (works with W&B when enabled)
            accelerator.log({"train/loss": float(loss.item()),
                             "train/lr": float(sched.get_last_lr()[0]) if hasattr(sched, "get_last_lr") else float(args.lr),
                             "train/epoch": float(epoch+1)}, step=global_step)

            # optional periodic checkpoint
            if accelerator.is_main_process and args.save_every_steps and (global_step % args.save_every_steps == 0):
                unwrapped = accelerator.unwrap_model(model)
                save_dir = os.path.join(args.output_dir, f"step-{global_step:06d}")
                os.makedirs(save_dir, exist_ok=True)
                unwrapped.save_pretrained(save_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
                tokenizer.save_pretrained(save_dir)
                save_preprocessor_config(save_dir, collate.cfg)
                print(f"Saved intermediate checkpoint to {save_dir}")
                accelerator.log({"ckpt/step": global_step}, step=global_step)

        # quick eval
        if val_loader:
            model.eval()
            tot, cnt = 0.0, 0
            with torch.no_grad():
                for vb in val_loader:
                    vo = model(
                        input_ids=vb["input_ids"],
                        attention_mask=vb["attention_mask"],
                        labels=vb["labels"],
                        aa_seq=vb["aa_seq"],
                        stru_str=vb["stru_str"],
                    )
                    tot += float(vo.loss.item()) * vb["input_ids"].size(0)
                    cnt += vb["input_ids"].size(0)
            val_loss = tot/max(cnt,1)
            if accelerator.is_main_process:
                print(f"[epoch {epoch+1}] val_loss={val_loss:.4f}")
            accelerator.log({"val/loss": float(val_loss), "val/epoch": float(epoch+1)}, step=global_step)
            model.train()

    # Final save
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size=args.max_shard_size)
        tokenizer.save_pretrained(args.output_dir)
        save_preprocessor_config(args.output_dir, collate.cfg)
        print(f"Saved final checkpoint to {args.output_dir}")
    accelerator.end_training()
    accelerator.wait_for_everyone()


# ---------------------------
# main() just parses and calls train(args)
# ---------------------------
def main():
    args = get_args()
    train(args)


if __name__ == "__main__":
    main()
