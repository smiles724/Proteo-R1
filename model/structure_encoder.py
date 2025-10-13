import torch
import torch.nn as nn
from typing import List, Optional, Union
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer


class StructureEncoder(nn.Module):
    """
    Token-level structure encoder (e.g., Foldseek 3Di strings) using HF ESM.
    Returns padded token representations (B, L_max, D) and a boolean mask over tokens.
    """
    def __init__(
        self,
        config_path: str,
        out_dim: int,
        load_pretrained: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.tokenizer: EsmTokenizer = EsmTokenizer.from_pretrained(config_path)
        if load_pretrained:
            self.model: EsmForMaskedLM = EsmForMaskedLM.from_pretrained(config_path)
        else:
            cfg = EsmConfig.from_pretrained(config_path)
            self.model = EsmForMaskedLM(cfg)

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        hidden = self.model.config.hidden_size
        self.proj = nn.Linear(hidden, out_dim, bias=False)

        # cache IDs
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.sep_token_id

    @torch.no_grad()
    def _encode_batch(
        self,
        sequences: List[str],
        return_tensors: str = "pt",
        device: Optional[Union[str, torch.device]] = None,
        **encode_kwargs,
    ):
        batch = self.tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,    # adds BOS/EOS
            padding=True,
            truncation=False,
            return_tensors=return_tensors,
            **encode_kwargs,
        )
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        outputs = self.model.esm(**batch, output_hidden_states=False, return_dict=True)
        last_hidden_state = outputs.last_hidden_state  # (B, L_full, H)
        return batch, last_hidden_state

    def forward(
        self,
        sequences: List[str],
        get_mask_logits: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Args:
          sequences: list of structure strings (e.g., Foldseek 3Di tokens)
        Returns:
          padded_out: (B, L_max_res, D)
          mask_out:   (B, L_max_res) boolean (True for valid tokens, excl. BOS/EOS)
          mask_logits:(B, L_full, V) or None
        """
        batch, last_hidden_state = self._encode_batch(sequences, device=device)
        full_lens = batch["attention_mask"].sum(dim=1)             # includes BOS/EOS
        B = last_hidden_state.size(0)

        # remove BOS/EOS to get residue tokens
        residue_reprs = []
        for i in range(B):
            L_i = int(full_lens[i].item())
            start, end = 1, max(1, L_i - 1)
            residue_reprs.append(last_hidden_state[i, start:end, :])

        padded = pad_sequence(residue_reprs, batch_first=True)     # (B, L_max, H)
        Lmax = padded.size(1)
        idxs = torch.arange(Lmax, device=padded.device).unsqueeze(0).expand(B, Lmax)
        residue_lens = (full_lens - 2).clamp_min(0)
        mask = idxs < residue_lens.unsqueeze(1)                    # (B, L_max)

        padded = normalize(padded, dim=-1)
        padded = padded * mask.unsqueeze(-1)
        padded_out = self.proj(padded)                             # (B, L_max, D)

        mask_logits = self.model.lm_head(last_hidden_state) if get_mask_logits else None
        return padded_out, mask, mask_logits

    def get_repr(self, proteins: List[str], batch_size: int = 64, verbose: bool = False):
        """Backward-compatible utility: returns (emb, mask, None) with token-level reps."""
        device = next(self.parameters()).device
        if isinstance(proteins, str):
            proteins = [proteins]
        chunks, masks = [], []
        iterator = range(0, len(proteins), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Computing structure token embeddings")
        for i in iterator:
            emb, mask, _ = self.forward(proteins[i:i+batch_size], device=device)
            chunks.append(emb)
            masks.append(mask)
        return torch.cat(chunks, dim=0), torch.cat(masks, dim=0), None

if __name__ == "__main__":
    # Example usage for STRUCTURE encoder (Foldseek local model)
    import os
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="StructureEncoder demo (Foldseek local)")
    parser.add_argument(
        "--structure-config",
        default="/protrek/weights/ProTrek_35M/foldseek_t12_35M",
        help="Local path to the Foldseek model directory (HF-style folder with config + weights).",
    )
    parser.add_argument(
        "--out-dim",
        type=int,
        default=256,
        help="Projection output dimension per token.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run on; 'auto' picks CUDA if available.",
    )
    parser.add_argument(
        "--get-mask-logits",
        action="store_true",
        help="Also return MLM logits for masked tokens (for inspection).",
    )
    args = parser.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    assert os.path.exists(args.structure_config), (
        f"Structure config path not found: {args.structure_config}\n"
        "Please point --structure-config to your local Foldseek model directory."
    )

    enc = StructureEncoder(
        config_path=args.structure_config,
        out_dim=args.out_dim,
        load_pretrained=False,
        gradient_checkpointing=False,
    ).to(device)

    # ---- Replace these with your real Foldseek 3Di strings ----
    # These are placeholders; use actual 3Di token sequences from your data.
    stru_seqs = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNP"]

    reps, mask, logits = enc(stru_seqs, get_mask_logits=args.get_mask_logits, device=device)

    # Shapes:
    #   reps:   (B, L_max_residue, out_dim)
    #   mask:   (B, L_max_residue)  [True for valid residue tokens; BOS/EOS removed]
    #   logits: (B, L_full_incl_specials, vocab_size) if --get-mask-logits
    print("Representations:", reps.shape)
    print("Mask:", None if mask is None else mask.shape)
    if logits is not None:
        print("Mask logits:", logits.shape)

    # Optional sanity checks
    if mask is not None:
        valid_lens = mask.sum(dim=1).tolist()
        print("Valid residue lengths (per item):", valid_lens)

    # Show head parameter shape (proj: H_esm -> out_dim)
    print("proj.weight shape:", tuple(enc.proj.weight.shape))