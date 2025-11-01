import torch
import torch.nn as nn
from typing import List, Optional, Union
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import normalize
from tqdm import tqdm
from transformers import EsmConfig, EsmForMaskedLM

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None


class StructureEncoder(nn.Module):
    """
    Token-level structure encoder (e.g., Foldseek 3Di strings) using HF ESM.
    Returns padded token representations (B, L_max, D) and a boolean mask over tokens.
    """

    def __init__(
            self,
            config_path: str = None,
            out_dim: int = 1024,
            load_pretrained: bool = True,
            gradient_checkpointing: bool = False,
            attn_implementation: str = None
    ):
        super().__init__()
        self.out_dim = out_dim

        if attn_implementation is None:
            attn_implementation = "flash_attention_2" if flash_attn is not None else "sdpa"
            if attn_implementation == "sdpa":
                print(f"[WARN] flash_attention_2 is not activated for {self.__class__.__name__} since flash_attn is not supported!")

        if load_pretrained:
            self.model = EsmForMaskedLM.from_pretrained(
                config_path,
                torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
                attn_implementation=attn_implementation
            )
        else:
            esm_config = EsmConfig.from_pretrained(
                config_path,
                torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
                attn_implementation=attn_implementation
            )
            self.model = EsmForMaskedLM(esm_config)

        self.model.esm.contact_head = None

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        hidden = self.model.config.hidden_size
        # self.proj = nn.Linear(hidden, out_dim, bias=False)  # why initialize a new projection layer?
        self.out = nn.Linear(hidden, out_dim, bias=True)

    # @torch.no_grad()
    def _encode_batch(
            self,
            sequences: List[str],
            return_tensors: str = "pt",
            device: Optional[Union[str, torch.device]] = None,
            **encode_kwargs,
    ):
        batch = self.tokenizer.batch_encode_plus(
            sequences,
            add_special_tokens=True,  # adds BOS/EOS
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
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            get_mask_logits: bool = False,
    ):
        outputs = self.model.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True
        )
        last_hidden_state = outputs.last_hidden_state  # (B, L_full, H)

        full_lens = attention_mask.sum(dim=1)  # includes BOS/EOS
        B = last_hidden_state.size(0)

        # remove BOS/EOS to get residue tokens
        residue_reprs = []
        for i in range(B):
            L_i = int(full_lens[i].item())
            start, end = 1, max(1, L_i - 1)
            residue_reprs.append(last_hidden_state[i, start:end, :])

        padded = pad_sequence(residue_reprs, batch_first=True)  # (B, L_max, H)
        Lmax = padded.size(1)
        idxs = torch.arange(Lmax, device=padded.device).unsqueeze(0).expand(B, Lmax)
        residue_lens = (full_lens - 2).clamp_min(0)
        mask = idxs < residue_lens.unsqueeze(1)  # (B, L_max)

        padded = normalize(padded, dim=-1)
        padded = padded * mask.unsqueeze(-1)
        # padded_out = self.proj(padded)  # (B, L_max, D)
        padded_out = self.out(padded)  # (B, L_max, D)

        mask_logits = self.model.lm_head(last_hidden_state) if get_mask_logits else None
        return padded_out, mask, mask_logits

    def get_repr(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: torch.Tensor = None,
            batch_size: int = None,
            verbose: bool = False
    ):
        """Backward-compatible utility: returns (emb, mask, None) with token-level reps."""
        if batch_size is None:
            batch_size = len(input_ids)

        chunks, masks = [], []
        iterator = range(0, len(input_ids), batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="Computing structure token embeddings")
        for i in iterator:
            emb, mask, _ = self.forward(
                input_ids=input_ids[i:i + batch_size],
                attention_mask=attention_mask[i:i + batch_size],
            )
            chunks.append(emb)
            masks.append(mask)
        return torch.cat(chunks, dim=0), torch.cat(masks, dim=0), None
