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

    def __init__(self, config_path: str, out_dim: int, load_pretrained: bool = True, gradient_checkpointing: bool = False, ):
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
    def _encode_batch(self, sequences: List[str], return_tensors: str = "pt", device: Optional[Union[str, torch.device]] = None, **encode_kwargs, ):
        batch = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True,  # adds BOS/EOS
                                                 padding=True, truncation=False, return_tensors=return_tensors, **encode_kwargs, )
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
        outputs = self.model.esm(**batch, output_hidden_states=False, return_dict=True)
        last_hidden_state = outputs.last_hidden_state  # (B, L_full, H)
        return batch, last_hidden_state

    def forward(self, sequences: List[str], get_mask_logits: bool = False, device: Optional[Union[str, torch.device]] = None, ):
        """
        Args:
          sequences: list of structure strings (e.g., Foldseek 3Di tokens)
        Returns:
          padded_out: (B, L_max_res, D)
          mask_out:   (B, L_max_res) boolean (True for valid tokens, excl. BOS/EOS)
          mask_logits:(B, L_full, V) or None
        """
        batch, last_hidden_state = self._encode_batch(sequences, device=device)
        full_lens = batch["attention_mask"].sum(dim=1)  # includes BOS/EOS
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
        padded_out = self.proj(padded)  # (B, L_max, D)

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
            emb, mask, _ = self.forward(proteins[i:i + batch_size], device=device)
            chunks.append(emb)
            masks.append(mask)
        return torch.cat(chunks, dim=0), torch.cat(masks, dim=0), None
