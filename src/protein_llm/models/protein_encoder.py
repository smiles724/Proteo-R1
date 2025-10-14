import torch
import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional, Union

from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer


class ProteinEncoder(nn.Module):
    """
    HuggingFace ESM encoder wrapper with proper padding/masking.

    Key behavior:
      - Uses HF EsmTokenizer/EsmForMaskedLM
      - Computes per-sequence lengths from attention_mask
      - For residue-level reps, slices out BOS (CLS) at index 0 and EOS (SEP) at the last index
      - Pads variable-length residue sequences to a uniform length and returns (padded_reprs, mask)
      - Optionally returns masked LM logits for <mask> tokens

    Returns:
      If seq_level_reprs is False (default): (padded_reprs, mask, mask_logits)
        - padded_reprs: (B, L_max_residues, D)
        - mask:         (B, L_max_residues) boolean, True for valid residues
        - mask_logits:  (B, L_full, V) or None

      If seq_level_reprs is True: (seq_reprs, None, mask_logits)
        - seq_reprs:    (B, D) averaged over residues (excl. BOS/EOS)
    """

    def __init__(
            self,
            config_path: str = None,
            out_dim: int = 1024,
            load_pretrained: bool = True,
            gradient_checkpointing: bool = False,
            seq_level_reprs: bool = False,
    ):
        super().__init__()
        self.seq_level_reprs = seq_level_reprs
        self.out_dim = out_dim

        if load_pretrained:
            self.model: EsmForMaskedLM = EsmForMaskedLM.from_pretrained(config_path)
        else:
            esm_config = EsmConfig.from_pretrained(config_path)
            self.model = EsmForMaskedLM(esm_config)

        self.model.esm.contact_head = None

        self._tokenizer = None
        self._tokenizer_path = config_path

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        hidden = self.model.config.hidden_size
        # self.out = nn.Linear(hidden, out_dim, bias=False)
        self.out = nn.Linear(hidden, out_dim, bias=True)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = EsmTokenizer.from_pretrained(self._tokenizer_path)
        return self._tokenizer

    @torch.no_grad()
    def _encode_batch(
            self,
            sequences: List[str],
            return_tensors: str = "pt",
            device: Optional[Union[str, torch.device]] = None,
            **encode_kwargs,
    ) -> Tuple[dict, torch.Tensor]:
        """
        Tokenize and run the ESM encoder. Returns (batch_dict, last_hidden_state).
        batch_dict has: input_ids, attention_mask, (token_type_ids if any).
        last_hidden_state: (B, L_full, H)
        """
        batch = self.tokenizer.batch_encode_plus(sequences, add_special_tokens=True,  # ensures BOS/EOS
                                                 padding=True, truncation=False, return_tensors=return_tensors, **encode_kwargs, )
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
            sequences: list of raw amino-acid sequences (can include <mask> tokens as in HF ESM)
            get_mask_logits: if True, also return MLM logits over the vocabulary for each token
            device: torch device to place tensors on

        Returns:
            See class docstring.
        """
        batch, last_hidden_state = self._encode_batch(sequences, device=device)

        # (B,) lengths including BOS/EOS tokens, derived from attention_mask
        # NOTE: attention_mask==1 for real tokens incl. special tokens
        full_lens = batch["attention_mask"].sum(dim=1)  # shape (B,)

        # Build per-sequence residue-only slices by removing BOS/EOS
        # Residue length = full_len - 2 (clamped at >= 0)
        residue_lens = (full_lens - 2).clamp_min(0)

        B = last_hidden_state.size(0)
        H = last_hidden_state.size(-1)

        # Collect variable-length residue reps (exclude BOS at 0 and EOS at full_len-1)
        residue_reprs = []
        start = 1
        for i in range(B):
            L_i = int(full_lens[i].item())
            end = max(start, L_i - 1)  # exclusive, see ESM official doc
            residue_reprs.append(last_hidden_state[i, start:end, :])

        if self.seq_level_reprs:
            # Average across valid residues for sequence-level representation (handle empty safely)
            seq_reprs = []
            for i in range(B):
                ri = residue_reprs[i]
                if ri.numel() == 0:
                    # No residues (edge case) -> zero vector
                    seq_reprs.append(torch.zeros(H, device=ri.device, dtype=ri.dtype))
                else:
                    seq_reprs.append(ri.mean(dim=0))
            seq_reprs = torch.stack(seq_reprs, dim=0)  # (B, H)
            seq_reprs = self.out(seq_reprs)  # (B, out_dim)
            mask_out = None
            padded_out = seq_reprs
        else:
            # Pad variable-length residue sequences to a uniform length
            padded = pad_sequence(residue_reprs, batch_first=True)  # (B, L_max_res, H)
            # Build mask over residues (True for valid residues)
            Lmax = padded.size(1)
            # mask[i, j] = j < residue_lens[i]
            idxs = torch.arange(Lmax, device=padded.device).unsqueeze(0).expand(B, Lmax)
            mask = idxs < residue_lens.unsqueeze(1)

            # Optional: L2-normalize token embeddings, then zero-out pads via mask
            padded = normalize(padded, dim=-1)
            padded = padded * mask.unsqueeze(-1)

            padded_out = self.out(padded)  # (B, L_max_res, out_dim)
            mask_out = mask

        # Optionally compute logits over vocab for every token (B, L_full, V)
        if get_mask_logits:
            mask_logits = self.model.lm_head(last_hidden_state)
        else:
            mask_logits = None

        # Return (representations, mask, logits) where mask is None for seq-level mode
        return padded_out, mask_out, mask_logits


if __name__ == "__main__":
    # Example usage
    enc = ProteinEncoder(
        config_path="facebook/esm2_t33_650M_UR50D",
        out_dim=256,
        load_pretrained=True,
        gradient_checkpointing=False,
        seq_level_reprs=False,
    )
    seqs = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
        "K A <mask> I S Q",
    ]
    reps, mask, logits = enc(seqs, get_mask_logits=True)
    print("Representations:", reps.shape)
    print("Mask:", mask.shape if mask is not None else None)
    if logits is not None:
        print("Mask logits:", logits.shape)
