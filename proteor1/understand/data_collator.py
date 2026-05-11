"""
ProteoR1Understand Data Collator

Supports two modes:
1. On-the-fly mode (use_precomputed=False): Protenix O(N^2) memory limits this to batch_size=1.
2. Precomputed mode (use_precomputed=True): supports batch_size > 1 by padding embeddings.

Usage:
    from proteor1.understand import ProteoR1UnderstandDataCollator

    # On-the-fly mode (batch_size=1)
    collator = ProteoR1UnderstandDataCollator(
        tokenizer=processor.tokenizer,
        use_precomputed=False,  # default
    )
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collator)

    # Precomputed mode (supports batch_size > 1)
    collator = ProteoR1UnderstandDataCollator(
        tokenizer=processor.tokenizer,
        use_precomputed=True,
    )
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator)
"""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase


@dataclass
class ProteoR1UnderstandDataCollator(DataCollatorForSeq2Seq):
    """
    ProteoR1Understand Data Collator.

    Supports two modes:
    1. On-the-fly mode (use_precomputed=False): batch_size=1, passes protenix data through unchanged.
    2. Precomputed mode (use_precomputed=True): batch_size > 1, pads embeddings.

    Main responsibilities:
    1. Pad the text side (input_ids, attention_mask, labels).
    2. On-the-fly mode: pass protenix-related data through (no padding needed).
    3. Precomputed mode: pad protenix_s_embedding, protenix_esm_embedding, protenix_a_token,
       and return protenix_embedding_attention_mask so the model can recover valid positions.
    4. Optionally return metadata.
    """
    tokenizer: PreTrainedTokenizerBase = None
    return_metadata: bool = False
    use_precomputed: bool = False  # whether to use precomputed mode
    protein_token_id: Optional[int] = None  # used to verify token-count consistency
    return_chain_type_ids: bool = False  # whether to return chain_type_ids and cdr_region_type_ids

    def __call__(self, features, return_tensors=None):
        """
        Collate features into a batch.

        Args:
            features: List of feature dicts from dataset
                On-the-fly mode:
                - input_ids: [L] text token ids
                - attention_mask: [L]
                - labels: [L] (optional)
                - protenix_input_feature_dict: dict (Protenix features)
                - protenix_atom_array: AtomArray (optional)
                - protenix_token_array: TokenArray (optional)

                Precomputed mode:
                - input_ids: [L] text token ids
                - attention_mask: [L]
                - labels: [L] (optional)
                - protenix_s_embedding: [N_token, 384]
                - protenix_esm_embedding: [N_token, 2560]
                - protenix_a_token: [N_token, 768]

        Returns:
            batch dict with:
                On-the-fly mode (batch_size=1):
                - input_ids: [1, L]
                - attention_mask: [1, L]
                - labels: [1, L]
                - protenix_input_feature_dict: dict
                - protenix_atom_array: AtomArray
                - protenix_token_array: TokenArray

                Precomputed mode (batch_size >= 1):
                - input_ids: [B, L] padded
                - attention_mask: [B, L]
                - labels: [B, L]
                - protenix_s_embedding: [B, N_max, 384] padded
                - protenix_esm_embedding: [B, N_max, 2560] padded
                - protenix_a_token: [B, N_max, 768] padded
                - protenix_embedding_attention_mask: [B, N_max] (1=valid, 0=padding)
        """
        if self.use_precomputed:
            return self._collate_precomputed(features, return_tensors)
        else:
            return self._collate_realtime(features, return_tensors)

    def _collate_realtime(self, features, return_tensors=None):
        """On-the-fly mode: batch_size=1."""
        # Verify batch_size=1.
        if len(features) != 1:
            raise ValueError(
                f"ProteoR1UnderstandDataCollator (realtime mode) only supports batch_size=1 "
                f"due to Protenix O(N²) memory. Got batch_size={len(features)}. "
                f"Use DataLoader(batch_size=1) or set use_precomputed=True for batching."
            )

        # Split text features from protenix features.
        text_features = []
        protenix_data = {}
        position_ids = None
        metadata_list = []

        for feat in features:
            text_feat = {}
            for key, value in feat.items():
                if key in ["input_ids", "attention_mask"]:
                    text_feat[key] = value
                elif key == "labels":
                    # Convert np.ndarray/torch.Tensor to list so the parent DataCollatorForSeq2Seq
                    # takes the list branch instead of np.concatenate; avoids the slow
                    # torch.tensor(List[np.ndarray]) conversion warning.
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        value = value.tolist()
                    text_feat[key] = value
                elif key == "position_ids":
                    # position_ids needs special handling (adds a batch dim later).
                    position_ids = value
                elif key.startswith("protenix_"):
                    protenix_data[key] = value
                elif key == "metadata":
                    metadata_list.append(value)
            text_features.append(text_feat)

        # Let the parent pad the text side.
        batch = super(ProteoR1UnderstandDataCollator, self).__call__(
            features=text_features,
            return_tensors=return_tensors
        )

        # Append protenix data (batch_size=1, use the first sample directly).
        batch.update(protenix_data)

        # Append position_ids (batch_size=1, just unsqueeze).
        if position_ids is not None:
            if isinstance(position_ids, torch.Tensor):
                batch["position_ids"] = position_ids.unsqueeze(0)
            else:
                batch["position_ids"] = torch.tensor(position_ids).unsqueeze(0)

        # Optionally return metadata.
        if self.return_metadata and metadata_list:
            batch["metadata"] = metadata_list

        return batch

    def _collate_precomputed(self, features, return_tensors=None):
        """Precomputed mode: supports batch_size > 1, pads embeddings and position_ids."""
        # Split text features from the rest.
        text_features = []
        s_embeddings = []
        esm_embeddings = []
        a_token_embeddings = []
        position_ids_list = []
        residue_index_list = []
        asym_id_list = []
        cdr_mask_list = []  # CDR mask list
        chain_type_ids_list: list[torch.Tensor] = []  # chain type ids list
        cdr_region_type_ids_list: list[torch.Tensor] = []  # CDR region type ids list
        metadata_list = []

        for feat in features:
            text_feat = {}
            for key, value in feat.items():
                if key in ["input_ids", "attention_mask"]:
                    text_feat[key] = value
                elif key == "labels":
                    # Convert np.ndarray/torch.Tensor to list so the parent DataCollatorForSeq2Seq
                    # takes the list branch instead of np.concatenate; avoids the slow
                    # torch.tensor(List[np.ndarray]) conversion warning.
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        value = value.tolist()
                    text_feat[key] = value
                elif key == "protenix_s_embedding":
                    s_embeddings.append(value)
                elif key == "protenix_esm_embedding":
                    esm_embeddings.append(value)
                elif key == "protenix_a_token":
                    a_token_embeddings.append(value)
                elif key == "position_ids":
                    position_ids_list.append(
                        value if isinstance(value, torch.Tensor) else torch.tensor(value)
                    )
                elif key == "protenix_residue_index":
                    residue_index_list.append(
                        value if isinstance(value, torch.Tensor) else torch.tensor(value)
                    )
                elif key == "protenix_asym_id":
                    asym_id_list.append(
                        value if isinstance(value, torch.Tensor) else torch.tensor(value)
                    )
                elif key == "protenix_cdr_mask":
                    cdr_mask_list.append(
                        value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.bool)
                    )
                elif key == "chain_type_ids" and self.return_chain_type_ids:
                    chain_type_ids_list.append(
                        value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
                    )
                elif key == "cdr_region_type_ids" and self.return_chain_type_ids:
                    cdr_region_type_ids_list.append(
                        value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
                    )
                elif key == "metadata":
                    metadata_list.append(value)
            text_features.append(text_feat)

        # Verify each sample's protein-token count matches the embedding length.
        if self.protein_token_id is not None and s_embeddings:
            for i, (text_feat, s_emb) in enumerate(zip(text_features, s_embeddings)):
                input_ids = text_feat["input_ids"]
                if isinstance(input_ids, torch.Tensor):
                    protein_count = (input_ids == self.protein_token_id).sum().item()
                else:
                    protein_count = sum(1 for t in input_ids if t == self.protein_token_id)
                emb_len = s_emb.shape[0]
                if protein_count != emb_len:
                    metadata_info = metadata_list[i] if i < len(metadata_list) else {}
                    raise ValueError(
                        f"Sample {i} protein token mismatch in collator: "
                        f"input_ids has {protein_count} protein tokens, "
                        f"but embedding has {emb_len} tokens. "
                        f"metadata={metadata_info}"
                    )

        # Let the parent pad the text side.
        batch = super(ProteoR1UnderstandDataCollator, self).__call__(
            features=text_features,
            return_tensors=return_tensors
        )

        # Pad embeddings (right padding) and produce the matching attention_mask.
        if s_embeddings:
            padded_s, embedding_mask = self._pad_embeddings(s_embeddings)
            batch["protenix_s_embedding"] = padded_s
            # attention_mask: 1 marks valid positions, 0 marks padding.
            batch["protenix_embedding_attention_mask"] = embedding_mask

        if esm_embeddings:
            padded_esm, _ = self._pad_embeddings(esm_embeddings)
            batch["protenix_esm_embedding"] = padded_esm

        if a_token_embeddings:
            padded_a_token, _ = self._pad_embeddings(a_token_embeddings)
            batch["protenix_a_token"] = padded_a_token

        # position_ids: follow tokenizer.padding_side (default).
        if position_ids_list:
            batch["position_ids"] = self._pad_1d_tensors(position_ids_list)

        # protenix_residue_index / protenix_asym_id: always right-padded (matches the embedding).
        if residue_index_list:
            batch["protenix_residue_index"] = self._pad_1d_tensors(
                residue_index_list, padding_side="right"
            )
        if asym_id_list:
            batch["protenix_asym_id"] = self._pad_1d_tensors(
                asym_id_list, padding_side="right"
            )

        # protenix_cdr_mask: always right-padded (matches the embedding).
        # Handle mixed batches where some samples have a CDR mask and others do not.
        # Check whether any sample carries cdr_mask.
        has_any_cdr_mask = any("protenix_cdr_mask" in f for f in features)
        if has_any_cdr_mask:
            # Rebuild cdr_mask_list, synthesizing an all-zero mask for samples without one.
            cdr_mask_list = []
            for feat in features:
                if "protenix_cdr_mask" in feat:
                    value = feat["protenix_cdr_mask"]
                    cdr_mask_list.append(
                        value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.bool)
                    )
                else:
                    # For samples without cdr_mask, synthesize an all-zero mask
                    # whose length matches that sample's s_embedding.
                    n_token = feat["protenix_s_embedding"].shape[0]
                    cdr_mask_list.append(torch.zeros(n_token, dtype=torch.bool))

            batch["protenix_cdr_mask"] = self._pad_1d_tensors(
                cdr_mask_list, pad_value=0, padding_side="right"
            ).bool()

        # chain_type_ids and cdr_region_type_ids: only processed when return_chain_type_ids=True.
        # Follow tokenizer.padding_side and use -1 as the padding value.
        if self.return_chain_type_ids and chain_type_ids_list:
            batch["chain_type_ids"] = self._pad_1d_tensors(
                chain_type_ids_list, pad_value=-1
            )
        if self.return_chain_type_ids and cdr_region_type_ids_list:
            batch["cdr_region_type_ids"] = self._pad_1d_tensors(
                cdr_region_type_ids_list, pad_value=-1
            )

        # Optionally return metadata.
        if self.return_metadata and metadata_list:
            batch["metadata"] = metadata_list

        return batch

    def _pad_1d_tensors(
        self,
        tensor_list: list[torch.Tensor],
        pad_value: int = 0,
        padding_side: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Pad a list of 1D tensors.

        Args:
            tensor_list: list of [L_i] tensors
            pad_value: padding value, default 0
            padding_side: padding direction, default None (follows tokenizer.padding_side)

        Returns:
            padded: [B, L_max] padded tensor
        """
        from torch.nn.utils.rnn import pad_sequence

        if padding_side is None:
            padding_side = getattr(self.tokenizer, "padding_side", "right")

        if padding_side == "left":
            # Left padding: flip first, pad, then flip back.
            tensors = [t.flip(0) for t in tensor_list]
            padded = pad_sequence(tensors, batch_first=True, padding_value=pad_value)
            return padded.flip(1)
        else:
            # Right padding.
            return pad_sequence(tensor_list, batch_first=True, padding_value=pad_value)

    def _pad_embeddings(
        self,
        embeddings: list[torch.Tensor],
        pad_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a list of variable-length embeddings.

        Args:
            embeddings: list of [N_i, D] tensors
            pad_value: padding value

        Returns:
            padded: [B, N_max, D] padded embeddings
            attention_mask: [B, N_max], 1 marks valid positions and 0 marks padding
        """
        batch_size = len(embeddings)
        max_len = max(emb.shape[0] for emb in embeddings)
        hidden_dim = embeddings[0].shape[1]
        dtype = embeddings[0].dtype

        # Allocate the padded tensor.
        padded = torch.full(
            (batch_size, max_len, hidden_dim),
            pad_value,
            dtype=dtype,
        )

        # Allocate the attention mask (1=valid, 0=padding).
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        # Fill in.
        for i, emb in enumerate(embeddings):
            length = emb.shape[0]
            padded[i, :length, :] = emb
            attention_mask[i, :length] = 1

        return padded, attention_mask


def _move_value_to_device(
    value: Any,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Any:
    """Recursively move value to the given device."""
    if value is None:
        return None
    elif isinstance(value, torch.Tensor):
        moved_value = value.to(device)
        if dtype is not None and moved_value.is_floating_point():
            moved_value = moved_value.to(dtype)
        return moved_value
    elif isinstance(value, dict):
        return {k: _move_value_to_device(v, device, dtype) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        moved = [_move_value_to_device(v, device, dtype) for v in value]
        return type(value)(moved)
    else:
        # int, str, numpy array, etc. are left untouched.
        return value


def move_protenix_features_to_device(
    input_feature_dict: dict[str, Any],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, Any]:
    """
    Move tensors inside protenix_input_feature_dict to the given device.

    Supports nested dict/list structures and recurses through every tensor.

    Args:
        input_feature_dict: Protenix feature dict
        device: target device
        dtype: optional target dtype (only applies to float tensors)

    Returns:
        Moved feature dict.
    """
    # These keys are not tensors; skip them.
    non_tensor_keys = {
        "esm_sequence_lengths",
        "esm_unique_sequences",
        "esm_entity_id_to_sequence",
        "esm_embedding_dim",
    }

    moved_dict = {}
    for key, value in input_feature_dict.items():
        if key in non_tensor_keys:
            moved_dict[key] = value
        else:
            moved_dict[key] = _move_value_to_device(value, device, dtype)

    return moved_dict


def prepare_batch_for_model(
    batch: dict[str, Any],
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, Any]:
    """
    Prepare a batch for the model's forward pass.

    Moves every required tensor to device and handles protenix-specific data.

    Supports two modes:
    1. On-the-fly mode: protenix_input_feature_dict, protenix_atom_array, protenix_token_array.
    2. Precomputed mode: protenix_s_embedding, protenix_esm_embedding, protenix_a_token.

    Note: float tensors inside protenix_input_feature_dict are cast to the target dtype
    because the model uses a single dtype (upstream Protenix supports bf16 inference).

    Args:
        batch: batch dict returned by the collator
        device: target device
        dtype: optional target dtype (applied to float tensors)

    Returns:
        Prepared batch dict.
    """
    prepared = {}

    for key, value in batch.items():
        if key in ["input_ids", "attention_mask", "labels", "position_ids"]:
            prepared[key] = value.to(device)
        elif key == "protenix_input_feature_dict" and value is not None:
            # On-the-fly mode: protenix features also use the unified dtype (upstream Protenix supports bf16).
            prepared[key] = move_protenix_features_to_device(value, device, dtype=dtype)
        elif key in [
            "protenix_s_embedding", "protenix_esm_embedding", "protenix_a_token",
            "protenix_embedding_attention_mask", "protenix_residue_index", "protenix_asym_id",
            "protenix_cdr_mask", "chain_type_ids", "cdr_region_type_ids",
        ]:
            # Precomputed mode: move to device.
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(device)
            else:
                prepared[key] = value
        elif key in ["protenix_atom_array", "protenix_token_array", "protenix_metadata"]:
            # These do not need to move to the GPU.
            prepared[key] = value
        elif key == "metadata":
            # metadata is a list[dict]; no need to move.
            prepared[key] = value
        else:
            # Other tensors.
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(device)
            else:
                prepared[key] = value

    return prepared
