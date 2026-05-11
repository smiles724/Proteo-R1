# Copyright 2024 LMMs-Lab
# ProteoR1Understand with Liger Fused Linear Cross Entropy
#
# Strategy: reuse the loss computation in qwen3_lce_forward.
# - ProteoR1Understand only prepares inputs_embeds (merging protein embedding)
# - self.llm (patched to qwen3_lce_forward) handles Fused Linear CE
#
# Benefits:
# - Reuses Qwen3's validated loss computation
# - Inherits rmpad support and sequence parallelism
# - Minimal, easy-to-maintain code

from typing import Any, Optional, Tuple, Union

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast


def protenix_qwen_lce_forward(
    self,  # ProteoR1UnderstandModel
    input_ids: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    labels: torch.Tensor = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int = 0,
    # Protenix inputs - on-the-fly mode
    protenix_input_feature_dict: Optional[dict] = None,
    protenix_atom_array: Optional[Any] = None,
    protenix_token_array: Optional[Any] = None,
    # Protenix inputs - precomputed mode
    protenix_s_embedding: Optional[torch.Tensor] = None,
    protenix_esm_embedding: Optional[torch.Tensor] = None,
    protenix_a_token: Optional[torch.Tensor] = None,
    protenix_embedding_attention_mask: Optional[torch.Tensor] = None,
    # Protenix position info (used by protenix_pos_embed)
    protenix_residue_index: Optional[torch.Tensor] = None,
    protenix_asym_id: Optional[torch.Tensor] = None,
    # rmpad parameter (injected by the monkey_patch wrapper)
    use_rmpad: bool = False,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
    """
    ProteoR1Understand forward with Liger Fused Linear Cross Entropy.

    Strategy:
    1. Use _prepare_inputs_embeds to build inputs_embeds with the protein embedding merged in.
    2. Call self.llm (patched to qwen3_lce_forward) to compute the Fused Linear CE loss.

    This fully reuses Qwen3's validated loss computation and inherits rmpad and
    sequence parallelism support for free.

    Args:
        input_ids: [B, L] text token ids
        attention_mask: [B, L]
        labels: [B, L] for computing loss
        protenix_input_feature_dict: Protenix feature dict (on-the-fly mode)
        protenix_atom_array: biotite AtomArray (on-the-fly mode, optional)
        protenix_token_array: TokenArray (on-the-fly mode, optional)
        protenix_s_embedding: [B, N_max, 384] precomputed s embedding (precomputed mode)
        protenix_esm_embedding: [B, N_max, 2560] precomputed ESM embedding (precomputed mode)
        protenix_embedding_attention_mask: [B, N_max] embedding attention mask (precomputed mode)

    Returns:
        CausalLMOutputWithPast
    """
    # Step 1: Prepare inputs_embeds (merging the protein embedding).
    inputs_embeds = self._prepare_inputs_embeds(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        past_key_values=past_key_values,
        protenix_input_feature_dict=protenix_input_feature_dict,
        protenix_atom_array=protenix_atom_array,
        protenix_token_array=protenix_token_array,
        protenix_s_embedding=protenix_s_embedding,
        protenix_esm_embedding=protenix_esm_embedding,
        protenix_a_token=protenix_a_token,
        protenix_embedding_attention_mask=protenix_embedding_attention_mask,
        protenix_residue_index=protenix_residue_index,
        protenix_asym_id=protenix_asym_id,
    )

    # Step 2: Call self.llm (patched to qwen3_lce_forward).
    # qwen3_lce_forward handles:
    # - Fused Linear CE loss computation (no materialized logits)
    # - rmpad support (shift in segments according to seq_lens)
    # - Sequence parallelism (Ulysses SP gather)
    return self.llm(
        input_ids=None,  # use inputs_embeds, do not pass input_ids
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        labels=labels,
        use_cache=use_cache,
        cache_position=cache_position,
        num_logits_to_keep=logits_to_keep,  # qwen3_lce_forward expects num_logits_to_keep
        **kwargs,
    )
