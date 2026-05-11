# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ProtenixEncoder - end-to-end protein sequence encoder.

Has two parts:
1. ESM Encoder: ESM2-3B-ISM, produces token embedding (2560-dim) from amino-acid sequences.
2. Protenix Encoder: Pairformer, turns token embedding into structure-aware s embedding (384-dim).

Input:  the input_feature_dict produced by ProtenixProcessor.
Output: Pairformer s embedding (384-dim) and ESM embedding (2560-dim).

Usage:
    from temp.protenix_processor import ProtenixProcessor, protenix_collate_fn
    from temp.protenix_encoder import ProtenixEncoder

    # Load the Processor and the Encoder.
    processor = ProtenixProcessor.from_pretrained("pretrained/protenix_encoder")
    encoder = ProtenixEncoder.from_pretrained("pretrained/protenix_encoder")
    encoder.eval()

    # Process a JSON entry.
    processor_output = processor(json_entry)
    batch = protenix_collate_fn([processor_output])

    # Encoder forward.
    with torch.no_grad():
        s, esm_embedding = encoder(
            input_feature_dict=batch["input_feature_dict"],
            atom_array=batch["atom_array"],
            token_array=batch["token_array"],
        )

    # s: [N_token, 384] - Pairformer output
    # esm_embedding: [N_token, 2560] - ESM embedding

Pretrained directory layout:
    pretrained_path/
    ├── config.json                      # Protenix config file
    ├── protenix_mini_ism_v0.5.0.pt     # Protenix model weights
    └── esm2_t36_3B_UR50D_ism.pt        # ESM model weights
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from protenix.openfold_local.model.primitives import LayerNorm
from protenix.model.generator import InferenceNoiseScheduler
from protenix.model.modules.embedders import (
    ConstraintEmbedder,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from protenix.model.modules.pairformer import MSAModule, PairformerStack, TemplateEmbedder
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.protenix import update_input_feature_dict


# ============================================================================
# Utility Functions
# ============================================================================

def _convert_tensors_to_fp32(obj):
    """
    Recursively convert all floating point tensors to fp32.
    Matches the behavior of official autocasting_disable_decorator's conditioned_cast.

    Reference: protenix/utils/torch_utils.py:199-226 (conditioned_cast)

    Args:
        obj: Can be torch.Tensor, dict, list, tuple, or other types

    Returns:
        Converted object (floating point tensors converted to fp32, others unchanged)
    """
    if isinstance(obj, torch.Tensor) and torch.is_floating_point(obj):
        return obj.to(dtype=torch.float32)
    elif isinstance(obj, dict):
        return {k: _convert_tensors_to_fp32(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [_convert_tensors_to_fp32(v) for v in obj]
        return type(obj)(converted)
    return obj


# ============================================================================
# PairformerOutput Dataclass
# ============================================================================

@dataclass
class PairformerOutput:
    """
    Pairformer full output containing all inputs needed for Diffusion module.

    This dataclass provides complete access to intermediate features for:
    1. Training: Computing diffusion loss with GT coordinates
    2. Inference: Running DiffusionModule for structure prediction
    3. Feature extraction: Dumping features for downstream tasks

    Attributes:
        s_inputs: InputEmbedder output [N_token, 449]
        s_trunk: Pairformer single features [N_token, 384]
        z_trunk: Pairformer pair features [N_token, N_token, 128]
        esm_embedding: ESM token embedding [N_token, 2560]
        input_feature_dict: Processed input features with relp, d_lm, v_lm, pad_info
        n_token: Number of tokens
        n_atom: Number of atoms
        pair_z: Optional pair features for diffusion conditioning
        p_lm: Optional predicted local frame (from recycling)
        c_l: Optional predicted confidence logits
    """
    s_inputs: torch.Tensor          # [N_token, 449] InputEmbedder output
    s_trunk: torch.Tensor           # [N_token, 384] Pairformer single features
    z_trunk: torch.Tensor           # [N_token, N_token, 128] Pairformer pair features
    esm_embedding: torch.Tensor     # [N_token, 2560] ESM embedding
    input_feature_dict: Dict[str, Any]  # Processed input features (with relp, d_lm, v_lm, pad_info)
    n_token: int
    n_atom: int
    pair_z: Optional[torch.Tensor] = None  # Optional pair features for diffusion
    p_lm: Optional[torch.Tensor] = None    # Optional predicted local frame
    c_l: Optional[torch.Tensor] = None     # Optional confidence logits


# ============================================================================
# ESM Encoder Wrapper
# ============================================================================

class ESMEncoder(nn.Module):
    """
    ESM2-3B-ISM Encoder wrapper.

    Produces token-level embedding (2560-dim) from amino-acid sequences.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 2560,
        truncation_seq_length: int = 4094,  # matches protenix/data/compute_esm.py:181
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.truncation_seq_length = truncation_seq_length
        self.model = None
        self.alphabet = None
        self.batch_converter = None

        if model_path is not None:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load the ESM2 model (Encoder only, for feature extraction).

        ESM2 is an encoder-only model; there is no decoder.
        contact-regression.pt only holds the contact-prediction head weights,
        which is irrelevant to embedding extraction, so it is not loaded.

        The loading logic mirrors esm/pretrained.py:164-183 (_load_model_and_alphabet_core_v2).

        References:
        - protenix/data/compute_esm.py:39-66
        - esm/pretrained.py:164-183
        """
        import re
        import esm
        from esm.model.esm2 import ESM2
        from argparse import Namespace

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ESM model not found: {model_path}")

        print(f"Loading ESM model from {model_path}")

        # Register Namespace as a safe global (required by PyTorch 2.6+).
        # See runner/inference.py:61.
        torch.serialization.add_safe_globals([Namespace])

        # Load model data.
        # weights_only=False is required because the ESM checkpoint contains an argparse.Namespace.
        model_data = torch.load(model_path, map_location="cpu", weights_only=False)

        # Reuse the upgrade_state_dict from esm/pretrained.py:165-170.
        def upgrade_state_dict(state_dict):
            """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
            prefixes = ["encoder.sentence_encoder.", "encoder."]
            pattern = re.compile("^" + "|".join(prefixes))
            state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
            return state_dict

        # esm/pretrained.py:172-183 (_load_model_and_alphabet_core_v2)
        cfg = model_data["cfg"]["model"]
        state_dict = model_data["model"]
        state_dict = upgrade_state_dict(state_dict)

        self.alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
        self.model = ESM2(
            num_layers=cfg.encoder_layers,
            embed_dim=cfg.encoder_embed_dim,
            attention_heads=cfg.encoder_attention_heads,
            alphabet=self.alphabet,
            token_dropout=cfg.token_dropout,
        )

        # Load weights (contact_head is not needed).
        load_result = self.model.load_state_dict(state_dict, strict=False)

        # Verify the only missing key is contact_head.regression (used by contact prediction, which we skip).
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        if set(load_result.missing_keys) != expected_missing:
            raise RuntimeError(
                f"Unexpected missing keys in ESM state_dict: {set(load_result.missing_keys) - expected_missing}"
            )
        if load_result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys in ESM state_dict: {load_result.unexpected_keys}"
            )

        self.batch_converter = self.alphabet.get_batch_converter(self.truncation_seq_length)

        print(f"Loaded ESM model with {sum(p.numel() for p in self.model.parameters()):,} parameters")

    def to(self, *args, **kwargs):
        """Move the model to a specific device/dtype."""
        super().to(*args, **kwargs)
        if self.model is not None:
            self.model = self.model.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        sequence_lengths: list[int],
    ) -> list[torch.Tensor]:
        """
        Compute ESM embeddings.

        Matches the upstream compute_esm.py; returns one embedding per sequence.

        Args:
            input_ids: ESM token IDs [n_sequences, seq_len+2], including BOS/EOS
            sequence_lengths: actual length of each chain (excluding BOS/EOS)

        Returns:
            list[Tensor]: per-chain embedding, shape [seq_len, 2560]
        """
        if self.model is None:
            raise RuntimeError("ESM model not loaded. Call load_model() first.")

        device = next(self.model.parameters()).device
        batch_tokens = input_ids.to(device)

        # Forward pass.
        repr_layer = self.model.num_layers
        results = self.model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
        representations = results["representations"][repr_layer]

        # Extract the per-chain embedding (drop BOS).
        # Matches upstream: embeddings[label] = representation[i, 1:truncate_len+1].clone()
        chain_embeddings = []
        n_sequences = representations.shape[0]
        for i, length in enumerate(sequence_lengths):
            if i >= n_sequences:
                raise IndexError(
                    f"sequence_lengths has {len(sequence_lengths)} entries, "
                    f"but ESM output only has {n_sequences} sequences. "
                    f"Index {i} is out of bounds."
                )
            # ESM output shape: [n_sequences, seq_len+2, dim]; +2 accounts for BOS and EOS.
            seq_len_with_special = representations.shape[1]
            if length + 1 > seq_len_with_special:
                raise ValueError(
                    f"Sequence {i}: length={length}, but ESM output seq_len={seq_len_with_special - 2} "
                    f"(with BOS/EOS: {seq_len_with_special}). "
                    f"Cannot extract positions 1:{length + 1}."
                )
            emb = representations[i, 1:length + 1]  # drop BOS, keep `length` tokens
            chain_embeddings.append(emb)

        return chain_embeddings


# ============================================================================
# ProtenixEncoder - end-to-end protein encoder
# ============================================================================

class ProtenixEncoder(nn.Module):
    """
    End-to-end protein encoder: ESM + Pairformer + Diffusion (optional).

    Input: multi-chain amino-acid sequences (text) or a precomputed ESM embedding.
    Output: Pairformer s embedding (384-dim) + ESM embedding (2560-dim).

    Optional: pass load_diffusion=True to load the DiffusionModule for a_token (768-dim) extraction.
    """

    def __init__(
        self,
        configs,
        load_esm: bool = True,
        load_diffusion: bool = False,
        esm_model_path: Optional[str] = None,
        triangle_by_torch: Optional[bool] = None
    ) -> None:
        """
        Initialize the ProtenixEncoder.

        Args:
            configs: Protenix configuration
            load_esm: whether to load the ESM encoder; default True
            load_diffusion: whether to load the DiffusionModule; default False
                When True you can call forward_diffusion() to extract a_token features.
            esm_model_path: path to the ESM model; None means lazy loading
            triangle_by_torch: whether to use the PyTorch implementation of triangle ops.
                - None: use the value from the config file
                - True: force PyTorch. Useful for: 1) debugging, 2) older GPUs (e.g. V100)
                  that lack Triton, 3) environments where cuequivariance Triton kernels fail (e.g. WSL2)
                - False: force Triton kernels (cuequivariance)
        """
        super(ProtenixEncoder, self).__init__()
        self.configs = configs
        if triangle_by_torch is not None:
            self.configs.triangle_attention = "torch" if triangle_by_torch else "triattention"
            self.configs.triangle_multiplicative = "torch" if triangle_by_torch else "cuequivariance"

        torch.backends.cuda.matmul.allow_tf32 = self.configs.enable_tf32
        self.N_cycle = self.configs.model.N_cycle

        # ESM Encoder
        self.load_esm = load_esm
        self.esm_encoder = None
        if load_esm:
            esm_configs = configs.get("esm", {})
            self.esm_embedding_dim = esm_configs.get("embedding_dim", 2560)
            self.esm_encoder = ESMEncoder(
                model_path=esm_model_path,
                embedding_dim=self.esm_embedding_dim,
            )

        # Diffusion Module (optional, for a_token extraction)
        self.load_diffusion = load_diffusion
        self.diffusion_module = None
        if load_diffusion:
            from protenix.model.modules.diffusion import DiffusionModule
            self.diffusion_module = DiffusionModule(**configs.model.diffusion_module)

        # Diffusion noise schedule parameters (for forward_diffusion)
        # Initialize InferenceNoiseScheduler following official Protenix (protenix.py:102-103)
        self.noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )

        # Compute c_tau_last: the second-to-last value in noise schedule
        # This is the noise level at the final denoising step (before t_N=0)
        # Reference: generator.py:183-185 - for loop uses noise_schedule[:-1] and noise_schedule[1:]
        # so the last c_tau_last used is noise_schedule[-2]
        sample_diffusion_config = configs.sample_diffusion
        N_step = sample_diffusion_config.N_step
        noise_schedule = self.noise_scheduler(N_step=N_step)
        self.c_tau_last = float(noise_schedule[-2])  # Second-to-last value (last is 0)

        # Diffusion optimization settings (from config)
        # These defaults match official Protenix behavior:
        # - skip_amp.sample_diffusion=True: disable AMP for numerical stability
        # - enable_efficient_fusion=True: enable efficient fusion optimization
        # - enable_diffusion_shared_vars_cache=True: enable caching for shared variables
        skip_amp = configs.get("skip_amp", {})
        self.default_disable_amp = skip_amp.get("sample_diffusion", True)
        self.default_enable_efficient_fusion = configs.get("enable_efficient_fusion", True)
        self.default_use_cache = configs.get("enable_diffusion_shared_vars_cache", True)

        # Inference settings for attention chunking
        # This matches official Protenix behavior (protenix.py:304-307)
        infer_setting = configs.get("infer_setting", {})
        self.attn_chunk_size = infer_setting.get("chunk_size", None)

        # Protenix Encoder (copied from protenix/model/protenix.py)
        esm_configs = configs.get("esm", {})
        self.input_embedder = InputFeatureEmbedder(
            **configs.model.input_embedder, esm_configs=esm_configs
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        self.template_embedder = TemplateEmbedder(**configs.model.template_embedder)
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.constraint_embedder = ConstraintEmbedder(
            **configs.model.constraint_embedder
        )
        self.pairformer_stack = PairformerStack(**configs.model.pairformer)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def load_esm_model(self, esm_model_path: str):
        """Lazily load the ESM model."""
        if self.esm_encoder is None:
            self.esm_encoder = ESMEncoder(
                model_path=esm_model_path,
                embedding_dim=self.configs.get("esm", {}).get("embedding_dim", 2560),
            )
        else:
            self.esm_encoder.load_model(esm_model_path)

    def to(self, *args, **kwargs):
        """Move the model to a specific device/dtype."""
        super().to(*args, **kwargs)
        if self.esm_encoder is not None:
            self.esm_encoder = self.esm_encoder.to(*args, **kwargs)
        return self

    @property
    def device(self) -> torch.device:
        """Return the device the model lives on."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the model's dtype."""
        return next(self.parameters()).dtype

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[torch.Tensor, ...]:
        """
        Pairformer forward.

        Args:
            input_feature_dict: input feature dict
            N_cycle: number of cycles
            inplace_safe: whether to use in-place ops
            chunk_size: chunk size

        Returns:
            (s_inputs, s, z): input embedding, Pairformer output, pair embedding
        """
        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )
        z_constraint = None

        if "constraint_feature" in input_feature_dict:
            z_constraint = self.constraint_embedder(
                input_feature_dict["constraint_feature"]
            )

        s_init = self.linear_no_bias_sinit(s_inputs)
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict["relp"])
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init += z_constraint
        else:
            z_init = z_init + self.relative_position_encoding(
                input_feature_dict["relp"]
            )
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init = z_init + z_constraint

        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training and cycle_no == (N_cycle - 1)
            ):
                z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        return s_inputs, s, z

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        atom_array: Optional[Any] = None,
        token_array: Optional[Any] = None,
        return_full_output: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor], PairformerOutput]:
        """
        Forward pass (consumes ProtenixProcessor output).

        input_feature_dict should contain:
        - Protenix features (token_index, asym_id, ref_pos, ...)
        - ESM tokenization (esm_input_ids, esm_attention_mask, esm_sequence_lengths, ...)
        - Or a precomputed esm_token_embedding

        Args:
            input_feature_dict: full feature dict produced by ProtenixProcessor
            atom_array: Biotite AtomArray (used to fill the ESM embedding; optional)
            token_array: TokenArray (used to fill the ESM embedding; optional)
            return_full_output: If True, return PairformerOutput with all intermediate
                features needed for diffusion. Default False for backward compatibility.

        Returns:
            If return_full_output=False (default):
                (s, esm_embedding): Pairformer output [N_token, 384], ESM embedding [N_token, 2560]
            If return_full_output=True:
                PairformerOutput: Complete output with s_inputs, s_trunk, z_trunk,
                    esm_embedding, and processed input_feature_dict
        """
        # =================================================================
        # Step 1: compute or fetch the ESM embedding
        # =================================================================
        esm_embedding = input_feature_dict.get("esm_token_embedding", None)

        if esm_embedding is None:
            # Pull ESM inputs out of input_feature_dict.
            esm_input_ids = input_feature_dict.get("esm_input_ids", None)

            if esm_input_ids is not None and esm_input_ids.numel() > 0:
                # Need to compute the ESM embedding.
                if self.esm_encoder is None or self.esm_encoder.model is None:
                    raise RuntimeError(
                        "ESM encoder not loaded. Either:\n"
                        "1. Use ProtenixEncoder.from_pretrained(..., load_esm=True), or\n"
                        "2. Pre-compute esm_token_embedding in input_feature_dict"
                    )

                esm_sequence_lengths = input_feature_dict.get("esm_sequence_lengths", [])
                esm_unique_sequences = input_feature_dict.get("esm_unique_sequences", [])
                esm_entity_id_to_sequence = input_feature_dict.get("esm_entity_id_to_sequence", {})

                # Compute the ESM embedding for each unique sequence.
                unique_embeddings = self.esm_encoder(
                    input_ids=esm_input_ids,
                    sequence_lengths=esm_sequence_lengths,
                )

                # Build the sequence -> embedding mapping.
                seq_to_embedding = dict(zip(esm_unique_sequences, unique_embeddings))

                # Fill esm_token_embedding.
                esm_embedding = self._fill_esm_embedding(
                    input_feature_dict=input_feature_dict,
                    entity_id_to_sequence=esm_entity_id_to_sequence,
                    seq_to_embedding=seq_to_embedding,
                    atom_array=atom_array,
                    token_array=token_array,
                )
                input_feature_dict["esm_token_embedding"] = esm_embedding
            else:
                # No protein sequences; use a zero embedding.
                esm_dim = input_feature_dict.get("esm_embedding_dim", 2560)
                token_index = input_feature_dict["token_index"]
                if token_index.dim() == 2:
                    N_token = token_index.shape[1]
                    esm_embedding = torch.zeros(1, N_token, esm_dim, device=token_index.device)
                else:
                    N_token = token_index.shape[0]
                    esm_embedding = torch.zeros(N_token, esm_dim, device=token_index.device)
                input_feature_dict["esm_token_embedding"] = esm_embedding

        # =================================================================
        # Step 2: assemble the Protenix inputs
        # =================================================================
        inplace_safe = not (self.training or torch.is_grad_enabled())
        chunk_size = self.configs.infer_setting.chunk_size if inplace_safe else None

        input_feature_dict = self.relative_position_encoding.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        # =================================================================
        # Step 3: Pairformer forward
        # =================================================================
        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=self.N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # =================================================================
        # Step 4: Return results
        # =================================================================
        if return_full_output:
            # Compute n_token and n_atom
            n_token = s.shape[0]
            n_atom = input_feature_dict.get("ref_pos", input_feature_dict.get("ref_pos")).shape[0]

            return PairformerOutput(
                s_inputs=s_inputs,
                s_trunk=s,
                z_trunk=z,
                esm_embedding=esm_embedding,
                input_feature_dict=input_feature_dict,
                n_token=n_token,
                n_atom=n_atom,
            )

        return s, esm_embedding

    def forward_diffusion(
        self,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        gt_coords: torch.Tensor,
        coordinate_mask: Optional[torch.Tensor] = None,
        sigma: float = 0.0,
        centre_only: bool = True,
        n_sample: int = 1,
        disable_amp: Optional[bool] = None,
        enable_efficient_fusion: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        pair_z: Optional[torch.Tensor] = None,
        p_lm: Optional[torch.Tensor] = None,
        c_l: Optional[torch.Tensor] = None,
        inplace_safe: Optional[bool] = None,
        attn_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract Diffusion a_token features

        This method runs the DiffusionModule to extract the a_token feature
        after 24 DiffusionTransformer layers and LayerNorm.

        Args:
            input_feature_dict: Protenix feature dictionary (must have 'relp' computed)
            s_inputs: InputEmbedder output [N_token, c_s_inputs]
            s_trunk: Pairformer single features [N_token, c_s]
            z_trunk: Pairformer pair features [N_token, N_token, c_z]
            gt_coords: Ground truth atom coordinates [N_atom, 3]
            coordinate_mask: Coordinate mask [N_atom], optional
            sigma: Noise level (0 = no noise, use GT coordinates directly)
            centre_only: Only center coordinates without random rotation.
                When True (default), ensures deterministic output.
            n_sample: Number of samples for diffusion (default 1).
                When n_sample > 1, each sample gets independent noise (if sigma > 0)
                or independent random augmentation (if centre_only=False).
                This is reserved for future multi-sample inference.
            disable_amp: Whether to disable automatic mixed precision (AMP) during
                diffusion forward pass. Default True to match official Protenix
                behavior (skip_amp.sample_diffusion=True in config).
            enable_efficient_fusion: Whether to enable efficient fusion optimization.
                When True, applies LayerNorm and permute_final_dims to z_pair before
                DiffusionTransformer. Default False (config.enable_efficient_fusion=True
                in official, but we default to False for stability).
            use_cache: Whether to use pre-computed cache tensors. When True,
                expects pair_z, p_lm, c_l to be provided (from prepare_diffusion_cache).
            pair_z: Pre-computed pair conditioning tensor [N_token, N_token, c_z].
                Required if use_cache=True. From diffusion_conditioning.prepare_cache().
            p_lm: Pre-computed local frame tensor. Optional cache from
                atom_attention_encoder.prepare_cache().
            c_l: Pre-computed confidence logits tensor. Optional cache from
                atom_attention_encoder.prepare_cache().
            inplace_safe: Whether to use inplace operations. If None (default),
                automatically determined based on training mode and grad status:
                inplace_safe = not (self.training or torch.is_grad_enabled())

        Returns:
            a_token: Diffusion token features.
                - If n_sample=1: [N_token, 768]
                - If n_sample>1: [n_sample, N_token, 768]
        """
        if n_sample > 1:
            # diffusion_chunk_size to be implemented for n_sample > 1
            raise NotImplementedError()

        if self.diffusion_module is None:
            raise RuntimeError(
                "Diffusion module not loaded. Use load_diffusion=True in __init__ "
                "or ProtenixEncoder.from_pretrained(..., load_diffusion=True)"
            )

        # Determine inplace_safe if not explicitly specified
        # This matches official Protenix behavior (protenix.py:783)
        if inplace_safe is None:
            inplace_safe = not (self.training or torch.is_grad_enabled())

        # Determine attn_chunk_size if not explicitly specified
        # This matches official Protenix behavior (protenix.py:304-307)
        if attn_chunk_size is None:
            if inplace_safe:
                attn_chunk_size = self.attn_chunk_size  # Use config value in inference mode
            # else: keep None for training mode (no chunking)
        elif attn_chunk_size == 0:
            attn_chunk_size = None  # Explicit disable

        # Apply config defaults if not explicitly specified
        if disable_amp is None:
            disable_amp = self.configs.skip_amp.sample_diffusion
        if enable_efficient_fusion is None:
            enable_efficient_fusion = self.configs.enable_efficient_fusion
        if use_cache is None:
            use_cache = self.configs.enable_diffusion_shared_vars_cache

        # Fix 1: Auto-compute cache when use_cache=True but cache tensors are not provided
        # This matches official Protenix behavior in protenix.py:449-473
        if use_cache and pair_z is None:
            pair_z, p_lm, c_l = self.prepare_diffusion_cache(
                input_feature_dict=input_feature_dict,
                z_trunk=z_trunk,
                disable_amp=disable_amp,
            )

        from protenix.model.utils import centre_random_augmentation, expand_at_dim

        device = s_trunk.device
        dtype = s_trunk.dtype

        # Convert floating point tensors to float32 when disable_amp=True
        # This matches the official autocasting_disable_decorator behavior
        if disable_amp:
            s_inputs = _convert_tensors_to_fp32(s_inputs)
            s_trunk = _convert_tensors_to_fp32(s_trunk)
            z_trunk = _convert_tensors_to_fp32(z_trunk)
            gt_coords = _convert_tensors_to_fp32(gt_coords)
            coordinate_mask = _convert_tensors_to_fp32(coordinate_mask)
            pair_z = _convert_tensors_to_fp32(pair_z)
            p_lm = _convert_tensors_to_fp32(p_lm)
            c_l = _convert_tensors_to_fp32(c_l)

        # 1. Process GT coordinates: center (and optionally random rotate)
        # Official Protenix works WITHOUT batch dimension.
        # Input: gt_coords [N_atom, 3] (no batch dimension)
        # centre_random_augmentation expects [..., N_atom, 3] and returns [..., n_sample, N_atom, 3]
        # When input is [N_atom, 3], output is [n_sample, N_atom, 3]
        target_dtype = torch.float32 if disable_amp else dtype
        x_gt = centre_random_augmentation(
            x_input_coords=gt_coords,
            N_sample=n_sample,
            mask=coordinate_mask,
            centre_only=centre_only,  # True: only center, no random rotation
        ).to(target_dtype)  # [n_sample, N_atom, 3]

        # 2. Add noise (skip if sigma=0)
        if sigma > 0:
            # Use local Generator with fixed seed for reproducibility; avoids polluting global random state
            _rng = torch.Generator(device=x_gt.device).manual_seed(42)
            noise = sigma * torch.randn(x_gt.shape, generator=_rng, device=x_gt.device, dtype=x_gt.dtype)
            x_noisy = x_gt + noise
        else:
            x_noisy = x_gt

        # 3. Construct t_hat_noise_level [n_sample]
        # IMPORTANT: Use small epsilon when sigma=0 to avoid log(0) = -inf in FourierEmbedding
        # DiffusionConditioning.forward() computes: torch.log(t_hat / sigma_data)
        # When t_hat=0, this produces -inf which propagates as NaN
        # NOTE: Use c_tau_last (computed from InferenceNoiseScheduler) for precise alignment
        # with official Protenix inference. c_tau_last is noise_schedule[-2], the last
        # non-zero noise level before t_N=0. This matches the final denoising step output.
        # Reference: generator.py:183-185 shows the denoising loop iterates with c_tau_last
        # from noise_schedule[:-1], and the final c_tau_last value is noise_schedule[-2].
        t_hat_value = sigma if sigma > 0 else self.c_tau_last
        t_hat = torch.full((n_sample,), t_hat_value, device=device, dtype=target_dtype)

        # 4. Prepare Diffusion inputs
        # Official Protenix does NOT add batch dimension - tensors stay as:
        # s_inputs: [N_token, c_s_inputs]
        # s_trunk: [N_token, c_s]
        # z_trunk: [N_token, N_token, c_z]
        # The diffusion module handles expansion internally via expand_at_dim

        # 5. Extract a_token from DiffusionModule
        a_token = self._extract_a_token_from_diffusion(
            r_noisy=x_noisy,
            t_hat_noise_level=t_hat,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=None if pair_z is not None else z_trunk,
            n_sample=n_sample,
            disable_amp=disable_amp,
            enable_efficient_fusion=enable_efficient_fusion,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            inplace_safe=inplace_safe,
            attn_chunk_size=attn_chunk_size,
        )

        # 6. Handle n_sample dimension
        # Output from _extract_a_token_from_diffusion: [n_sample, N_token, 768]
        # When n_sample=1, squeeze to [N_token, 768]
        if n_sample == 1:
            a_token = a_token.squeeze(0)  # [N_token, 768]

        return a_token

    def prepare_diffusion_cache(
        self,
        input_feature_dict: dict[str, Any],
        z_trunk: torch.Tensor,
        disable_amp: bool = True,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Pre-compute cache tensors for diffusion forward pass.

        This method computes pair_z, p_lm, and c_l tensors that can be reused
        across multiple forward_diffusion calls (e.g., for different noise levels
        or multiple samples). This matches the official Protenix caching behavior
        in protenix.py:449-473.

        Args:
            input_feature_dict: Protenix feature dictionary (must have 'relp' computed)
            z_trunk: Pairformer pair features [N_token, N_token, c_z]
            disable_amp: Whether to disable AMP during cache computation
            inplace_safe: Whether to use inplace operations (default False)

        Returns:
            tuple of (pair_z, p_lm, c_l):
                - pair_z: Pre-computed pair conditioning [N_token, N_token, c_z]
                - p_lm: Pre-computed local frame tensor (or None)
                - c_l: Pre-computed confidence logits (or None)

        Usage:
            # Pre-compute cache once
            pair_z, p_lm, c_l = encoder.prepare_diffusion_cache(
                input_feature_dict, z_trunk
            )

            # Use cache for multiple forward calls
            a_token = encoder.forward_diffusion(
                ...,
                use_cache=True,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
            )
        """
        if self.diffusion_module is None:
            raise RuntimeError(
                "Diffusion module not loaded. Use load_diffusion=True in __init__ "
                "or ProtenixEncoder.from_pretrained(..., load_diffusion=True)"
            )

        diffusion = self.diffusion_module

        # Convert floating point tensors to float32 when disable_amp=True
        # This matches the official autocasting_disable_decorator behavior
        if disable_amp:
            z_trunk = _convert_tensors_to_fp32(z_trunk)
            # Convert input_feature_dict floating point tensors (make a copy to avoid modifying original)
            input_feature_dict = _convert_tensors_to_fp32(dict(input_feature_dict))

        amp_context = torch.amp.autocast('cuda', enabled=not disable_amp)

        with amp_context:
            # 1. Compute pair_z cache (from DiffusionConditioning.prepare_cache)
            # See protenix.py:450-454
            pair_z = diffusion.diffusion_conditioning.prepare_cache(
                input_feature_dict["relp"],
                z_trunk,
                inplace_safe=False,
            )

            # 2. Compute p_lm and c_l cache (from AtomAttentionEncoder.prepare_cache)
            # See protenix.py:455-470
            p_lm, c_l = diffusion.atom_attention_encoder.prepare_cache(
                input_feature_dict["ref_pos"],
                input_feature_dict["ref_charge"],
                input_feature_dict["ref_mask"],
                input_feature_dict["ref_element"],
                input_feature_dict["ref_atom_name_chars"],
                input_feature_dict["atom_to_token_idx"],
                input_feature_dict["d_lm"],
                input_feature_dict["v_lm"],
                input_feature_dict.get("pad_info", {}),
                "",  # mode placeholder
                pair_z,
                inplace_safe=False,
            )

        return pair_z, p_lm, c_l

    def _extract_a_token_from_diffusion(
        self,
        r_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Any],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: Optional[torch.Tensor],
        n_sample: int = 1,
        disable_amp: bool = True,
        enable_efficient_fusion: bool = False,
        pair_z: Optional[torch.Tensor] = None,
        p_lm: Optional[torch.Tensor] = None,
        c_l: Optional[torch.Tensor] = None,
        inplace_safe: bool = False,
        attn_chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Extract a_token from DiffusionModule (after LayerNorm)

        This is a simplified version of DiffusionModule.f_forward() that only
        runs up to layernorm_a without the AtomAttentionDecoder.

        NOTE: This method follows official Protenix convention where tensors do NOT
        have a batch dimension. The diffusion module handles n_sample expansion internally.

        Args:
            r_noisy: Scaled noisy coordinates [n_sample, N_atom, 3]
            t_hat_noise_level: Noise level [n_sample]
            input_feature_dict: Protenix feature dictionary
            s_inputs: InputEmbedder output [N_token, c_s_inputs]
            s_trunk: Pairformer single features [N_token, c_s]
            z_trunk: Pairformer pair features [N_token, N_token, c_z]
            n_sample: Number of samples
            disable_amp: Whether to disable AMP (automatic mixed precision)
            enable_efficient_fusion: Whether to enable efficient fusion optimization
            use_cache: Whether to use pre-computed cache tensors
            pair_z: Pre-computed pair conditioning tensor [N_token, N_token, c_z]
            p_lm: Pre-computed local frame tensor
            c_l: Pre-computed confidence logits tensor
            inplace_safe: Whether to use inplace operations (default False)

        Returns:
            a_token: [n_sample, N_token, 768]
        """
        from protenix.model.utils import expand_at_dim, permute_final_dims

        diffusion = self.diffusion_module

        # Convert floating point tensors to float32 when disable_amp=True
        # This matches the official autocasting_disable_decorator behavior
        if disable_amp:
            r_noisy = _convert_tensors_to_fp32(r_noisy)
            t_hat_noise_level = _convert_tensors_to_fp32(t_hat_noise_level)
            s_inputs = _convert_tensors_to_fp32(s_inputs)
            s_trunk = _convert_tensors_to_fp32(s_trunk)
            if z_trunk is not None:
                z_trunk = _convert_tensors_to_fp32(z_trunk)
            pair_z = _convert_tensors_to_fp32(pair_z)
            p_lm = _convert_tensors_to_fp32(p_lm)
            c_l = _convert_tensors_to_fp32(c_l)
            # Convert input_feature_dict floating point tensors (make a copy to avoid modifying original)
            input_feature_dict = _convert_tensors_to_fp32(dict(input_feature_dict))

        # Use autocast context manager to optionally disable AMP
        # This matches official Protenix behavior (skip_amp.sample_diffusion=True)
        # When disable_amp=True, computations run in full precision (float32)
        amp_context = torch.amp.autocast('cuda', enabled=not disable_amp)

        with amp_context:
            # Scale positions (as in DiffusionModule.forward)
            sigma_data = diffusion.sigma_data
            r_noisy = (
                r_noisy
                / torch.sqrt(sigma_data**2 + t_hat_noise_level**2)[..., None, None]
            )

            # 1. Diffusion Conditioning
            # Cache logic is now handled at the caller level (forward_diffusion):
            # - When cache is used, z_trunk=None is passed and pair_z is pre-computed
            # - When cache is not used, z_trunk is passed and pair_z=None
            s_single, z_pair = diffusion.diffusion_conditioning(
                t_hat_noise_level,
                input_feature_dict["relp"],
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                inplace_safe=inplace_safe,
                use_conditioning=True,
            )

            # 2. Expand embeddings to match n_sample
            # expand_at_dim with n=n_sample (typically 1) handles dimension expansion
            # When n_sample=1, this is effectively a no-op that adds a dimension
            # s_trunk: [N_token, c_s] -> [n_sample, N_token, c_s] (when n_sample=1: [1, N_token, c_s])
            # z_pair: [N_token, N_token, c_z] -> [n_sample, N_token, N_token, c_z]
            s_trunk_exp = expand_at_dim(s_trunk, dim=-3, n=n_sample)
            z_pair_exp = expand_at_dim(z_pair, dim=-4, n=n_sample)

            # 3. Atom Attention Encoder
            # Cache logic is now handled at the caller level (forward_diffusion):
            # - When cache is used, p_lm and c_l are pre-computed
            # - When cache is not used, p_lm=None and c_l=None
            a_token, q_skip, c_skip, p_skip = diffusion.atom_attention_encoder(
                input_feature_dict["atom_to_token_idx"],
                input_feature_dict["ref_pos"],
                input_feature_dict["ref_charge"],
                input_feature_dict["ref_mask"],
                input_feature_dict["ref_atom_name_chars"],
                input_feature_dict["ref_element"],
                input_feature_dict["d_lm"],
                input_feature_dict["v_lm"],
                input_feature_dict.get("pad_info", {}),
                r_l=r_noisy,
                s=s_trunk_exp,
                z=z_pair_exp,
                p_lm=p_lm,
                c_l=c_l,
                inplace_safe=inplace_safe,
                chunk_size=attn_chunk_size,
            )

            # 4. Upcast
            a_token = a_token.to(dtype=torch.float32)

            # 5. Add s_single contribution
            # See diffusion.py:451-458 for official implementation
            if inplace_safe:
                a_token += diffusion.linear_no_bias_s(
                    diffusion.layernorm_s(s_single)
                )
            else:
                a_token = a_token + diffusion.linear_no_bias_s(
                    diffusion.layernorm_s(s_single)
                )

            # 6. Prepare z for DiffusionTransformer
            # See diffusion.py:459-463 for official implementation
            if enable_efficient_fusion:
                # Apply LayerNorm and permute for efficient fusion
                # normalize is LayerNorm(c_z, create_offset=False, create_scale=False)
                z = diffusion.normalize(z_pair_exp.to(dtype=torch.float32))
                z = permute_final_dims(z, [2, 0, 1]).contiguous()
            else:
                z = z_pair_exp.to(dtype=torch.float32)

            # 7. Diffusion Transformer (24 layers)
            a_token = diffusion.diffusion_transformer(
                a=a_token.to(dtype=torch.float32),
                s=s_single.to(dtype=torch.float32),
                z=z,
                inplace_safe=inplace_safe,
                chunk_size=attn_chunk_size,
                enable_efficient_fusion=enable_efficient_fusion,
            )

            # 8. Final LayerNorm (target feature)
            a_token = diffusion.layernorm_a(a_token)

        return a_token

    def _fill_esm_embedding(
        self,
        input_feature_dict: dict[str, Any],
        entity_id_to_sequence: dict[str, str],
        seq_to_embedding: dict[str, torch.Tensor],
        atom_array: Optional[Any] = None,
        token_array: Optional[Any] = None,
    ) -> torch.Tensor:
        """
        Fill ESM embeddings at the token level.

        **Matches the upstream ESMFeaturizer.__call__() logic exactly.**

        Upstream code (esm_featurizer.py:75-132):
            N_token = len(token_array)
            x = torch.zeros([N_token, self.embedding_dim])
            centre_atoms_indices = token_array.get_annotation("centre_atom_index")
            centre_atom_array = atom_array[centre_atoms_indices]
            is_protein = centre_atom_array.chain_mol_type == "protein"
            protein_entity_ids = set(centre_atom_array.label_entity_id[is_protein])

            for entity_id in protein_entity_ids:
                x_esm = self.load_esm_embedding(sequence)
                entity_mask = centre_atom_array.label_entity_id == entity_id
                res_index = centre_atom_array.res_id[entity_mask] - 1
                x[entity_mask] = x_esm[res_index]

        Args:
            input_feature_dict: contains token_index and related features
            entity_id_to_sequence: entity_id -> sequence mapping
            seq_to_embedding: sequence -> embedding mapping
            atom_array: Biotite AtomArray (used to read entity_id and res_id)
            token_array: TokenArray (used to read centre_atom_index)

        Returns:
            esm_token_embedding: [N_token, esm_dim]
        """
        # Read dimensions.
        token_index = input_feature_dict["token_index"]
        N_token = token_index.shape[0]

        # Determine the ESM embedding dim.
        esm_dim = input_feature_dict.get("esm_embedding_dim", 2560)
        if esm_dim is None:
            esm_dim = self.configs.get("esm", {}).get("embedding_dim", 2560)

        # Determine device and dtype.
        device = token_index.device
        # Read dtype from seq_to_embedding to stay consistent with the ESM output.
        dtype = torch.float32
        if seq_to_embedding:
            first_emb = next(iter(seq_to_embedding.values()))
            dtype = first_emb.dtype

        # Upstream: x = torch.zeros([N_token, self.embedding_dim])
        esm_embedding = torch.zeros(N_token, esm_dim, device=device, dtype=dtype)

        if entity_id_to_sequence is None or len(entity_id_to_sequence) == 0:
            return esm_embedding

        if len(seq_to_embedding) == 0:
            return esm_embedding

        if atom_array is None or token_array is None:
            return esm_embedding

        # Upstream: centre_atoms_indices = token_array.get_annotation("centre_atom_index")
        centre_atoms_indices = token_array.get_annotation("centre_atom_index")
        # Upstream: centre_atom_array = atom_array[centre_atoms_indices]
        centre_atom_array = atom_array[centre_atoms_indices]

        # Upstream: is_protein = centre_atom_array.chain_mol_type == "protein"
        is_protein = centre_atom_array.chain_mol_type == "protein"
        # Upstream: protein_entity_ids = set(centre_atom_array.label_entity_id[is_protein])
        protein_entity_ids = set(centre_atom_array.label_entity_id[is_protein])

        # Iterate over each protein entity and fill in the embedding.
        for entity_id in protein_entity_ids:
            if entity_id not in entity_id_to_sequence:
                continue

            sequence = entity_id_to_sequence[entity_id]
            if sequence not in seq_to_embedding:
                continue

            # Upstream: x_esm = self.load_esm_embedding(sequence)
            entity_emb = seq_to_embedding[sequence]

            # Upstream: entity_mask = centre_atom_array.label_entity_id == entity_id
            entity_mask = centre_atom_array.label_entity_id == entity_id

            # Upstream: res_index = centre_atom_array.res_id[entity_mask] - 1
            res_index = centre_atom_array.res_id[entity_mask] - 1

            # Upstream: x[entity_mask] = x_esm[res_index]
            res_index_tensor = torch.from_numpy(res_index).long().to(device)
            esm_embedding[entity_mask] = entity_emb[res_index_tensor].to(device)

        return esm_embedding

    @classmethod
    def _load_config(
        cls,
        config_path: str,
        dtype: str = "bf16",
    ):
        """
        Load the config file (internal helper, shared by from_config and from_pretrained).

        Args:
            config_path: config-file path (a JSON file, or a directory containing config.json)
            dtype: data type

        Returns:
            ConfigDict config object
        """
        import json
        from ml_collections.config_dict import ConfigDict

        # Accept either a directory or a file path.
        if os.path.isdir(config_path):
            config_json_path = os.path.join(config_path, "config.json")
        else:
            config_json_path = config_path

        if not os.path.exists(config_json_path):
            raise FileNotFoundError(f"Config file not found: {config_json_path}")

        print(f"Loading config from {config_json_path}")
        with open(config_json_path, "r") as f:
            configs_dict = json.load(f)

        configs_dict["dtype"] = dtype
        configs = ConfigDict(configs_dict)

        return configs

    @classmethod
    def from_config(
        cls,
        config_path: str,
        init_esm: bool = False,
        device: str = "cuda",
        dtype: str = "bf16",
    ) -> "ProtenixEncoder":
        """
        Create a ProtenixEncoder from a config file (does not load weights).

        Useful when you want random initialization or a custom weight-loading flow.

        Note: ESMEncoder uses a lazy-loading design — the model structure is built only
        when load_model(path) is called, which reads the config from the checkpoint and
        constructs the model. With init_esm=True only the ESMEncoder wrapper is created;
        the inner ESM2 model is still None and cannot do a forward pass.
        To use ESM, call from_pretrained() instead, or call load_esm_model(path) afterward.

        Args:
            config_path: config-file path (a JSON file, or a directory containing config.json)
            init_esm: whether to create the ESM encoder wrapper; default False.
                      Note: even when True, the ESM2 model structure is not built (lazy load).
                      Call load_esm_model(path) afterward to use it.
            device: device
            dtype: data type

        Returns:
            ProtenixEncoder instance (Protenix weights random-initialized, ESM not built).
        """
        import warnings

        configs = cls._load_config(config_path, dtype=dtype)

        # Create the Encoder (does not load any weights).
        # Note: ESMEncoder is lazy-loaded; at this point esm_encoder.model = None.
        encoder = cls(configs, load_esm=init_esm, esm_model_path=None)

        protenix_params = sum(
            p.numel() for n, p in encoder.named_parameters()
            if not n.startswith("esm_encoder")
        )

        print(f"Created ProtenixEncoder from config (weights not loaded)")
        print(f"  Protenix parameters: {protenix_params:,}")

        if init_esm:
            # ESMEncoder is lazy-loaded; model=None here, so it cannot be used yet.
            warnings.warn(
                "init_esm=True but ESMEncoder uses lazy loading. "
                "The ESM2 model structure is NOT built yet (esm_encoder.model=None). "
                "Call load_esm_model(path) to load ESM weights and build the model, "
                "or use from_pretrained() instead.",
                UserWarning,
            )
            print(f"  ESM: wrapper created, but model=None (lazy loading, call load_esm_model() to initialize)")

        encoder = encoder.to(device)

        return encoder

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        load_esm: bool = True,
        load_diffusion: bool = False,
        device: str = "cuda",
        dtype: str = "bf16",
        triangle_by_torch: Optional[bool] = None
    ) -> "ProtenixEncoder":
        """
        Load a fully-initialized ProtenixEncoder from a pretrained directory.

        Expected directory layout:
            pretrained_path/
            ├── config.json                      # Protenix config file
            ├── protenix_mini_ism_v0.5.0.pt     # Protenix model weights
            └── esm2_t36_3B_UR50D_ism.pt        # ESM model weights (optional)

        Args:
            pretrained_path: path to the pretrained model directory
            load_esm: whether to load the ESM encoder; default True
            load_diffusion: whether to load the DiffusionModule; default False
                When True you can call forward_diffusion() to extract a_token features.
            device: device
            dtype: data type
            triangle_by_torch: whether to use the PyTorch implementation of triangle ops

        Returns:
            ProtenixEncoder instance.
        """
        from argparse import Namespace
        torch.serialization.add_safe_globals([Namespace])

        # Check the directory exists.
        if not os.path.isdir(pretrained_path):
            raise FileNotFoundError(f"Pretrained directory not found: {pretrained_path}")

        # Locate the Protenix weight file.
        protenix_checkpoint_path = os.path.join(pretrained_path, "protenix_mini_ism_v0.5.0.pt")
        if not os.path.exists(protenix_checkpoint_path):
            raise FileNotFoundError(
                f"Protenix checkpoint not found in {pretrained_path}\n"
                "Expected: protenix_mini_ism_v0.5.0.pt"
            )

        # Locate the ESM weight file.
        esm_checkpoint_path = None
        if load_esm:
            esm_checkpoint_path = os.path.join(pretrained_path, "esm2_t36_3B_UR50D_ism.pt")
            if not os.path.exists(esm_checkpoint_path):
                raise FileNotFoundError(
                    f"ESM checkpoint not found in {pretrained_path}\n"
                    "Expected: esm2_t36_3B_UR50D_ism.pt"
                )

        # Load the config and create the model (reuses _load_config).
        configs = cls._load_config(pretrained_path, dtype=dtype)
        encoder = cls(
            configs,
            load_esm=load_esm,
            load_diffusion=load_diffusion,
            esm_model_path=None,
            triangle_by_torch=triangle_by_torch,
        )

        # Load Protenix weights.
        print(f"Loading Protenix checkpoint from {protenix_checkpoint_path}")
        checkpoint = torch.load(protenix_checkpoint_path, map_location="cpu")

        # Strip the DDP prefix.
        state_dict = checkpoint.get("model", checkpoint)
        sample_key = list(state_dict.keys())[0]
        if sample_key.startswith("module."):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        # Decide whether to load diffusion weights based on load_diffusion.
        if load_diffusion:
            # Load every weight (including diffusion_module).
            encoder_keys = [
                k for k in state_dict.keys()
                if not k.startswith(("confidence_head", "distogram_head"))
            ]
        else:
            # Drop diffusion + confidence-head weights.
            encoder_keys = [
                k for k in state_dict.keys()
                if not k.startswith(("diffusion_module", "confidence_head", "distogram_head"))
            ]
        encoder_state = {k: state_dict[k] for k in encoder_keys}

        # Load the Protenix weights.
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        if missing:
            # Drop ESM encoder-related missing keys (they are loaded separately).
            missing = [k for k in missing if not k.startswith("esm_encoder")]
            if missing:
                print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")

        protenix_params = sum(p.numel() for n, p in encoder.named_parameters() if not n.startswith("esm_encoder"))
        print(f"Loaded Protenix encoder with {protenix_params:,} parameters")

        if load_diffusion:
            diffusion_params = sum(
                p.numel() for n, p in encoder.named_parameters()
                if n.startswith("diffusion_module")
            )
            print(f"Loaded Diffusion module with {diffusion_params:,} parameters")

        # Load the ESM model.
        if load_esm and esm_checkpoint_path is not None:
            encoder.load_esm_model(esm_checkpoint_path)
        elif load_esm and esm_checkpoint_path is None:
            print("Warning: ESM checkpoint not found, ESM encoder will not be loaded")

        encoder = encoder.to(device)

        return encoder


# ============================================================================
# Auxiliary classes
# ============================================================================

class ProteinLLMProjector(nn.Module):
    """
    Project a Protenix embedding into the LLM embedding dimensionality.
    """

    def __init__(
        self,
        protenix_dim: int = 384,
        llm_dim: int = 4096,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        if hidden_dim is None:
            self.proj = nn.Linear(protenix_dim, llm_dim)
        else:
            self.proj = nn.Sequential(
                nn.Linear(protenix_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, llm_dim),
            )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.proj(s)
