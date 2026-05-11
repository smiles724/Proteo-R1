"""
ProteoR1Understand - protein multimodal language model built on Protenix + Qwen3.

Architecture:
- LLM: Qwen3 (AutoModelForCausalLM)
- Protein Encoder: ProtenixEncoder (ESM + Pairformer)
- Projector: MLP that projects the Protenix embedding into the LLM hidden size

Main differences vs PLLM (ProTrek + Qwen2.5):
1. Single encoder (Protenix) replaces the dual encoder (protein + structure).
2. Single projector replaces the dual projector.
3. Single placeholder <protein> replaces <aa_seq> + <3d_struct>.
4. Protenix produces a structure-aware embedding of shape (N_token, 384).

Usage:
    from proteor1.understand import ProteoR1UnderstandConfig, ProteoR1UnderstandProcessor, ProteoR1UnderstandModel

    # Build the processor.
    processor = ProteoR1UnderstandProcessor(
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507"),
        protenix_encoder_path="pretrained/protenix_mini_ism_v0.5.0",
    )

    # Build the model.
    config = ProteoR1UnderstandConfig(
        base_model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
        protenix_encoder_path="pretrained/protenix_mini_ism_v0.5.0",
        protein_token_id=processor.protein_token_id,
    )
    model = ProteoR1UnderstandModel(config)
    model.resize_token_embeddings(len(processor.tokenizer))

    # Process input.
    inputs = processor(text="Analyze this protein: <protein>", protein_json=json_entry)

    # Forward.
    outputs = model(**inputs)
"""

import logging
import os
import shutil
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Cache,
    GenerationMixin,
    PreTrainedModel,
)

# Reuse the Bagel sincos position-encoding implementation.
from proteor1.understand._pos_embed import get_1d_sincos_pos_embed_from_grid
from proteor1.understand.configuration import (
    ProteoR1UnderstandConfig,
)
from proteor1.understand.processing import (
    ProteoR1UnderstandProcessor,
)
from proteor1.understand.protenix_encoder.modeling_protenix_encoder import (
    ProtenixEncoder,
)

logger = logging.getLogger(__name__)

try:
    import flash_attn
except (ImportError, ModuleNotFoundError):
    flash_attn = None


def get_1d_sincos_pos_embed(embed_dim: int, max_positions: int) -> torch.Tensor:
    """
    Generate a 1D sincos position embedding (reuses the Bagel implementation).

    Args:
        embed_dim: embedding dimensionality
        max_positions: maximum number of positions

    Returns:
        pos_embed: [max_positions, embed_dim]
    """
    pos = np.arange(max_positions, dtype=np.float32)
    emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    return torch.from_numpy(emb).float()


class ProtenixPositionEmbedding(nn.Module):
    """
    Hierarchical protein position embedding (non-learnable).

    Analogous to Bagel's PositionEmbedding:
    - Bagel: 2D image -> concat(sincos(row), sincos(col))
    - This module: multi-chain protein -> concat(sincos(residue_index), sincos(asym_id))

    Dim layout:
    - First hidden_size/2: residue position within the chain (residue_index)
    - Last hidden_size/2: chain identifier (asym_id)

    Args:
        hidden_size: LLM hidden size (must be even)
        max_residues: max residues per chain (default 4096)
        max_chains: max number of chains (default 64)
    """

    def __init__(
        self,
        hidden_size: int,
        max_residues: int = 4096,
        max_chains: int = 64,
    ):
        super().__init__()
        assert hidden_size % 2 == 0, f"hidden_size must be even, got {hidden_size}"

        self.hidden_size = hidden_size
        self.max_residues = max_residues
        self.max_chains = max_chains

        # Precompute the sincos tables (frozen, Bagel-style).
        residue_embed = get_1d_sincos_pos_embed(hidden_size // 2, max_residues)
        chain_embed = get_1d_sincos_pos_embed(hidden_size // 2, max_chains)

        # Register as frozen Parameters (matches Bagel's PositionEmbedding).
        # We use nn.Parameter rather than register_buffer because FSDP2 shards Parameters
        # but not buffers — leaving them as buffers triggers a missing _local_tensor attr in merge_fsdp.py.
        self.residue_embed = nn.Parameter(residue_embed, requires_grad=False)
        self.chain_embed = nn.Parameter(chain_embed, requires_grad=False)

    def forward(
        self,
        residue_index: torch.LongTensor,
        asym_id: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Args:
            residue_index: [N] residue index within the chain (Protenix is 1-based)
            asym_id: [N] chain ID (Protenix is 0-based)

        Returns:
            pos_embed: [N, hidden_size]
        """
        # Protenix residue_index is 1-based; convert to 0-based.
        res_idx = (residue_index - 1).clamp(0, self.max_residues - 1)
        # asym_id is already 0-based.
        chain_idx = asym_id.clamp(0, self.max_chains - 1)

        # Look up + concat (matches Bagel's concat(emb_h, emb_w)).
        res_emb = self.residue_embed[res_idx]  # [N, D/2]
        chain_emb = self.chain_embed[chain_idx]  # [N, D/2]

        return torch.cat([res_emb, chain_emb], dim=-1)  # [N, D]


class PrefixProjector(nn.Module):
    """
    Per-token MLP: (N_token, D_in) -> (N_token, D_out)

    Project a Protenix embedding into the LLM hidden dimensionality.

    Args:
        in_dim: input dimensionality
        mid_dim: hidden-layer dimensionality
        out_dim: output dimensionality
        dropout: dropout probability
        s_dim: s-embedding dimensionality; when provided, applies LayerNorm to the last s_dim dims
            - "s" mode: in_dim=384, s_dim=384
            - "concat" mode: in_dim=2944, s_dim=384
            - "esm" mode: s_dim=None
    """

    def __init__(
        self,
        in_dim: int,
        mid_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        s_dim: int = None,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.s_dim = s_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim, bias=True),
        )

        # Apply LayerNorm to the s-embedding portion.
        self.s_prenorm = nn.LayerNorm(s_dim) if s_dim is not None else None

        if dtype is not None or device is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_token, D_in] or [B, N_token, D_in]
        if self.s_prenorm is not None:
            # Apply LayerNorm to the last s_dim dims (uniform handling of "s" and "concat" modes).
            x = torch.cat([x[..., : -self.s_dim], self.s_prenorm(x[..., -self.s_dim :])], dim=-1)
        return self.net(x)


class ProteoR1UnderstandModel(PreTrainedModel, GenerationMixin):
    """
    ProteoR1UnderstandModel: Protenix + Qwen3 protein multimodal language model.

    Naming follows the transformers MLLM convention (e.g. LlavaForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration).

    Architecture:
    - llm: Qwen3ForCausalLM
    - protenix_encoder: ProtenixEncoder (ESM + Pairformer)
    - projector: PrefixProjector (MLP)

    Forward:
    1. text embedding: llm.get_input_embeddings()(input_ids)
    2. protein embedding: protenix_encoder(input_feature_dict) -> s (384-dim) or esm_embedding (2560-dim)
    3. project: projector(protein_embedding) -> (N_token, hidden_size)
    4. fill: text_embeds[protein_token_mask] = projected_protein_embeds
    5. llm forward: llm(inputs_embeds=text_embeds, ...)
    """

    config_class = ProteoR1UnderstandConfig
    base_model_prefix = "proteor1_understand"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _supports_flash_attn_2 = True
    # The ESM model's lm_head.weight is tied to embed_tokens.weight.
    _tied_weights_keys = [
        "protenix_encoder.esm_encoder.model.lm_head.weight",
    ]

    def __init__(self, config: ProteoR1UnderstandConfig):
        super().__init__(config)

        attn_implementation = "flash_attention_2" if flash_attn is not None else "sdpa"
        if attn_implementation == "sdpa":
            logger.warning(f"flash_attention_2 is not activated for {self.__class__.__name__}")

        # ============ LLM ============
        if config.load_pretrained:
            self.llm = AutoModelForCausalLM.from_pretrained(
                config.base_model_name_or_path,
                torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
            )
        else:
            llm_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
            if config.vocab_size is not None:
                llm_config.vocab_size = config.vocab_size
                logger.info(f"Using vocab_size from config ({config.vocab_size})")
            self.llm = AutoModelForCausalLM.from_config(
                llm_config,
                torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
                attn_implementation=attn_implementation,
            )

        # Sync vocab_size
        if self.config.vocab_size is None and hasattr(self.llm.config, "vocab_size"):
            self.config.vocab_size = self.llm.config.vocab_size

        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = False

        self.hidden_size = self.llm.config.hidden_size

        # ============ Protenix Encoder ============
        # Read LLM dtype, used to unify the whole model's dtype (FSDP2 requires all parameters to share a dtype).
        model_dtype = next(self.llm.parameters()).dtype

        if config.protenix_encoder_path is not None and os.path.isdir(config.protenix_encoder_path):
            self.protenix_encoder = ProtenixEncoder.from_pretrained(
                config.protenix_encoder_path,
                triangle_by_torch=config.triangle_by_torch,
                load_esm=config.load_esm,
                device="cpu",  # load on CPU first; move uniformly later
            )
            for module in self.protenix_encoder.modules():
                module._is_hf_initialized = True

            # Cast protenix_encoder to the same dtype as the LLM (FSDP2 requires uniform dtype).
            # Upstream Protenix config.json defaults to dtype="bf16" and supports bf16 inference.
            # The forward pass uses autocast to keep numerics stable.
            self.protenix_encoder = self.protenix_encoder.to(dtype=model_dtype)
            logger.info(f"Converted protenix_encoder to {model_dtype} (official Protenix supports bf16)")
        else:
            self.protenix_encoder = None
            if config.protenix_encoder_path is not None:
                logger.warning(f"protenix_encoder_path '{config.protenix_encoder_path}' not found")

        # ============ Projector ============

        # The input dim follows embedding_mode.
        projector_in_dim = config.projector_input_dim  # 384, 2560, or 2944

        # proj_hid defaults to the LLM hidden_size.
        proj_hid = config.proj_hid if config.proj_hid is not None else self.hidden_size

        # In "s" and "concat" modes, pass s_dim to enable LayerNorm on the s embedding.
        embedding_mode = config.get_embedding_mode()
        s_dim = config.protenix_s_dim if embedding_mode in ("s", "esm+s") else None

        self.projector = PrefixProjector(
            in_dim=projector_in_dim,
            mid_dim=proj_hid,
            out_dim=self.hidden_size,
            dropout=config.dropout,
            s_dim=s_dim,
            dtype=model_dtype,
        )

        # ============ Protenix Position Embedding (optional) ============
        if config.use_protenix_pos_embed:
            self.protenix_pos_embed = ProtenixPositionEmbedding(
                hidden_size=self.hidden_size,
                max_residues=config.max_residues,
                max_chains=config.max_chains,
            )
            # Cast to model dtype (matching the projector).
            self.protenix_pos_embed = self.protenix_pos_embed.to(dtype=model_dtype)
            logger.info(
                f"Protenix position embedding enabled: "
                f"max_residues={config.max_residues}, max_chains={config.max_chains}, "
                f"hidden_size={self.hidden_size}"
            )
        else:
            self.protenix_pos_embed = None

        # ============ CDR Mask Embedding (optional) ============
        if config.use_cdr_mask_embedding:
            # Use a zero vector as the mask embedding (frozen).
            # CDR-region embeddings will be replaced with this zero vector.
            self.cdr_mask_embedding = nn.Parameter(
                torch.zeros(self.hidden_size, dtype=model_dtype),
                requires_grad=False,
            )
            logger.info(
                f"CDR mask embedding enabled: hidden_size={self.hidden_size}, " f"initialized with zeros (frozen)"
            )
        else:
            self.cdr_mask_embedding = None

        # ============ Freeze ============
        self.freeze_params()

        # ============ Post init ============
        self.config.load_pretrained = False
        self.config.freeze_choice = "none"
        self.config.hidden_size = self.llm.config.hidden_size
        # Sync num_hidden_layers for generate() API (DynamicCache needs it)
        self.config.num_hidden_layers = self.llm.config.num_hidden_layers
        # Sync tie_word_embeddings from LLM config so that save_pretrained/from_pretrained
        # correctly handle weight tying. Without this, PretrainedConfig's default (True)
        # causes lm_head.weight to be dropped during save and replaced with
        # embed_tokens.weight on load, corrupting models where they differ (e.g. Qwen3-8B).
        self.config.tie_word_embeddings = self.llm.config.tie_word_embeddings

        self._sync_generation_config()
        self._merge_no_split_modules()

        # Log protein token IDs for verification
        logger.info(
            f"ProteoR1UnderstandModel initialized with "
            f"protein_token_id={self.config.protein_token_id}, "
            f"protein_start_token_id={self.config.protein_start_token_id}, "
            f"protein_end_token_id={self.config.protein_end_token_id}"
        )

        # TODO: self.post_init()

    def _merge_no_split_modules(self):
        """Merge _no_split_modules from every child module."""
        all_no_split_modules = []

        # _no_split_modules from the LLM (Qwen3).
        if hasattr(self.llm, "_no_split_modules") and self.llm._no_split_modules:
            all_no_split_modules.extend(self.llm._no_split_modules)

        # Key ProtenixEncoder modules (must not be split by FSDP/DeepSpeed).
        if self.protenix_encoder is not None:
            # ESM2's TransformerLayer (Transformer block).
            all_no_split_modules.append("TransformerLayer")
            # Protenix Pairformer-related modules.
            all_no_split_modules.append("PairformerBlock")
            all_no_split_modules.append("MSABlock")

        self._no_split_modules = list(set(all_no_split_modules))
        logger.info(f"Merged _no_split_modules: {self._no_split_modules}")

    def _sync_generation_config(self):
        """Synchronize generation config."""
        if hasattr(self.config, "_name_or_path") and self.config._name_or_path:
            try:
                from transformers import GenerationConfig

                saved_gen_config = GenerationConfig.from_pretrained(self.config._name_or_path)
                self.generation_config = saved_gen_config
                logger.info(f"Loaded generation_config from: {self.config._name_or_path}")
                return
            except Exception:
                pass

        if hasattr(self.config, "base_model_name_or_path") and self.config.base_model_name_or_path:
            try:
                from transformers import GenerationConfig

                base_gen_config = GenerationConfig.from_pretrained(self.config.base_model_name_or_path)
                self.generation_config = base_gen_config
                logger.info(f"Loaded generation_config from base: {self.config.base_model_name_or_path}")
            except Exception as e:
                logger.warning(f"Could not load generation_config: {e}")
                if hasattr(self.llm, "generation_config") and self.llm.generation_config is not None:
                    self.generation_config = self.llm.generation_config
        else:
            if hasattr(self.llm, "generation_config") and self.llm.generation_config is not None:
                self.generation_config = self.llm.generation_config

        # Sync token ids (with fallback to generation_config)
        if hasattr(self.llm.config, "eos_token_id"):
            self.config.eos_token_id = self.llm.config.eos_token_id
        if getattr(self.config, "eos_token_id", None) is None and hasattr(self.generation_config, "eos_token_id"):
            self.config.eos_token_id = self.generation_config.eos_token_id

        if hasattr(self.llm.config, "pad_token_id"):
            self.config.pad_token_id = self.llm.config.pad_token_id
        if getattr(self.config, "pad_token_id", None) is None and hasattr(self.generation_config, "pad_token_id"):
            self.config.pad_token_id = self.generation_config.pad_token_id

        if hasattr(self.llm.config, "bos_token_id"):
            self.config.bos_token_id = self.llm.config.bos_token_id
        if getattr(self.config, "bos_token_id", None) is None and hasattr(self.generation_config, "bos_token_id"):
            self.config.bos_token_id = self.generation_config.bos_token_id

    def freeze_params(self, freeze_choice: str = None):
        """
        Freeze parameters (conservative: only mutate what must change; leave the rest alone).

        Args:
            freeze_choice:
                - "none": do not change any parameter (keep the model defaults)
                - "encoder": freeze protenix_encoder; leave the rest alone
                - "non_projector": freeze llm and protenix_encoder; leave projector alone
        """
        if freeze_choice is None:
            freeze_choice = self.config.freeze_choice

        if freeze_choice == "none":
            # Keep defaults; do not modify anything.
            pass

        elif freeze_choice == "encoder":
            # Freeze the encoder only; preserve the state of every other parameter.
            for name, param in self.named_parameters():
                if name.startswith("protenix_encoder."):
                    param.requires_grad = False

        elif freeze_choice == "non_projector":
            # Freeze every parameter outside the projector.
            for name, param in self.named_parameters():
                if not name.startswith("projector."):
                    param.requires_grad = False

        else:
            raise ValueError(f"Unknown freeze_choice: {freeze_choice}")

    def _post_init(self):
        """Called after from_pretrained loads the model."""
        super()._post_init()
        # Re-sync generation config after loading from checkpoint
        self._sync_generation_config()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for all components."""
        # Enable for LLM
        if hasattr(self.llm, "gradient_checkpointing_enable"):
            self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

        # Enable for ProtenixEncoder (when it has trainable parameters).
        # Note: native ESM2 and Protenix modules need their own handling.
        # ProtenixEncoder internally uses checkpoint_blocks, so no extra wiring is needed here.

        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for all components."""
        # Disable for LLM
        if hasattr(self.llm, "gradient_checkpointing_disable"):
            self.llm.gradient_checkpointing_disable()

        self.gradient_checkpointing = False

    def load_protenix_weights(self, protenix_ckpt_path: str = None):
        """
        Explicitly load Protenix encoder weights.

        Similar to PLLM.load_protrek_weights(): provides the ability to load weights from a checkpoint.
        ProtenixEncoder.from_pretrained() is already called in __init__; this method exists for cases
        that need to load weights separately.

        Args:
            protenix_ckpt_path: path to the Protenix checkpoint.
                When None, fall back to config.protenix_encoder_path.
        """
        if protenix_ckpt_path is None:
            protenix_ckpt_path = self.config.protenix_encoder_path

        if protenix_ckpt_path is None or not os.path.isdir(protenix_ckpt_path):
            raise ValueError(
                f"protenix_ckpt_path not found: {protenix_ckpt_path}. "
                f"Please provide a valid path to the Protenix checkpoint."
            )

        if self.protenix_encoder is None:
            # Read LLM dtype to unify the dtype (FSDP2 requirement).
            model_dtype = next(self.llm.parameters()).dtype

            self.protenix_encoder = ProtenixEncoder.from_pretrained(
                protenix_ckpt_path,
                load_esm=self.config.load_esm,
                triangle_by_torch=self.config.triangle_by_torch,
                device="cpu",
            )
            # Cast to the same dtype as the LLM (upstream Protenix supports bf16).
            self.protenix_encoder = self.protenix_encoder.to(dtype=model_dtype)
            logger.info(f"Loaded ProtenixEncoder from {protenix_ckpt_path} (dtype={model_dtype})")
            # Re-merge _no_split_modules — __init__ may have skipped Protenix modules.
            self._merge_no_split_modules()
        else:
            logger.warning(f"ProtenixEncoder already exists. To reload, set self.protenix_encoder = None first.")

    # ============ Embedding-related methods ============

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llm.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(
        self,
        new_num_tokens: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """Resize token embeddings"""
        model_embeds = self.llm.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)

        if hasattr(self.llm.config, "vocab_size"):
            self.config.vocab_size = self.llm.config.vocab_size
            logger.info(f"Synced vocab_size: {self.config.vocab_size}")

        return model_embeds

    def get_decoder(self):
        return self.llm.model

    def set_decoder(self, decoder):
        self.llm.model = decoder

    @property
    def device(self):
        return next(self.llm.parameters()).device

    def train(self, mode: bool = True):
        """Override train()"""
        super().train(mode)
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = not mode
        return self

    def eval(self):
        """Override eval()"""
        super().eval()
        if hasattr(self.llm.config, "use_cache"):
            self.llm.config.use_cache = True
        return self

    # ============ Forward ============

    @staticmethod
    def _is_first_forward(past_key_values) -> bool:
        """
        Determine whether this is the first forward (prompt phase).

        In generate(), past_key_values is initialized to an empty DynamicCache() even in the prompt
        phase, so `past_key_values is None` is insufficient — we must also check whether the cache is empty.

        Returns:
            True: prompt phase (first forward)
            False: generation phase (autoregressive decoding)
        """
        if past_key_values is None:
            return True
        if hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0:
            return True
        return False

    def _prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor],
        past_key_values: Optional[Tuple],
        # Protenix inputs - on-the-fly mode
        protenix_input_feature_dict: Optional[dict] = None,
        protenix_atom_array: Optional[Any] = None,
        protenix_token_array: Optional[Any] = None,
        # Protenix inputs - precomputed mode
        protenix_s_embedding: Optional[torch.Tensor] = None,
        protenix_esm_embedding: Optional[torch.Tensor] = None,
        protenix_a_token: Optional[torch.Tensor] = None,
        protenix_embedding_attention_mask: Optional[torch.Tensor] = None,
        # Protenix position info - precomputed mode (used by protenix_pos_embed)
        protenix_residue_index: Optional[torch.Tensor] = None,
        protenix_asym_id: Optional[torch.Tensor] = None,
        # CDR mask - precomputed mode (used by cdr_mask_embedding)
        protenix_cdr_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare inputs_embeds: merge protein embedding into text embedding.

        Factored out so forward and liger forward can share the protein-embedding wiring.

        Args:
            input_ids: [B, L] text token ids
            inputs_embeds: when already provided, returned as-is
            past_key_values: used to detect the first forward
            protenix_input_feature_dict: Protenix feature dict (on-the-fly mode)
            protenix_atom_array: biotite AtomArray (on-the-fly mode, optional)
            protenix_token_array: TokenArray (on-the-fly mode, optional)
            protenix_s_embedding: [B, N_max, 384] precomputed s embedding (precomputed mode)
            protenix_esm_embedding: [B, N_max, 2560] precomputed ESM embedding (precomputed mode)
            protenix_a_token: [B, N_max, 768] precomputed a_token embedding (precomputed mode)
            protenix_embedding_attention_mask: [B, N_max] embedding attention mask (precomputed mode)
            protenix_residue_index: [B, N_max] residue index within the chain (precomputed mode; for protenix_pos_embed)
            protenix_asym_id: [B, N_max] chain ID (precomputed mode; for protenix_pos_embed)
            protenix_cdr_mask: [B, N_max] CDR mask (precomputed mode; for cdr_mask_embedding)

        Returns:
            inputs_embeds: [B, L, hidden_size] text embedding with the protein embedding merged in
        """
        if input_ids is None and inputs_embeds is None:
            raise RuntimeError("input_ids and inputs_embeds cannot both be None")

        # If inputs_embeds is already provided, return it directly.
        if inputs_embeds is not None:
            return inputs_embeds

        # Read text embeddings.
        text_embeds = self.get_input_embeddings()(input_ids)

        # Detect the first forward (we only process the protein embedding on the first forward).
        if not self._is_first_forward(past_key_values):
            # Subsequent forwards (autoregressive generation) skip protein embedding handling.
            return text_embeds

        # ========== Precomputed mode: use the precomputed embedding directly ==========
        if protenix_s_embedding is not None or protenix_esm_embedding is not None:
            # embedding_mode picks which embedding to use.
            embedding_mode = self.config.get_embedding_mode()

            if embedding_mode == "esm":
                # ESM embedding only (2560).
                if protenix_esm_embedding is None:
                    raise RuntimeError(
                        "embedding_mode='esm' but protenix_esm_embedding is None. "
                        "Please provide precomputed ESM embedding."
                    )
                protein_embedding = protenix_esm_embedding  # [B, N_max, 2560]

            elif embedding_mode == "s":
                # Protenix s embedding only (384).
                if protenix_s_embedding is None:
                    raise RuntimeError(
                        "embedding_mode='s' but protenix_s_embedding is None. "
                        "Please provide precomputed s embedding."
                    )
                protein_embedding = protenix_s_embedding  # [B, N_max, 384]

            elif embedding_mode == "a":
                if protenix_a_token is None:
                    raise RuntimeError(
                        "embedding_mode='a' but protenix_a_token is None. "
                        "Please provide precomputed a_token embedding."
                    )
                protein_embedding = protenix_a_token  # [B, N_max, 768]

            elif embedding_mode == "esm+s":
                # ESM + s concatenated (2944).
                if protenix_esm_embedding is None or protenix_s_embedding is None:
                    raise RuntimeError(
                        "embedding_mode='esm+s' but protenix_esm_embedding or protenix_s_embedding is None. "
                        "Please provide both precomputed embeddings."
                    )
                protein_embedding = torch.cat(
                    [protenix_esm_embedding, protenix_s_embedding], dim=-1
                )  # [B, N_max, 2944]

            elif embedding_mode == "esm+a":
                if protenix_esm_embedding is None or protenix_a_token is None:
                    raise RuntimeError(
                        "embedding_mode='esm+a' but protenix_esm_embedding or protenix_a_token is None. "
                        "Please provide both precomputed embeddings."
                    )
                protein_embedding = torch.cat(
                    [protenix_esm_embedding, protenix_a_token], dim=-1
                )  # [B, N_max, 2560 + 768]

            else:
                raise ValueError(embedding_mode)

            # Cast dtype to match the projector.
            protein_embedding = protein_embedding.to(self.projector.net[0].weight.dtype)

            # Project to the LLM hidden size.
            projected_protein = self.projector(protein_embedding)  # [B, N_max, hidden_size]

            # Extract the valid portion from the padded embedding: [B, N_max, D] + [B, N_max] -> [N_total_valid, D].
            # protenix_embedding_attention_mask comes from data_collator: 1 = valid position, 0 = padding.
            valid_mask = protenix_embedding_attention_mask.bool()
            valid_embeddings = projected_protein[valid_mask]

            # Add the Protenix position embedding (if enabled).
            # Precomputed mode: position info comes from protenix_residue_index/protenix_asym_id.
            if self.protenix_pos_embed is not None:
                if protenix_residue_index is None or protenix_asym_id is None:
                    raise RuntimeError(
                        "protenix_pos_embed is enabled but protenix_residue_index/protenix_asym_id not provided. "
                        "Please provide position info for precomputed mode."
                    )
                # Extract the valid portion of padded residue_index/asym_id.
                valid_residue_index = protenix_residue_index[valid_mask]
                valid_asym_id = protenix_asym_id[valid_mask]

                pos_embed = self.protenix_pos_embed(valid_residue_index, valid_asym_id)
                valid_embeddings = valid_embeddings + pos_embed.to(valid_embeddings.dtype)

            # Apply the CDR mask embedding (if enabled).
            # CDR-position embeddings are replaced with cdr_mask_embedding (typically zero).
            if self.cdr_mask_embedding is not None and protenix_cdr_mask is not None:
                # Extract the valid portion of padded cdr_mask.
                valid_cdr_mask = protenix_cdr_mask[valid_mask]  # [N_valid] bool
                # Replace CDR-position embeddings with mask_embedding.
                valid_embeddings[valid_cdr_mask] = self.cdr_mask_embedding.to(valid_embeddings.dtype)

            # Use the attention mask to extract valid embeddings; fill them into the protein_token_mask positions.
            protein_token_mask = input_ids == self.config.protein_token_id  # [B, L]

            # Verify counts match.
            expected_count = protein_token_mask.sum().item()
            actual_count = valid_embeddings.shape[0]

            if expected_count != actual_count:
                # Detailed debug info.
                batch_size = input_ids.shape[0]
                per_sample_expected = [protein_token_mask[i].sum().item() for i in range(batch_size)]
                per_sample_emb_mask = [protenix_embedding_attention_mask[i].sum().item() for i in range(batch_size)]
                raise ValueError(
                    f"protein_token count mismatch: expected {expected_count} in input_ids, "
                    f"got {actual_count} valid embeddings. "
                    f"batch_size={batch_size}, "
                    f"per_sample_expected={per_sample_expected}, "
                    f"per_sample_emb_mask={per_sample_emb_mask}"
                )

            # Write into text_embeds.
            text_embeds[protein_token_mask] = valid_embeddings.to(text_embeds.dtype)

        # ========== On-the-fly mode: use ProtenixEncoder ==========
        elif self.protenix_encoder is not None:
            if protenix_input_feature_dict is None:
                raise RuntimeError("protenix_input_feature_dict cannot be None")

            # Compute the protein embedding.
            # autocast for bf16 inference (matches upstream Protenix inference).
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                s, esm_embedding = self.protenix_encoder(
                    input_feature_dict=protenix_input_feature_dict,
                    atom_array=protenix_atom_array,
                    token_array=protenix_token_array,
                )

            # embedding_mode picks which embedding to use.
            embedding_mode = self.config.get_embedding_mode()

            if embedding_mode == "esm":
                # ESM embedding only (2560).
                protein_embedding = esm_embedding  # [N_token, 2560]
            elif embedding_mode == "concat":
                # ESM + s concatenated (2944).
                protein_embedding = torch.cat([esm_embedding, s], dim=-1)  # [N_token, 2944]
            else:
                # Protenix s embedding only (384).
                protein_embedding = s  # [N_token, 384]

            # Project to the LLM hidden size (projector is already bfloat16).
            projected_protein = self.projector(
                protein_embedding.to(self.projector.net[0].weight.dtype)
            )  # [N_token, hidden_size]

            # Add the Protenix position embedding (if enabled).
            if self.protenix_pos_embed is not None:
                # Pull the position info from input_feature_dict.
                residue_index = protenix_input_feature_dict["residue_index"]  # [N_token]
                asym_id = protenix_input_feature_dict["asym_id"]  # [N_token]

                # Make sure they live on the right device.
                if not isinstance(residue_index, torch.Tensor):
                    residue_index = torch.tensor(residue_index, device=projected_protein.device)
                if not isinstance(asym_id, torch.Tensor):
                    asym_id = torch.tensor(asym_id, device=projected_protein.device)

                pos_embed = self.protenix_pos_embed(residue_index, asym_id)
                projected_protein = projected_protein + pos_embed.to(projected_protein.dtype)

            # Locate the <protein> token positions and fill them in.
            protein_token_mask = input_ids == self.config.protein_token_id  # [B, L]
            expected_count = protein_token_mask.sum().item()
            actual_count = projected_protein.shape[0]

            if expected_count != actual_count:
                raise ValueError(
                    f"protein_token count mismatch: expected {expected_count} in input_ids, "
                    f"got {actual_count} from Protenix encoder"
                )

            # Write the protein embedding into text_embeds.
            text_embeds[protein_token_mask] = projected_protein.to(text_embeds.dtype)

        else:
            raise RuntimeError(
                "No embedding source available. Either provide precomputed embeddings "
                "(protenix_s_embedding/protenix_esm_embedding) or load ProtenixEncoder."
            )

        return text_embeds

    def forward(
        self,
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
        # Protenix position info - precomputed mode (used by protenix_pos_embed)
        protenix_residue_index: Optional[torch.Tensor] = None,
        protenix_asym_id: Optional[torch.Tensor] = None,
        # CDR mask - precomputed mode (used by cdr_mask_embedding)
        protenix_cdr_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass.

        Supports two modes:
        1. On-the-fly mode: pass protenix_input_feature_dict; ProtenixEncoder runs at call time.
        2. Precomputed mode: pass protenix_s_embedding/protenix_esm_embedding/protenix_a_token; the Encoder is skipped.

        Args:
            input_ids: [B, L] text token ids
            attention_mask: [B, L]
            position_ids: [B, L] compressed position IDs (optional)
                Uses the Bagel-style compression: the protein region (including <protein_start>,
                <protein>, <protein_end>) shares a single position_id; this saves position space
                and lets RoPE treat the protein as a single unit.
                Example:
                    input_ids: [<protein_start>, prot, prot, prot, <protein_end>, text, text]
                    position:  [0,               0,    0,    0,    0,             1,    2   ]
                When not provided, Qwen3 computes the standard incrementing position_ids from
                attention_mask automatically.
            labels: [B, L] for computing loss
            protenix_input_feature_dict: Protenix feature dict (on-the-fly mode)
            protenix_atom_array: biotite AtomArray (on-the-fly mode, optional)
            protenix_token_array: TokenArray (on-the-fly mode, optional)
            protenix_s_embedding: [B, N_max, 384] precomputed s embedding (precomputed mode)
            protenix_esm_embedding: [B, N_max, 2560] precomputed ESM embedding (precomputed mode)
            protenix_a_token: [B, N_max, 768] precomputed a_token embedding (precomputed mode)
            protenix_embedding_attention_mask: [B, N_max] embedding attention mask (precomputed mode)
            protenix_residue_index: [B, N_max] residue index within the chain (precomputed mode; for protenix_pos_embed)
            protenix_asym_id: [B, N_max] chain ID (precomputed mode; for protenix_pos_embed)
            protenix_cdr_mask: [B, N_max] CDR mask (precomputed mode; for cdr_mask_embedding)

        Returns:
            CausalLMOutputWithPast
        """
        # Prepare inputs_embeds (merge protein embedding).
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
            protenix_cdr_mask=protenix_cdr_mask,
        )

        # Work around the Flash Attention packed-sequence misdetection.
        # When batch_size=1 with compressed position_ids, transformers' _is_packed_sequence()
        # treats the non-monotonic position_ids as packed sequences, producing an incorrect
        # attention mask. Pass cu_seq_lens explicitly to denote the correct sequence boundary
        # (a single full sequence). Only needed at the first forward (prompt phase); not in
        # autoregressive decoding.
        flash_attn_kwargs = {}
        if position_ids is not None and inputs_embeds.shape[0] == 1 and self._is_first_forward(past_key_values):
            seq_len = inputs_embeds.shape[1]
            flash_attn_kwargs = {
                "cu_seq_lens_q": torch.tensor([0, seq_len], dtype=torch.int32, device=inputs_embeds.device),
                "cu_seq_lens_k": torch.tensor([0, seq_len], dtype=torch.int32, device=inputs_embeds.device),
                "max_length_q": seq_len,
                "max_length_k": seq_len,
            }

        # LLM forward.
        # Note: position_ids and cache_position pass through fine because ProteoR1Understand
        # uses the placeholder-expansion mechanism, so input_ids and inputs_embeds have equal length.
        # When position_ids=None Qwen3 derives positions from cache_position automatically.
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **flash_attn_kwargs,
            **kwargs,  # output_attentions, output_hidden_states, etc. pass through kwargs
        )

        return out

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Optional[int] = None,
        # On-the-fly mode
        protenix_input_feature_dict: Optional[dict] = None,
        protenix_atom_array: Optional[Any] = None,
        protenix_token_array: Optional[Any] = None,
        # Precomputed mode
        protenix_s_embedding: Optional[torch.Tensor] = None,
        protenix_esm_embedding: Optional[torch.Tensor] = None,
        protenix_a_token: Optional[torch.Tensor] = None,
        protenix_embedding_attention_mask: Optional[torch.Tensor] = None,
        # Protenix position info - precomputed mode (used by protenix_pos_embed)
        protenix_residue_index: Optional[torch.Tensor] = None,
        protenix_asym_id: Optional[torch.Tensor] = None,
        # CDR mask - precomputed mode (used by cdr_mask_embedding)
        protenix_cdr_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Prepare inputs for generation.

        Fix incrementing of compressed position_ids during autoregressive generation:
        - Transformers by default slices position_ids[:, -1:] and does not increment.
        - This method records an offset at the prompt stage and computes the correct
          position_ids during generation.

        Example:
            Prompt: position_ids=[0,0,0,0,0,1,2], cache_position=[0,1,2,3,4,5,6]
            offset = 6 - 2 = 4
            Generate token #1: cache_position=[7], position_ids = 7 - 4 = [3]
            Generate token #2: cache_position=[8], position_ids = 8 - 4 = [4]
        """
        position_ids = kwargs.get("position_ids", None)
        is_first = self._is_first_forward(past_key_values)

        # Prompt stage: compute and remember the per-sample offset.
        if is_first and position_ids is not None and cache_position is not None:
            # offset = last cache_position - last position_ids
            # Each sample may have a different protein length, so offsets differ.
            # position_ids: [B, L], cache_position: [L]
            last_cache_pos = cache_position[-1]  # scalar tensor
            last_position_ids = position_ids[:, -1]  # [B]
            self._generation_position_offsets = last_cache_pos - last_position_ids  # [B]

        # Generation stage: apply each sample's offset to correct position_ids.
        if not is_first and cache_position is not None:
            if hasattr(self, "_generation_position_offsets"):
                # cache_position: [1], offsets: [B] → corrected: [B, 1]
                corrected = cache_position.unsqueeze(0) - self._generation_position_offsets.unsqueeze(1)
                kwargs["position_ids"] = corrected

        model_inputs = self.llm.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        model_inputs.update(
            # On-the-fly mode
            protenix_input_feature_dict=protenix_input_feature_dict,
            protenix_atom_array=protenix_atom_array,
            protenix_token_array=protenix_token_array,
            # Precomputed mode
            protenix_s_embedding=protenix_s_embedding,
            protenix_esm_embedding=protenix_esm_embedding,
            protenix_a_token=protenix_a_token,
            protenix_embedding_attention_mask=protenix_embedding_attention_mask,
            protenix_residue_index=protenix_residue_index,
            protenix_asym_id=protenix_asym_id,
            protenix_cdr_mask=protenix_cdr_mask,
        )

        return model_inputs


def test():
    from proteor1.understand.data_collator import (
        move_protenix_features_to_device,
    )

    protenix_path = "pretrained/protenix_mini_ism_v0.5.0"
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Build the processor.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    processor = ProteoR1UnderstandProcessor(
        tokenizer=tokenizer,
        protenix_encoder_path=protenix_path,
    )

    # Build the config.
    config = ProteoR1UnderstandConfig(
        base_model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
        protenix_encoder_path=None,
        use_protenix_pos_embed=False,
        protein_token_id=processor.protein_token_id,
        protein_start_token_id=processor.protein_start_token_id,
        protein_end_token_id=processor.protein_end_token_id,
        triangle_by_torch=True,
        embedding_mode="concat",
    )

    # Build the model.
    print("Creating ProteoR1UnderstandModel model...")
    model = ProteoR1UnderstandModel(config)
    model.load_protenix_weights(protenix_path)
    model.resize_token_embeddings(len(processor.tokenizer))
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Smoke-test data.
    test_prot_json = {
        "name": "test_protein",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "MVRLFYNPIKYLFYRRSCKKRLRKALKKLNFYHPPKECCQIYRLLENAPGGTYFITENMTNELIMIAKDPVDKKIKSVKLYLTGNYIKINQHYYINIYMYLMRYNQIYKYPLICFSKYSKIL",
                    "count": 1,
                }
            }
        ],
    }

    # Process input.
    inputs = processor(
        text="<protein> Analyze this protein: ",
        protein_json=test_prot_json,
    )
    print(f"\nProcessor output:")
    print(f"  input_ids shape: {inputs.input_ids.shape}")

    # Move the model to GPU.
    # to_device pattern: protenix_encoder stays float32 while LLM and projector use bfloat16.
    model.to(device=device, dtype=dtype)

    # Build the batch (move every tensor to GPU).
    # Note: protenix_input_feature_dict keeps its dtype; Protenix internals use float32.
    batch = {
        "input_ids": inputs.input_ids.to(device),
        "attention_mask": inputs.attention_mask.to(device),
        "protenix_input_feature_dict": move_protenix_features_to_device(
            inputs.get("protenix_input_feature_dict"), device, dtype=None
        ),
        "protenix_atom_array": inputs.get("protenix_atom_array"),
        "protenix_token_array": inputs.get("protenix_token_array"),
    }

    # Forward pass (no labels, no loss).
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(**batch)
    print(f"  logits shape: {outputs.logits.shape}")

    # # Test save and reload.
    # test_dir = "temp/test_protenix_qwen_model"
    # shutil.rmtree(test_dir, ignore_errors=True)
    # model.save_pretrained(test_dir)
    # print(f"\nSaved model to {test_dir}")
    #
    # # Reload.
    # loaded_config = ProteoR1UnderstandConfig.from_pretrained(test_dir)
    # loaded_model = ProteoR1UnderstandModel.from_pretrained(test_dir, config=loaded_config)
    # loaded_model.load_protenix_weights(protenix_path)
    # loaded_model.to(device=device, dtype=dtype)
    # print(f"Loaded model from {test_dir}")

    # # Smoke-test forward on loaded_model.
    # print("\nTesting loaded model forward...")
    # loaded_model.eval()
    # with torch.no_grad():
    #     loaded_outputs = loaded_model(**batch)
    # print(f"  loaded_model logits shape: {loaded_outputs.logits.shape}")

    # Smoke-test generation.
    print("\nTesting generation...")
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            **batch,
            max_new_tokens=50,
            do_sample=False,
        )
    response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated: {response[:200]}...")

    # Cleanup.
    # shutil.rmtree(test_dir, ignore_errors=True)
    # print("\nTest passed!")


if __name__ == "__main__":
    test()
