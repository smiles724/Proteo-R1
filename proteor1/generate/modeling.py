"""
ProteoR1GenerateModel - HuggingFace PreTrainedModel wrapper for Boltz1.

Design based on ADR-8 (Config Simplification) and ADR-1 (PreTrainedModel Shell):
- Dual-mode loading: load_pretrained controls initialization source
- Mode A (Continue Training): Boltz1.from_pretrained(ckpt_path)
- Mode B (HF Checkpoint Resume): Boltz1(**config.boltz_hparams)
- After initialization, boltz_hparams is saved to config for future restores

Key Design Decisions (from ADR):
- ADR-1: Model uses PreTrainedModel shell wrapping Boltz1
- ADR-4: Full fp32 precision via _keep_in_fp32_modules
- ADR-5: torch.compile disabled by default
- ADR-7: Dual-track distributed strategy (DDP priority, FSDP2 exclude optional)
"""

import copy
import gc
import os
import random
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor
from transformers import PreTrainedModel, GenerationMixin, AutoModelForCausalLM, AutoConfig
from transformers.utils import logging

from proteor1.generate.configuration import (
    ProteoR1GenerateConfig,
    DEFAULT_C_S_INPUTS,
    ATOM_ENCODER_DIM,
)
from proteor1.generate.data_load import const
from proteor1.generate.loss.distogram import distogram_loss
from proteor1.generate.loss.validation import factored_token_lddt_dist_loss, factored_lddt_loss
from proteor1.generate.modeling_boltz import Boltz1
from proteor1.generate.modules.conditioner import (
    build_conditioner,
    CDRAlignedConditioner,
)

logger = logging.get_logger(__name__)


# Flash attention availability check
try:
    import flash_attn
    _FLASH_ATTN_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _FLASH_ATTN_AVAILABLE = False


def _to_json_serializable(obj: Any) -> Any:
    """
    Recursively convert DictConfig/ListConfig to plain Python types for JSON serialization.

    This is needed because boltz_hparams may contain omegaconf.DictConfig objects
    from the upstream checkpoint, which are not JSON serializable.
    """
    # Handle DictConfig from omegaconf
    try:
        from omegaconf import DictConfig, ListConfig
        if isinstance(obj, DictConfig):
            return {k: _to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, ListConfig):
            return [_to_json_serializable(v) for v in obj]
    except ImportError:
        pass

    # Handle regular dicts and lists
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]

    return obj


def _to_dot_accessible(obj: Any) -> Any:
    """
    Recursively convert plain dicts to DictConfig for dot notation access.

    This is needed because Boltz1.__init__ expects DictConfig objects for
    training_args/validation_args (uses dot notation like `self.training_args.diffusion_multiplicity`),
    but after JSON serialization and deserialization, they become plain dicts.
    """
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf
        if isinstance(obj, (DictConfig, ListConfig)):
            return obj
        if isinstance(obj, dict):
            return OmegaConf.create({k: _to_dot_accessible(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return OmegaConf.create([_to_dot_accessible(v) for v in obj])
    except ImportError:
        # Fallback to AttributeDict if omegaconf not available
        from proteor1.generate.utils import AttributeDict
        if isinstance(obj, dict):
            return AttributeDict({k: _to_dot_accessible(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [_to_dot_accessible(v) for v in obj]

    return obj


class ProteoR1GenerateModel(PreTrainedModel, GenerationMixin):
    """
    HuggingFace PreTrainedModel wrapper for Boltz1.

    Supports dual-mode loading:

    Mode A (Continue Training from upstream checkpoint):
        config = ProteoR1GenerateConfig(
            boltz_ckpt_path="/path/to/upstream.ckpt",
            load_pretrained=True,
        )
        model = ProteoR1GenerateModel(config)
        # -> Boltz1.from_pretrained() loads model + hparams
        # -> config.boltz_hparams populated from boltz.hparams

    Mode B (Resume from HuggingFace checkpoint):
        # After save_pretrained, config.json contains boltz_hparams
        model = ProteoR1GenerateModel.from_pretrained("/path/to/hf_checkpoint")
        # -> load_pretrained=False in saved config
        # -> Boltz1(**config.boltz_hparams) initializes from saved params
        # -> Weights loaded from model.safetensors

    Attributes:
        config_class: ProteoR1GenerateConfig
        base_model_prefix: "proteor1_generate"
        _keep_in_fp32_modules: ["boltz"] - Ensures fp32 precision
        _exclude_from_fsdp: ["boltz"]
    """

    config_class = ProteoR1GenerateConfig
    base_model_prefix = "proteor1_generate"
    supports_gradient_checkpointing = True

    # ===== FSDP2 Configuration =====
    # _no_split_modules: Modules that should not be split across FSDP shards
    # Empty because Boltz1 will be excluded from FSDP entirely in exclude mode
    _no_split_modules = []

    # _exclude_from_fsdp: Modules to exclude from FSDP2 sharding
    # When FSDP2 exclude mode is implemented, these modules use DDP-style
    # gradient synchronization instead of FSDP sharding
    _exclude_from_fsdp = ["boltz"]

    # ===== Precision Control =====
    # Boltz1 requires full fp32 precision for numerical stability
    # (SVD, Attention, LayerNorm operations are sensitive to precision)
    # text_conditioner is also kept in fp32 to match Boltz1's expected dtype
    # NOTE: text_conditioner is added dynamically in __init__ if it exists,
    # because when conditioning_method="none", text_conditioner is None and
    # HuggingFace will raise ValueError if it's listed but doesn't exist.
    _keep_in_fp32_modules = ["boltz"]
    _keep_in_fp32_modules_strict = ["boltz"]

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    def __init__(self, config: ProteoR1GenerateConfig):
        super().__init__(config)
        self.config = config

        # Determine attention implementation
        attn_implementation = "flash_attention_2" if _FLASH_ATTN_AVAILABLE else "sdpa"
        if attn_implementation == "sdpa":
            logger.warning(f"flash_attention_2 is not available for {self.__class__.__name__}")

        # ============ Understanding Model (Qwen3) ============
        if getattr(config, "skip_understanding_model_load", False):
            self._init_skipped_understanding_model()
        else:
            self._init_understanding_model(attn_implementation)
            self._setup_understanding_model()

        # Initialize Boltz1 based on loading mode
        if config.load_pretrained and config.boltz_ckpt_path:
            # Mode A: Load from upstream checkpoint
            self._init_from_pretrained(config)
        elif config.boltz_hparams:
            # Mode B: Initialize from boltz_hparams (HF checkpoint resume)
            self._init_from_hparams(config)
        else:
            raise ValueError(
                "ProteoR1GenerateModel requires either:\n"
                "  - Mode A: load_pretrained=True with boltz_ckpt_path\n"
                "  - Mode B: load_pretrained=False with boltz_hparams\n"
                f"Got: load_pretrained={config.load_pretrained}, "
                f"boltz_ckpt_path={config.boltz_ckpt_path}, "
                f"boltz_hparams={'set' if config.boltz_hparams else None}"
            )

        # Save boltz hparams to config (for save_pretrained)
        # Use _to_json_serializable to convert DictConfig to plain dicts
        self.config.boltz_hparams = _to_json_serializable(dict(self.boltz.hparams))

        self.mode = config.training_mode

        # ============ Text Conditioning Injection (CDRAlignedConditioner) ============
        self._init_conditioner()

        # Prevent re-loading on future from_pretrained calls
        # After this point, the model should use Mode B for restoration
        self.config.load_pretrained = False
        self.config.boltz_ckpt_path = None

        # Mark all Boltz1 modules as already initialized to prevent
        # post_init from reinitializing them with random weights.
        # This is crucial when loading from checkpoint.
        for module in self.boltz.modules():
            object.__setattr__(module, "_is_hf_initialized", True)

        self.post_init()

    def _init_understanding_model(self, attn_implementation: str):
        """
        Initialize Qwen3 understanding model.

        Supports two modes:
        - load_pretrained=True: Load from HuggingFace pretrained weights
        - load_pretrained=False: Initialize from config only (for loading from checkpoint)
        """
        model_id = self.config.understanding_model_id
        logger.info(f"Initializing understanding model: {model_id}")

        # Determine dtype
        dtype_str = getattr(self.config, "understanding_dtype", "bfloat16")
        torch_dtype = self.dtype_map.get(dtype_str, torch.bfloat16)

        load_pretrained = getattr(self.config, "load_pretrained", True)

        if load_pretrained:
            # Load from pretrained weights
            self.understanding_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation=attn_implementation,
            )
            logger.info(f"Loaded understanding model from pretrained: {model_id}")
        else:
            # Initialize from config only (for checkpoint loading)
            # Use no_init_weights to skip random initialization - weights will be loaded from checkpoint
            und_config = AutoConfig.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            # NOTE: We don't use no_init_weights() here because:
            # 1. It creates meta tensors that cause issues with .to() later
            # 2. LDM module has buffers that need real tensor initialization
            # 3. The overhead of random init for Qwen3-4B is acceptable
            # Weights will be overwritten by from_pretrained's checkpoint loading anyway
            self.understanding_model = AutoModelForCausalLM.from_config(
                und_config,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
            )
            logger.info(f"Initialized understanding model from config: {model_id}")

        # Store key config values
        self.hidden_size = self.understanding_model.config.hidden_size
        self.vocab_size = self.understanding_model.config.vocab_size
        self.num_hidden_layers = self.understanding_model.config.num_hidden_layers

        # Sync to our config
        self.config.hidden_size = self.hidden_size
        self.config.vocab_size = self.vocab_size
        self.config.num_hidden_layers = self.num_hidden_layers

        # Sync RoPE config from understanding model to ensure EPTMoT uses identical RoPE
        # If config already specifies these values, use them instead of Qwen3's defaults.
        # This allows independent tuning of LDM's RoPE parameters.
        # NOTE: Qwen3's RoPE parameters (especially rope_theta=5M, max_position_embeddings=262k)
        # may be excessively large for LDM's typical sequence lengths (think tokens + VAE tokens,
        # usually < 10k). Consider using smaller values for better position resolution.
        # und_config = self.understanding_model.config
        # if self.config.rope_theta is None and hasattr(und_config, "rope_theta"):
        #     self.config.rope_theta = und_config.rope_theta
        # if self.config.rope_scaling is None and hasattr(und_config, "rope_scaling"):
        #     self.config.rope_scaling = und_config.rope_scaling
        # if self.config.max_position_embeddings is None and hasattr(und_config, "max_position_embeddings"):
        #     self.config.max_position_embeddings = und_config.max_position_embeddings
        # logger.info(
        #     f"RoPE config for EPTMoT: "
        #     f"rope_theta={self.config.rope_theta}, "
        #     f"rope_scaling={self.config.rope_scaling}, "
        #     f"max_position_embeddings={self.config.max_position_embeddings}"
        # )

        logger.info(f"Understanding model: hidden_size={self.hidden_size}, "
                   f"layers={self.num_hidden_layers}, dtype={torch_dtype}")

    def _init_skipped_understanding_model(self):
        """Initialize metadata needed when Qwen3 loading is intentionally skipped."""
        required_fields = ("hidden_size", "vocab_size", "num_hidden_layers")
        missing = [field for field in required_fields if getattr(self.config, field, None) is None]
        if missing:
            raise ValueError(
                "skip_understanding_model_load=True requires config fields: " + ", ".join(missing)
            )

        self.understanding_model = None
        self.hidden_size = self.config.hidden_size
        self.vocab_size = self.config.vocab_size
        self.num_hidden_layers = self.config.num_hidden_layers
        logger.info(
            "Skipped understanding model load: "
            f"hidden_size={self.hidden_size}, layers={self.num_hidden_layers}"
        )

    def _setup_understanding_model(self):
        """Setup understanding model: trainability, generation config."""
        if self.understanding_model is None:
            logger.info("Skipping understanding model setup because it was not loaded")
            return

        # Set trainability
        trainable = self.config.is_understanding_trainable()
        self.set_understanding_trainable(trainable)

        # Disable use_cache during training (incompatible with gradient checkpointing)
        if hasattr(self.understanding_model.config, "use_cache"):
            self.understanding_model.config.use_cache = False

        # Sync generation config from base model
        self._sync_generation_config()

        logger.info(f"Understanding model setup complete: trainable={trainable}")

    def _sync_generation_config(self):
        """Sync generation config from understanding model."""
        if self.understanding_model is None:
            return

        # Try to load from base model
        if hasattr(self.config, "understanding_model_id"):
            try:
                from transformers import GenerationConfig
                base_gen_config = GenerationConfig.from_pretrained(
                    self.config.understanding_model_id,
                    # token=getattr(self.config, "hf_token", None),
                )
                self.generation_config = base_gen_config
                logger.info(f"Loaded generation_config from: {self.config.understanding_model_id}")
            except Exception as e:
                logger.warning(f"Could not load generation_config: {e}")
                if hasattr(self.understanding_model, "generation_config"):
                    self.generation_config = self.understanding_model.generation_config

        # Sync token ids
        und_config = self.understanding_model.config
        if hasattr(und_config, "eos_token_id"):
            self.config.eos_token_id = und_config.eos_token_id
        if hasattr(und_config, "pad_token_id"):
            self.config.pad_token_id = und_config.pad_token_id
        if hasattr(und_config, "bos_token_id"):
            self.config.bos_token_id = und_config.bos_token_id

    def set_understanding_trainable(self, trainable: bool):
        """Set understanding model trainability."""
        if self.understanding_model is None:
            logger.info("Skipping understanding model trainability update because it was not loaded")
            return

        self.understanding_model.requires_grad_(trainable)
        if trainable:
            self.understanding_model.train()
        else:
            self.understanding_model.eval()
        logger.info(f"Understanding model: {'TRAINABLE' if trainable else 'FROZEN'}")

    def _init_conditioner(self):
        """
        Initialize text conditioning injection module.

        This creates a CDRAlignedConditioner if conditioning_method is "add_embed",
        or sets self.text_conditioner to None if conditioning_method is "none".

        The conditioner projects text_conditioning from the Understanding Model
        to 384 dimensions (ATOM_ENCODER_DIM) and aligns to CDR positions.
        The output is then added to s_inputs[:, :, :384] in Boltz's forward pass,
        preserving the discrete features in s_inputs[:, :, 384:].
        """
        conditioning_method = getattr(self.config, "conditioning_method", "none")

        if conditioning_method == "none":
            self.text_conditioner = None
            logger.info("Text conditioning: DISABLED (conditioning_method='none')")
            return

        # Get dimensions
        # c_atom_encoder: target dimension for text projection (384 = AtomAttentionEncoder output)
        c_atom_encoder = ATOM_ENCODER_DIM
        c_text = getattr(self.config, "conditioning_c_text", None)
        if c_text is None:
            # Use understanding model's hidden size
            c_text = self.hidden_size
        use_zero_init = getattr(self.config, "conditioning_use_zero_init", True)
        use_learnable_scale = getattr(self.config, "conditioning_use_learnable_scale", True)
        use_layernorm = getattr(self.config, "conditioning_use_layernorm", False)

        # Get confidence scaler config from config
        confidence_scaler_type = getattr(self.config, "confidence_scaler_type", None)
        confidence_scaler_params = getattr(self.config, "confidence_scaler_params", None)
        if confidence_scaler_type is not None and confidence_scaler_type != "none":
            logger.info(
                f"Confidence scaler: ENABLED (type='{confidence_scaler_type}', "
                f"params={confidence_scaler_params})"
            )

        # Build conditioner
        self.text_conditioner = build_conditioner(
            conditioning_method=conditioning_method,
            c_atom_encoder=c_atom_encoder,
            c_text=c_text,
            use_zero_init=use_zero_init,
            use_learnable_scale=use_learnable_scale,
            confidence_scaler_type=confidence_scaler_type,
            confidence_scaler_params=confidence_scaler_params,
            use_layernorm=use_layernorm,
        )

        if self.text_conditioner is not None:
            # Dynamically add text_conditioner to _keep_in_fp32_modules
            # This ensures HuggingFace keeps it in fp32 to match Boltz1's expected dtype
            if "text_conditioner" not in self._keep_in_fp32_modules:
                self._keep_in_fp32_modules = self._keep_in_fp32_modules + ["text_conditioner"]
            if "text_conditioner" not in self._keep_in_fp32_modules_strict:
                self._keep_in_fp32_modules_strict = self._keep_in_fp32_modules_strict + ["text_conditioner"]

            logger.info(
                f"Text conditioning: ENABLED (method='{conditioning_method}', "
                f"c_atom_encoder={c_atom_encoder}, c_text={c_text}, "
                f"zero_init={use_zero_init})"
            )

    def _init_from_pretrained(self, config: ProteoR1GenerateConfig):
        """
        Mode A: Load Boltz1 from upstream Boltz1 checkpoint.

        Uses Boltz1.from_pretrained() which:
        1. Loads hyper_parameters from checkpoint
        2. Creates Boltz1 with those hparams
        3. Loads state_dict
        4. Loads EMA state if available

        Args:
            config: ProteoR1GenerateConfig with boltz_ckpt_path set.
        """
        if not config.boltz_ckpt_path:
            raise ValueError(
                "load_pretrained=True requires boltz_ckpt_path to be set"
            )

        if not os.path.exists(config.boltz_ckpt_path):
            raise FileNotFoundError(
                f"Boltz checkpoint not found: {config.boltz_ckpt_path}"
            )

        logger.info(f"Mode A: Loading Boltz1 from checkpoint: {config.boltz_ckpt_path}")

        # Build override kwargs for Boltz1.from_pretrained
        override_kwargs = self._build_override_kwargs(config)

        # Load Boltz1 with overrides
        self.boltz = Boltz1.from_pretrained(
            config.boltz_ckpt_path,
            **override_kwargs
        )

        logger.info(f"Boltz1 loaded from {config.boltz_ckpt_path}")

    def _init_from_hparams(self, config: ProteoR1GenerateConfig):
        """
        Mode B: Initialize Boltz1 from boltz_hparams (HF checkpoint resume).

        Creates Boltz1 from saved hyperparameters. Weights will be loaded
        separately by HuggingFace's from_pretrained mechanism.

        Args:
            config: ProteoR1GenerateConfig with boltz_hparams set.
        """
        if not config.boltz_hparams:
            raise ValueError(
                "load_pretrained=False requires boltz_hparams to be set"
            )

        logger.info("Mode B: Initializing Boltz1 from boltz_hparams")

        # Create a deep copy to avoid mutating config.boltz_hparams
        # (shallow copy would share nested dicts like training_args)
        hparams = copy.deepcopy(config.boltz_hparams)

        # Convert plain dicts to DictConfig for dot notation access
        # (Boltz1.__init__ uses `self.training_args.diffusion_multiplicity` syntax)
        hparams = _to_dot_accessible(hparams)

        # Apply EMA overrides from config
        hparams["ema"] = config.ema
        hparams["ema_decay"] = config.ema_decay

        # Initialize Boltz1 from hparams
        self.boltz = Boltz1(**hparams)

        # Initialize EMA if enabled (weights will be loaded from checkpoint)
        if config.ema:
            self.boltz.on_load_checkpoint()

        logger.info("Boltz1 initialized from boltz_hparams")

    def _build_override_kwargs(self, config: ProteoR1GenerateConfig) -> Dict[str, Any]:
        """
        Build override kwargs for Boltz1.from_pretrained().

        Note: training_args and validation_args are NOT passed here.
        Boltz1.from_pretrained() uses hyper_parameters.update(**override_kwargs),
        which would completely replace training_args dict instead of merging.

        To override specific training/validation args, use:
        - config.get_effective_training_args() in Trainer (merges overrides)
        - config.get_effective_validation_args() for validation

        Args:
            config: ProteoR1GenerateConfig with override settings.

        Returns:
            Dict of kwargs to override checkpoint defaults.
        """
        overrides = {}

        # EMA settings - these are top-level keys, safe to override
        overrides["ema"] = config.ema
        overrides["ema_decay"] = config.ema_decay

        # DO NOT pass training_args/validation_args here!
        # They would completely replace the checkpoint's args instead of merging.
        # Use config.get_effective_training_args() in Trainer to get merged args.

        # Forward inference-related parameters if set via config kwargs
        # These are passed via ProteoR1GenerateConfig(**kwargs) and stored by PretrainedConfig
        inference_params = [
            "predict_args",
            "structure_prediction_training",
            "sequence_prediction_training",
            "confidence_prediction",
            "confidence_imitate_trunk",
            "structure_inpainting",
            "alpha_pae",
            "diffusion_process_args",
        ]
        for param in inference_params:
            if hasattr(config, param):
                overrides[param] = getattr(config, param)

        return overrides

    def _forward_understanding(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            precomputed_hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Forward pass through the understanding model.

        Supports two modes:
        1. Real-time computation: Uses Understanding Model to compute hidden states
        2. Precomputed mode: Directly returns precomputed hidden states

        Parameters
        ----------
        input_ids : Optional[torch.Tensor]
            Input token IDs [B, L]. Required for real-time mode.
        attention_mask : Optional[torch.Tensor]
            Attention mask [B, L].
        labels : Optional[torch.Tensor]
            Labels for SFT loss computation. Ignored in precomputed mode.
        precomputed_hidden_states : Optional[torch.Tensor]
            Precomputed hidden states [B, L, hidden_dim]. When provided, skips
            Understanding Model forward and returns these directly.

        Returns
        -------
        tuple
            (sft_loss, last_hidden_states) where sft_loss is None in precomputed mode
        """
        # Precomputed mode: skip Understanding Model
        if precomputed_hidden_states is not None:
            # Apply attention mask to zero out padding positions
            if attention_mask is not None:
                attention_mask_inv_bool = ~attention_mask.bool()
                # Clone to avoid modifying the original tensor
                precomputed_hidden_states = precomputed_hidden_states.clone()
                precomputed_hidden_states[attention_mask_inv_bool] = 0
            return None, precomputed_hidden_states

        # Real-time computation mode (original logic)
        if input_ids is None:
            raise ValueError("input_ids is required when precomputed_hidden_states is not provided")
        if self.understanding_model is None:
            raise ValueError(
                "understanding_model is not loaded; provide precomputed_hidden_states "
                "or set skip_understanding_model_load=False"
            )

        outputs = self.understanding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )

        sft_loss = outputs.loss  # None if labels is None

        # Get last hidden states: [B, L, hidden_size]
        last_hidden_states = outputs.hidden_states[-1]

        # Apply attention mask to zero out padding positions
        if attention_mask is not None:
            attention_mask_inv_bool = ~attention_mask.bool()
            last_hidden_states[attention_mask_inv_bool] = 0

        return sft_loss, last_hidden_states

    def forward(
        self,
        #
        text: Optional[Dict[str, torch.Tensor]] = None,
        boltz: Optional[Dict[str, torch.Tensor]] = None,
        #
        mode: Optional[str] = None,
        recycling_steps: Optional[int] = None,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: Optional[int] = None,
        diffusion_samples: Optional[int] = None,
        run_confidence_sequentially: bool = False,
        **kwargs,
    ) -> Dict[str, Tensor]:

        # Validate required inputs
        if text is None:
            raise ValueError("text dict is required for forward pass")
        if boltz is None:
            raise ValueError("boltz dict is required for forward pass")

        mode = mode or self.mode
        losses = {}

        # Check for precomputed hidden states
        precomputed = text.get("precomputed_hidden_states")

        sft_loss, text_conditioning = self._forward_understanding(
            input_ids=text.get("input_ids") if precomputed is None else None,
            attention_mask=text.get("attention_mask"),
            labels=text.get("labels") if precomputed is None else None,
            precomputed_hidden_states=precomputed,
        )
        # Only track SFT loss when understanding is being trained and not using precomputed
        # Note: When labels are all -100 (e.g., kv_by_gt_seq mode), sft_loss is nan
        # In precomputed mode, sft_loss is always None
        if mode in ["joint", "understanding_only"] and sft_loss is not None:
            losses["sft_loss"] = sft_loss

        # ============ Text Conditioning Injection ============
        # If text_conditioner is enabled, compute 384-dim text_conditioning tensor
        # that will be added to s_inputs[:, :, :384] in Boltz's forward pass.
        # This injects text information into the AtomAttentionEncoder output,
        # while preserving the discrete features in s_inputs[:, :, 384:].
        text_cond_tensor = None
        if self.text_conditioner is not None:
            # Apply CDR-aligned conditioning to compute [B, N_token, 384] tensor
            # text dict contains: input_ids, attention_mask, chain_type_ids, cdr_region_type_ids
            # boltz dict contains: chain_type, region_type, and other features
            text_cond_tensor = self.text_conditioner(
                text_conditioning=text_conditioning,
                text_mask=text["attention_mask"],
                chain_type_ids=text["chain_type_ids"],
                cdr_region_type_ids=text["cdr_region_type_ids"],
                boltz_chain_type=boltz["chain_type"],
                boltz_region_type=boltz["region_type"],
                target_dtype=torch.float32,  # Explicitly specify fp32 to match Boltz1
                cdr_confidence=text.get("cdr_confidence"),  # Optional confidence for dynamic scaling
            )

        if self.training:
            boltz_output_dict = self.compute_boltz_loss(
                boltz_batch=boltz,
                text_cond_tensor=text_cond_tensor,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                multiplicity_diffusion_train=multiplicity_diffusion_train,
                diffusion_samples=diffusion_samples,
                run_confidence_sequentially=run_confidence_sequentially,
            )
        else:
            boltz_output_dict = self.compute_boltz_metrics(
                boltz_batch=boltz,
                text_cond_tensor=text_cond_tensor,
                recycling_steps=recycling_steps,
                num_sampling_steps=num_sampling_steps,
                diffusion_samples=diffusion_samples,
                run_confidence_sequentially=run_confidence_sequentially,
            )

        losses.update(**boltz_output_dict)
        return losses

    def compute_boltz_loss(
            self,
            boltz_batch: Dict,
            text_cond_tensor: Optional[torch.Tensor],
            recycling_steps: Optional[int] = None,
            num_sampling_steps: Optional[int] = None,
            multiplicity_diffusion_train: Optional[int] = None,
            diffusion_samples: Optional[int] = None,
            run_confidence_sequentially: bool = False,
    ):
        # Get effective training args (with YAML overrides applied)
        training_args = self.config.get_effective_training_args()

        if recycling_steps is None:
            recycling_steps = training_args.get("recycling_steps", 3)
        recycling_steps = random.randint(0, int(recycling_steps))

        if num_sampling_steps is None:
            num_sampling_steps = training_args.get("sampling_steps", 200)
        if multiplicity_diffusion_train is None:
            multiplicity_diffusion_train = training_args.get("diffusion_multiplicity", 1)
        if diffusion_samples is None:
            diffusion_samples = training_args.get("diffusion_samples", 1)

        boltz_outputs = self.boltz(
            feats=boltz_batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            multiplicity_diffusion_train=multiplicity_diffusion_train,
            diffusion_samples=diffusion_samples,
            run_confidence_sequentially=run_confidence_sequentially,
            text_conditioning=text_cond_tensor,
        )

        # Compute losses
        if self.boltz.structure_prediction_training or self.boltz.sequence_prediction_training:
            disto_loss, _ = distogram_loss(
                boltz_outputs,
                boltz_batch,
            )
            diffusion_loss_dict = self.boltz.structure_module.compute_loss(
                boltz_batch,
                boltz_outputs,
                multiplicity=training_args.get("diffusion_multiplicity", 1),
                **self.boltz.diffusion_loss_args,
            )
        else:
            disto_loss = 0.0
            diffusion_loss_dict = {"loss": 0.0, "loss_breakdown": {}}

        confidence_loss_dict = {
            "loss": torch.tensor(0.0).to(boltz_batch["token_index"].device),
            "loss_breakdown": {},
        }

        # Aggregate losses using config loss weights (not training_args)
        # Loss weights are stored in config, not in training_args
        loss = (
                self.config.confidence_loss_weight * confidence_loss_dict["loss"]
                + self.config.diffusion_loss_weight * diffusion_loss_dict["loss"]
                + self.config.distogram_loss_weight * disto_loss
        )

        loss_breakdown = diffusion_loss_dict.get("loss_breakdown", {}) or {}
        return dict(
            boltz_loss=loss,
            boltz_distogram_loss=disto_loss,
            boltz_diffusion_loss=diffusion_loss_dict["loss"],
            **{f"boltz_{k}": v for k, v in loss_breakdown.items()}
        )

    def compute_boltz_metrics(
            self,
            boltz_batch: Dict,
            text_cond_tensor: Optional[torch.Tensor],
            recycling_steps: Optional[int] = None,
            num_sampling_steps: Optional[int] = None,
            diffusion_samples: Optional[int] = None,
            run_confidence_sequentially: bool = False,
    ):
        # Get effective validation args (with YAML overrides applied)
        validation_args = self.config.get_effective_validation_args()

        if recycling_steps is None:
            recycling_steps = validation_args.get("recycling_steps", 3)
        if num_sampling_steps is None:
            num_sampling_steps = validation_args.get("sampling_steps", 200)
        if diffusion_samples is None:
            diffusion_samples = validation_args.get("diffusion_samples", 1)

        # Ensure types for Pyright
        assert recycling_steps is not None
        assert num_sampling_steps is not None
        assert diffusion_samples is not None

        boltz_outputs = self.boltz(
            feats=boltz_batch,
            recycling_steps=recycling_steps,
            num_sampling_steps=num_sampling_steps,
            diffusion_samples=diffusion_samples,
            run_confidence_sequentially=run_confidence_sequentially,
            text_conditioning=text_cond_tensor,
        )

        try:
            # Compute distogram LDDT
            boundaries = torch.linspace(2, 22.0, 63)
            lower = torch.tensor([1.0])
            upper = torch.tensor([22.0 + 5.0])
            exp_boundaries = torch.cat((lower, boundaries, upper))
            mid_points = ((exp_boundaries[:-1] + exp_boundaries[1:]) / 2).to(
                boltz_outputs["pdistogram"]
            )

            # Compute predicted dists
            preds = boltz_outputs["pdistogram"]
            pred_softmax = torch.softmax(preds, dim=-1)
            pred_softmax = pred_softmax.argmax(dim=-1)
            pred_softmax = torch.nn.functional.one_hot(
                pred_softmax, num_classes=preds.shape[-1]
            )
            pred_dist = (pred_softmax * mid_points).sum(dim=-1)
            true_center = boltz_batch["disto_center"]
            true_dists = torch.cdist(true_center, true_center)

            # Compute lddt's
            boltz_batch["token_disto_mask"] = boltz_batch["token_disto_mask"]
            disto_lddt_dict, disto_total_dict = factored_token_lddt_dist_loss(
                feats=boltz_batch,
                true_d=true_dists,
                pred_d=pred_dist,
            )

            true_coords, rmsds, best_rmsds, true_coords_resolved_mask = (
                self.boltz.get_true_coordinates(
                    batch=boltz_batch,
                    out=boltz_outputs,
                    diffusion_samples=diffusion_samples,
                    symmetry_correction=validation_args.get("symmetry_correction", False),
                )
            )

            all_lddt_dict, all_total_dict = factored_lddt_loss(
                feats=boltz_batch,
                atom_mask=true_coords_resolved_mask,
                true_atom_coords=true_coords,
                pred_atom_coords=boltz_outputs["sample_atom_coords"],
                multiplicity=diffusion_samples,
            )
        except RuntimeError as e:  # catch out of memory exceptions
            if "out of memory" in str(e):
                print("| WARNING: ran out of memory, skipping batch")
                torch.cuda.empty_cache()
                gc.collect()
                return
            else:
                raise e

        # if the multiplicity used is > 1 then we take the best lddt of the different samples
        # AF3 combines this with the confidence based filtering
        best_lddt_dict, best_total_dict = {}, {}
        best_complex_lddt_dict, best_complex_total_dict = {}, {}
        B = true_coords.shape[0] // diffusion_samples
        if diffusion_samples > 1:
            # NOTE: we can change the way we aggregate the lddt
            complex_total = 0
            complex_lddt = 0
            for key in all_lddt_dict.keys():
                complex_lddt += all_lddt_dict[key] * all_total_dict[key]
                complex_total += all_total_dict[key]
            complex_lddt /= complex_total + 1e-7
            best_complex_idx = complex_lddt.reshape(-1, diffusion_samples).argmax(dim=1)
            for key in all_lddt_dict:
                best_idx = all_lddt_dict[key].reshape(-1, diffusion_samples).argmax(dim=1)
                best_lddt_dict[key] = all_lddt_dict[key].reshape(-1, diffusion_samples)[
                    torch.arange(B), best_idx
                ]
                best_total_dict[key] = all_total_dict[key].reshape(-1, diffusion_samples)[
                    torch.arange(B), best_idx
                ]
                best_complex_lddt_dict[key] = all_lddt_dict[key].reshape(-1, diffusion_samples)[
                    torch.arange(B), best_complex_idx
                ]
                best_complex_total_dict[key] = all_total_dict[key].reshape(
                    -1, diffusion_samples
                )[torch.arange(B), best_complex_idx]
        else:
            best_lddt_dict = all_lddt_dict
            best_total_dict = all_total_dict
            best_complex_lddt_dict = all_lddt_dict
            best_complex_total_dict = all_total_dict

        # Store metrics as dicts with "value" and "total" keys for later aggregation
        output_dict = dict(lddt={}, disto_lddt={}, complex_lddt={}, rmsd=None, best_rmsd=None)
        for m in const.out_types:
            if m == "ligand_protein":
                if torch.any(
                    boltz_batch["pocket_feature"][
                        :, :, const.pocket_contact_info["POCKET"]
                    ].bool()
                ):
                    output_dict["lddt"]["pocket_ligand_protein"] = {
                        "value": best_lddt_dict[m], "total": best_total_dict[m]
                    }
                    output_dict["disto_lddt"]["pocket_ligand_protein"] = {
                        "value": disto_lddt_dict[m], "total": disto_total_dict[m]
                    }
                    output_dict["complex_lddt"]["pocket_ligand_protein"] = {
                        "value": best_complex_lddt_dict[m], "total": best_complex_total_dict[m]
                    }
                else:
                    output_dict["lddt"]["ligand_protein"] = {
                        "value": best_lddt_dict[m], "total": best_total_dict[m]
                    }
                    output_dict["disto_lddt"]["ligand_protein"] = {
                        "value": disto_lddt_dict[m], "total": disto_total_dict[m]
                    }
                    output_dict["complex_lddt"]["ligand_protein"] = {
                        "value": best_complex_lddt_dict[m], "total": best_complex_total_dict[m]
                    }
            else:
                output_dict["lddt"][m] = {
                    "value": best_lddt_dict[m], "total": best_total_dict[m]
                }
                output_dict["disto_lddt"][m] = {
                    "value": disto_lddt_dict[m], "total": disto_total_dict[m]
                }
                output_dict["complex_lddt"][m] = {
                    "value": best_complex_lddt_dict[m], "total": best_complex_total_dict[m]
                }

        output_dict["rmsd"] = rmsds
        output_dict["best_rmsd"] = best_rmsds
        return output_dict

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[Dict[str, Any]] = None,
        save_function=torch.save,
        safe_serialization: bool = False,
        **kwargs,
    ):
        """
        Save model to directory.

        Ensures boltz_hparams is up-to-date and saves EMA state separately.

        Args:
            save_directory: Directory to save model
            is_main_process: Whether this is the main process (for distributed)
            state_dict: Optional state dict override
            save_function: Function to use for saving
            safe_serialization: Whether to use safetensors format. Default False
                because Boltz has weight-tied tensors that safetensors doesn't handle well.
            **kwargs: Additional arguments
        """
        # Ensure config contains latest hparams
        if hasattr(self.boltz, "hparams"):
            self.config.boltz_hparams = _to_json_serializable(dict(self.boltz.hparams))

        # Call parent save
        super().save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            safe_serialization=safe_serialization,
            **kwargs,
        )

        # Save EMA state separately if enabled
        if is_main_process and self.config.ema and self.boltz.ema is not None:
            ema_path = os.path.join(save_directory, "ema.pt")
            ema_checkpoint = {}
            self.boltz.on_save_checkpoint(ema_checkpoint)
            save_function(ema_checkpoint.get("ema", {}), ema_path)
            logger.info(f"Saved EMA state to: {ema_path}")

    # ===== EMA Lifecycle Methods =====
    # These methods delegate to Boltz1's EMA management
    # Called by BoltzTrainer at appropriate training lifecycle points

    def on_train_start(self):
        """Initialize EMA at training start (called by BoltzTrainer)."""
        self.boltz.on_train_start()

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int):
        """Update EMA after optimizer step (called by BoltzTrainer)."""
        self.boltz.on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_start(self):
        """Restore EMA at epoch start (called by BoltzTrainer)."""
        self.boltz.on_train_epoch_start()

    def prepare_eval(self):
        """Switch to EMA parameters for evaluation."""
        self.boltz.prepare_eval()

    def restore_train_params(self):
        """Restore training parameters after evaluation."""
        if self.config.ema and self.boltz.ema is not None:
            self.boltz.ema.restore(self.boltz.parameters())

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """Save EMA state to checkpoint dict."""
        self.boltz.on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self):
        """Initialize EMA for loading (called before loading state dict)."""
        self.boltz.on_load_checkpoint()

    # ===== Gradient Checkpointing =====

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for memory-efficient training.

        This enables gradient checkpointing on the understanding model (e.g., Qwen3).
        Boltz1 has its own gradient checkpointing controlled via its config.

        Args:
            gradient_checkpointing_kwargs: Optional kwargs passed to understanding model's
                gradient_checkpointing_enable method.
        """
        self.gradient_checkpointing = True

        # Enable gradient checkpointing on understanding model
        if hasattr(self, 'understanding_model') and self.understanding_model is not None:
            if hasattr(self.understanding_model, 'gradient_checkpointing_enable'):
                self.understanding_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
                )
                logger.info("Gradient checkpointing enabled for understanding model")
            else:
                logger.warning(
                    "Understanding model does not support gradient_checkpointing_enable"
                )
        else:
            logger.warning("No understanding model found for gradient checkpointing")

    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing.

        This disables gradient checkpointing on the understanding model (e.g., Qwen3).
        """
        self.gradient_checkpointing = False

        # Disable gradient checkpointing on understanding model
        if hasattr(self, 'understanding_model') and self.understanding_model is not None:
            if hasattr(self.understanding_model, 'gradient_checkpointing_disable'):
                self.understanding_model.gradient_checkpointing_disable()
                logger.info("Gradient checkpointing disabled for understanding model")
            else:
                logger.warning(
                    "Understanding model does not support gradient_checkpointing_disable"
                )
        else:
            logger.warning("No understanding model found for gradient checkpointing")

    # ===== Device Property =====

    @property
    def device(self) -> torch.device:
        """Return device of model parameters."""
        return next(self.boltz.parameters()).device

    # ===== Prediction Helpers =====

    @torch.no_grad()
    def predict(
        self,
        feats: Optional[Dict[str, Tensor]] = None,
        text: Optional[Dict[str, Tensor]] = None,
        boltz: Optional[Dict[str, Tensor]] = None,
        recycling_steps: Optional[int] = None,
        sampling_steps: Optional[int] = None,
        diffusion_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run prediction on input features with optional text conditioning.

        This method computes text conditioning from the understanding model and
        delegates to Boltz1.predict() for structure prediction and confidence scoring.

        Supports two input formats for backward compatibility:
        1. Legacy: predict(feats={...}) - Uses feats as boltz features, no text conditioning
        2. New: predict(text={...}, boltz={...}) - Uses text and boltz dicts separately

        Args:
            feats: Legacy feature dictionary (alias for boltz, for backward compatibility)
            text: Optional dict with text inputs (input_ids, attention_mask, chain_type_ids, cdr_region_type_ids)
            boltz: Feature dictionary from BoltzFeaturizer
            recycling_steps: Number of recycling iterations. If None, uses predict_args default.
            sampling_steps: Number of diffusion sampling steps. If None, uses predict_args default.
            diffusion_samples: Number of structure samples to generate. If None, uses predict_args default.

        Returns:
            Prediction dictionary containing coordinates and confidence scores
        """
        # Handle backward compatibility: feats is alias for boltz
        if boltz is None and feats is not None:
            boltz = feats
        if boltz is None:
            raise ValueError("Either 'boltz' or 'feats' must be provided")

        # Compute text conditioning if text inputs are provided
        text_cond_tensor = None
        if text is not None and self.text_conditioner is not None:
            # Check for precomputed hidden states (precomputed mode)
            precomputed = text.get("precomputed_hidden_states")
            input_ids = text.get("input_ids")
            attention_mask = text.get("attention_mask")

            # Determine if we have valid inputs for either mode
            has_precomputed = precomputed is not None and precomputed.numel() > 0
            has_input_ids = input_ids is not None and input_ids.numel() > 0

            if (has_precomputed or has_input_ids) and attention_mask is not None:
                # Get text conditioning via _forward_understanding
                # (handles both precomputed and real-time modes)
                _, text_conditioning = self._forward_understanding(
                    input_ids=input_ids if not has_precomputed else None,
                    attention_mask=attention_mask,
                    labels=None,  # No labels for inference
                    precomputed_hidden_states=precomputed,
                )

                # Apply CDR-aligned conditioning
                text_cond_tensor = self.text_conditioner(
                    text_conditioning=text_conditioning,
                    text_mask=text.get("attention_mask"),
                    chain_type_ids=text.get("chain_type_ids"),
                    cdr_region_type_ids=text.get("cdr_region_type_ids"),
                    boltz_chain_type=boltz.get("chain_type"),
                    boltz_region_type=boltz.get("region_type"),
                    target_dtype=torch.float32,
                    cdr_confidence=text.get("cdr_confidence"),  # Optional confidence for dynamic scaling
                )

        # Delegate to Boltz1.predict() which handles forward pass and confidence scoring
        # Boltz1.predict() will use predict_args defaults if parameters are None
        return self.boltz.predict(
            batch=boltz,
            text_conditioning=text_cond_tensor,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
            diffusion_samples=diffusion_samples,
        )


def test():
    config = ProteoR1GenerateConfig(
        boltz_ckpt_path="ckpts/upstream/stage_4.ckpt",
        load_pretrained=True,
        ema=True,
    )
    model = ProteoR1GenerateModel(config)

    model.save_pretrained("temp/checkpoint")
    restored = ProteoR1GenerateModel.from_pretrained("temp/checkpoint")

    print()


if __name__ == '__main__':
    test()
