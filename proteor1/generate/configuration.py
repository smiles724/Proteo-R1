"""
Simplified ProteoR1GenerateConfig - Only training hyperparameters.

Design based on ADR-8 (Config Simplification):
- Config only contains training process hyperparameters
- Model structure parameters are loaded from checkpoint via boltz_hparams
- Dual-mode loading: load_pretrained controls initialization source

Mode A (Continue Training):
    load_pretrained=True + boltz_ckpt_path
    -> Boltz1.from_pretrained(ckpt_path) loads structure params
    -> config.boltz_hparams is populated from boltz.hparams

Mode B (HF Checkpoint Resume):
    load_pretrained=False + boltz_hparams (from saved config.json)
    -> Boltz1(**config.boltz_hparams) initializes from saved params
"""

from typing import Any, Dict, Optional

from transformers import PretrainedConfig


# Default dimension constants for conditioning
DEFAULT_C_S_INPUTS = 455  # Boltz s_inputs dimension (384 + 2*33 + 1 + 4)
DEFAULT_C_TEXT = 3584     # Qwen3-4B hidden size

# AtomAttentionEncoder output dimension (continuous representation part of s_inputs)
# s_inputs[:, :, :384] = AtomAttentionEncoder output (continuous, dense)
# s_inputs[:, :, 384:] = discrete features (atom type embeddings, bond features, etc.)
ATOM_ENCODER_DIM = 384


class ProteoR1GenerateConfig(PretrainedConfig):
    """
    Simplified Config for ProteoR1GenerateModel.

    Only stores training process hyperparameters. Model structure parameters
    are loaded from checkpoint and stored in boltz_hparams for save/restore.

    Attributes:
        boltz_ckpt_path: Path to original upstream structure-design checkpoint (Mode A).
        load_pretrained: If True, load from boltz_ckpt_path (Mode A).
                        If False, initialize from boltz_hparams (Mode B).
        skip_understanding_model_load: If True, skip loading the Qwen3
            understanding model inside the generation model. This is intended
            for precomputed-hidden-state OSS checkpoints and requires
            hidden_size, vocab_size, and num_hidden_layers in the config.
        diffusion_loss_weight: Weight for diffusion loss in total loss.
        distogram_loss_weight: Weight for distogram loss in total loss.
        confidence_loss_weight: Weight for confidence loss in total loss.
        ema: Whether to use Exponential Moving Average.
        ema_decay: EMA decay factor.
        override_training_args: Optional dict to override training params from ckpt.
        override_validation_args: Optional dict to override validation params from ckpt.
        boltz_hparams: Full hyperparameters from Boltz1 (populated after loading).
        conditioning_method: Method for injecting text conditioning into Boltz.
            - "none": No conditioning injection (original Boltz behavior)
            - "add_embed": CDR-aligned add-embed injection
        conditioning_c_s_inputs: Dimension of Boltz s_inputs (default 455).
        conditioning_c_text: Dimension of text hidden states (default 3584 for Qwen3-4B).
        conditioning_use_zero_init: Whether to zero-initialize the output projection
            for backward compatibility (default True).
        conditioning_use_learnable_scale: Whether to use a learnable scale parameter
            for text conditioning (default True). This is INDEPENDENT of confidence scaling.
        confidence_scaler_type: Type of confidence-based dynamic scaling strategy.
            - None or "none": No confidence scaling (returns all 1s)
            - "identity": scale = confidence (direct use, baseline)
            - "threshold": scale = 1 if conf >= threshold else 0 (hard cutoff)
            - "power": scale = conf^alpha (exponential penalty for low confidence)
        confidence_scaler_params: Strategy-specific parameters for confidence scaler.
            - threshold: {"threshold": 0.9}
            - power: {"alpha": 2.0}
    """

    model_type = "proteor1_generate"

    def __init__(
        self,
        # ===== Checkpoint Control =====
        understanding_model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        boltz_ckpt_path: Optional[str] = None,
        load_pretrained: bool = True,
        understanding_dtype: str = "bfloat16",
        boltz_dtype: str = "float32",
        skip_understanding_model_load: bool = False,
        hidden_size: Optional[int] = None,
        vocab_size: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,

        # ===== Loss Weights (Training Control) =====
        diffusion_loss_weight: float = 4.0,
        distogram_loss_weight: float = 0.03,
        confidence_loss_weight: float = 0.0001,

        # ===== Training Parameter Overrides (Optional, None=use ckpt default) =====
        override_training_args: Optional[Dict[str, Any]] = None,
        override_validation_args: Optional[Dict[str, Any]] = None,

        # ===== EMA Control =====
        ema: bool = True,
        ema_decay: float = 0.999,

        # ===== Boltz1 Full Hyperparameters (Populated by Model) =====
        boltz_hparams: Optional[Dict[str, Any]] = None,
        training_mode: str = "generation_only",  # "joint", "understanding_only", "generation_only"

        # ===== Text Conditioning Injection Control =====
        conditioning_method: str = "none",  # "none" or "add_embed"
        conditioning_c_s_inputs: int = DEFAULT_C_S_INPUTS,
        conditioning_c_text: Optional[int] = None,  # Will be set from understanding model if None
        conditioning_use_zero_init: bool = True,
        conditioning_use_learnable_scale: bool = True,  # Independent of confidence scaling
        conditioning_use_layernorm: bool = False,  # Apply LayerNorm to CDR embeddings before scaling

        # ===== Confidence-based Dynamic Scaling (for ablation experiments) =====
        confidence_scaler_type: Optional[str] = None,  # None, "none", "identity", "threshold", etc.
        confidence_scaler_params: Optional[Dict[str, Any]] = None,  # Strategy-specific params

        **kwargs,
    ):
        super().__init__(**kwargs)

        # Checkpoint control
        self.understanding_model_id = understanding_model_id
        self.boltz_ckpt_path = boltz_ckpt_path
        self.load_pretrained = load_pretrained
        self.understanding_dtype = understanding_dtype
        self.boltz_dtype = boltz_dtype
        self.skip_understanding_model_load = skip_understanding_model_load
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers

        # Loss weights
        self.diffusion_loss_weight = diffusion_loss_weight
        self.distogram_loss_weight = distogram_loss_weight
        self.confidence_loss_weight = confidence_loss_weight

        # Training parameter overrides
        self.override_training_args = override_training_args or {}
        self.override_validation_args = override_validation_args or {}

        # EMA control
        self.ema = ema
        self.ema_decay = ema_decay

        # Boltz1 full hyperparameters (populated after loading)
        self.boltz_hparams = boltz_hparams

        # Training
        self.training_mode = training_mode

        # Text conditioning injection
        self.conditioning_method = conditioning_method
        self.conditioning_c_s_inputs = conditioning_c_s_inputs
        self.conditioning_c_text = conditioning_c_text
        self.conditioning_use_zero_init = conditioning_use_zero_init
        self.conditioning_use_learnable_scale = conditioning_use_learnable_scale
        self.conditioning_use_layernorm = conditioning_use_layernorm

        # Confidence-based dynamic scaling (for ablation experiments)
        self.confidence_scaler_type = confidence_scaler_type
        self.confidence_scaler_params = confidence_scaler_params or {}

    def is_understanding_trainable(self) -> bool:
        """
        Whether understanding model should be trainable.

        Derived from training_mode:
        - "joint": understanding is trainable
        - "understanding_only": understanding is trainable
        - "generation_only": understanding is frozen
        """
        return self.training_mode in ["joint", "understanding_only"]

    def is_boltz_trainable(self) -> bool:
        """
        Whether LDM (generation model) should be trainable.

        Derived from training_mode:
        - "joint": LDM is trainable
        - "generation_only": LDM is trainable
        - "understanding_only": LDM is not used (but kept frozen if loaded)
        """
        return self.training_mode in ["joint", "generation_only"]

    def get_effective_training_args(self) -> Dict[str, Any]:
        """
        Get effective training arguments (ckpt default + overrides).

        Returns:
            Dict with merged training arguments.
        """
        if self.boltz_hparams is None:
            return self.override_training_args.copy()

        # Get base training args from boltz_hparams
        base_args = self.boltz_hparams.get("training_args", {})
        if base_args is None:
            base_args = {}

        # Apply overrides
        return {**base_args, **self.override_training_args}

    def get_effective_validation_args(self) -> Dict[str, Any]:
        """
        Get effective validation arguments (ckpt default + overrides).

        Returns:
            Dict with merged validation arguments.
        """
        if self.boltz_hparams is None:
            return self.override_validation_args.copy()

        # Get base validation args from boltz_hparams
        base_args = self.boltz_hparams.get("validation_args", {})
        if base_args is None:
            base_args = {}

        # Apply overrides
        return {**base_args, **self.override_validation_args}
