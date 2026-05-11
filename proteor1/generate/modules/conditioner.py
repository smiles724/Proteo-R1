"""
CDR-Aligned Conditioner Module for Text-to-Structure Conditioning.

This module implements the CDRAlignedConditioner that injects text_conditioning
from the Understanding Model (Qwen3) into the first 384 dimensions of Boltz's s_inputs tensor.

Key Design Principles:
1. Zero-initialization ensures initial behavior matches original Boltz
2. CDR-aligned 1:1 mapping based on chain_type and region_type
3. Only modify s_inputs[:, :, :384] (AtomAttentionEncoder output), preserve [:, :, 384:] (discrete features)

Architecture Context:
- s_inputs: [B, N_token, 455]
  - [:, :, :384] = AtomAttentionEncoder continuous output -> inject text_conditioning here
  - [:, :, 384:] = discrete features (atom types, bonds, etc.) -> preserve unchanged
- This design allows text_conditioning to influence both s_init and z_init through the shared s_inputs

Data Format Expected:
- text_conditioning: [B, L_text, c_text] from Understanding Model hidden states
- s_inputs: [B, N_token, c_s_inputs] from Boltz InputEmbedder
- chain_type_ids: [B, L_text] - 1=Heavy, 2=Light
- cdr_region_type_ids: [B, L_text] - 2=CDR1, 4=CDR2, 6=CDR3
- boltz_chain_type: [B, N_token] - 1=Heavy, 2=Light, 3=Antigen
- boltz_region_type: [B, N_token] - 1-9 (2/4/6 are CDRs)
"""

import logging
import torch
from torch import nn, Tensor
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


# =====================================================
# Confidence Scaler Strategies
# =====================================================

class ConfidenceScaler(nn.Module):
    """Confidence-based dynamic scaling module.

    Computes per-position scale factors based on confidence values.
    Supported strategies: none, identity, threshold, power.

    Parameters
    ----------
    scaler_type : str
        The type of scaler to use. Options:
        - "none": No confidence scaling (returns all 1s)
        - "identity": scale = confidence (direct use, baseline)
        - "threshold": scale = 1 if conf >= threshold else 0 (hard cutoff)
        - "power": scale = conf^alpha (exponential penalty for low confidence)
    params : Dict[str, Any]
        Strategy-specific parameters:
        - threshold: {"threshold": 0.9}
        - power: {"alpha": 2.0}
    """

    def __init__(self, scaler_type: str = "none", params: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.scaler_type = scaler_type
        self.params = params or {}

    def forward(self, confidence: Tensor, dtype: torch.dtype = None) -> Tensor:
        """Compute scale factors from confidence values.

        Parameters
        ----------
        confidence : Tensor
            Confidence values [n_align], range [0, 1].
        dtype : torch.dtype, optional
            Target dtype for output.

        Returns
        -------
        Tensor
            Scale factors [n_align].
        """
        if dtype is None:
            dtype = confidence.dtype

        conf = confidence.to(dtype=dtype)

        if self.scaler_type == "none":
            # Return all ones (no scaling effect)
            return torch.ones_like(conf)

        elif self.scaler_type == "identity":
            # Direct use of confidence (baseline)
            return conf

        elif self.scaler_type == "threshold":
            # Hard threshold: 1 if conf >= threshold else 0
            threshold = self.params.get("threshold", 0.9)
            return (conf >= threshold).to(dtype=dtype)

        elif self.scaler_type == "power":
            # Exponential penalty: conf^alpha
            alpha = self.params.get("alpha", 2.0)
            return torch.pow(conf, alpha)

        else:
            raise ValueError(
                f"Unknown scaler_type: {self.scaler_type}. "
                f"Supported types: 'none', 'identity', 'threshold', 'power'"
            )


def build_confidence_scaler(
    scaler_type: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Optional[ConfidenceScaler]:
    """Factory function to build a confidence scaler.

    Parameters
    ----------
    scaler_type : Optional[str]
        The type of scaler. If None or "none", returns None.
    params : Optional[Dict[str, Any]]
        Strategy-specific parameters.

    Returns
    -------
    Optional[ConfidenceScaler]
        The scaler module, or None if scaler_type is None or "none".
    """
    if scaler_type is None or scaler_type == "none":
        return None
    return ConfidenceScaler(scaler_type=scaler_type, params=params)


# Chain type constants (matching boltz_dataset.py)
CHAIN_TYPE_HEAVY = 1
CHAIN_TYPE_LIGHT = 2

# CDR region type constants (matching boltz_dataset.py)
CDR_TYPES = [2, 4, 6]  # CDR1, CDR2, CDR3


class CDRAlignedConditioner(nn.Module):
    """
    CDR-Aligned Add-Embed Conditioner.

    Injects text_conditioning into s_inputs at CDR positions with 1:1 alignment.
    Uses chain_type_ids and cdr_region_type_ids to align text tokens to protein tokens.

    The conditioner outputs a [B, N_token, 384] tensor that is added to the first 384
    dimensions of s_inputs (the AtomAttentionEncoder continuous output), while
    preserving the remaining 71 dimensions (discrete features like atom types).

    Parameters
    ----------
    c_atom_encoder : int
        Dimension of AtomAttentionEncoder output (default 384).
        This is the target dimension for text_conditioning projection.
    c_text : int
        Dimension of text hidden states from Understanding Model (default 3584 for Qwen3-4B).
    use_zero_init : bool
        Whether to zero-initialize the output projection for backward compatibility.
        When True, the initial output is zero, ensuring original Boltz behavior.

    Forward Signature
    -----------------
    forward(
        text_conditioning,     # [B, L_text, 3584]
        text_mask,             # [B, L_text]
        chain_type_ids,        # [B, L_text]
        cdr_region_type_ids,   # [B, L_text]
        boltz_chain_type,      # [B, N_token]
        boltz_region_type,     # [B, N_token]
    ) -> Tensor  # [B, N_token, 384] - to be added to s_inputs[:, :, :384]
    """

    def __init__(
        self,
        c_atom_encoder: int = 384,
        c_text: int = 3584,
        use_zero_init: bool = True,
        use_learnable_scale: bool = True,
        confidence_scaler_type: Optional[str] = None,
        confidence_scaler_params: Optional[Dict[str, Any]] = None,
        use_layernorm: bool = False,
    ):
        super().__init__()

        self.c_atom_encoder = c_atom_encoder
        self.c_text = c_text

        # Text projection: c_text -> c_atom_encoder (384)
        # Two-layer MLP with SiLU activation
        self.proj = nn.Sequential(
            nn.Linear(c_text, c_atom_encoder * 2),
            nn.SiLU(),
            nn.Linear(c_atom_encoder * 2, c_atom_encoder),
        )

        # Zero-initialize the last layer for backward compatibility
        if use_zero_init:
            last_layer = self.proj[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.zeros_(last_layer.weight)
                if last_layer.bias is not None:
                    nn.init.zeros_(last_layer.bias)

        # Learnable scale parameter (initialized to 1.0)
        # This allows the model to learn the optimal scaling
        # Can be disabled via use_learnable_scale=False
        # Always use nn.Parameter for checkpoint compatibility, control learning via requires_grad
        self.use_learnable_scale = use_learnable_scale
        self.scale = nn.Parameter(torch.ones(1), requires_grad=use_learnable_scale)

        # Confidence scaler for dynamic scaling based on confidence values
        # This is INDEPENDENT of learnable_scale - both can be enabled/disabled separately
        # Final scale = learnable_scale * confidence_scale (if both enabled)
        self.confidence_scaler = build_confidence_scaler(confidence_scaler_type, confidence_scaler_params)
        self.use_confidence_scaling = confidence_scaler_type is not None and confidence_scaler_type != "none"

        # Optional LayerNorm applied to hidden states before projection
        # This normalizes the input text_conditioning [B, L_text, c_text] before proj
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.layernorm = nn.LayerNorm(c_text)
        else:
            self.layernorm = None

    def forward(
        self,
        text_conditioning: Tensor,
        text_mask: Tensor,
        chain_type_ids: Tensor,
        cdr_region_type_ids: Tensor,
        boltz_chain_type: Tensor,
        boltz_region_type: Tensor,
        target_dtype: Optional[torch.dtype] = None,
        cdr_confidence: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute text_conditioning tensor to be added to s_inputs[:, :, :384].

        This method projects text embeddings to the AtomAttentionEncoder dimension (384)
        and aligns them to protein CDR positions. The output should be added to the first
        384 dimensions of s_inputs by the caller.

        Parameters
        ----------
        text_conditioning : Tensor
            Understanding Model hidden states [B, L_text, c_text].
        text_mask : Tensor
            Attention mask for text [B, L_text], 1=valid, 0=padding.
        chain_type_ids : Tensor
            Chain type for each text token [B, L_text], 1=Heavy, 2=Light.
        cdr_region_type_ids : Tensor
            CDR region type for each text token [B, L_text], 2/4/6=CDR1/2/3.
        boltz_chain_type : Tensor
            Chain type for each protein token [B, N_token], 1=H, 2=L, 3=Ag.
        boltz_region_type : Tensor
            Region type for each protein token [B, N_token], 1-9.
        target_dtype : Optional[torch.dtype]
            Target dtype for output tensor. If None, uses text_conditioning.dtype.
        cdr_confidence : Optional[Tensor]
            Confidence values for each text token [B, L_text], range [0, 1].
            Required when using confidence-based dynamic scaling.

        Returns
        -------
        Tensor
            Text conditioning to add to s_inputs[:, :, :384].
            Shape: [B, N_token, 384] (c_atom_encoder).
            Non-CDR positions are zero.
        """
        B = text_conditioning.shape[0]
        N_token = boltz_chain_type.shape[1]
        device = text_conditioning.device
        dtype = target_dtype if target_dtype is not None else text_conditioning.dtype

        # Project text to atom encoder space (384 dimensions)
        # Cast text_conditioning to proj's dtype for mixed precision training
        proj_dtype = next(self.proj.parameters()).dtype
        text_conditioning = text_conditioning.to(dtype=proj_dtype)

        # Apply LayerNorm to hidden states before projection (if enabled)
        if self.use_layernorm and self.layernorm is not None:
            text_conditioning = self.layernorm(text_conditioning)

        text_proj = self.proj(text_conditioning)  # [B, L_text, c_atom_encoder]
        text_proj = text_proj.to(dtype=dtype)  # Cast to target dtype

        # Initialize output tensor with zeros (non-CDR positions remain zero)
        result = torch.zeros(B, N_token, self.c_atom_encoder, device=device, dtype=dtype)

        # Iterate over each CDR region for alignment
        for chain_type in [CHAIN_TYPE_HEAVY, CHAIN_TYPE_LIGHT]:
            for cdr_type in CDR_TYPES:
                for b in range(B):
                    # Find text positions for this CDR region
                    text_cdr_mask = (
                        (chain_type_ids[b] == chain_type) &
                        (cdr_region_type_ids[b] == cdr_type) &
                        (text_mask[b].bool())
                    )
                    text_indices = text_cdr_mask.nonzero(as_tuple=True)[0]

                    # Find protein positions for this CDR region
                    protein_cdr_mask = (
                        (boltz_chain_type[b] == chain_type) &
                        (boltz_region_type[b] == cdr_type)
                    )
                    protein_indices = protein_cdr_mask.nonzero(as_tuple=True)[0]

                    # 1:1 alignment (take minimum length due to possible cropping)
                    n_text = len(text_indices)
                    n_protein = len(protein_indices)
                    n_align = min(n_text, n_protein)

                    # Log significant length mismatch (>20% difference) at debug level
                    # to avoid excessive output during training
                    if n_align > 0 and n_text != n_protein:
                        diff_ratio = abs(n_text - n_protein) / max(n_text, n_protein)
                        if diff_ratio > 0.2:
                            logger.debug(
                                f"CDR length mismatch: chain={chain_type}, cdr={cdr_type}, "
                                f"text={n_text}, protein={n_protein}, aligning={n_align}"
                            )

                    if n_align > 0:
                        # Get projected text embeddings for aligned positions
                        text_emb = text_proj[b, text_indices[:n_align]]  # [n_align, c_atom_encoder]

                        # Compute scale factor from two INDEPENDENT components:
                        # 1. Learnable scale (self.scale) - global scalar, can be disabled
                        # 2. Confidence scale - per-position, can be disabled
                        # Final scale = learnable_scale * confidence_scale (element-wise)

                        # Start with learnable scale (always a scalar)
                        scale_factor = self.scale.to(dtype=dtype)  # scalar

                        # Multiply by confidence scale if enabled (per-position)
                        if self.use_confidence_scaling and cdr_confidence is not None and self.confidence_scaler is not None:
                            conf_values = cdr_confidence[b, text_indices[:n_align]]  # [n_align]
                            conf_scale = self.confidence_scaler(conf_values, dtype=dtype)  # [n_align]
                            # scale_factor becomes [n_align, 1] for broadcasting with text_emb
                            scale_factor = scale_factor * conf_scale.unsqueeze(-1)

                        # Place scaled text embeddings at corresponding protein positions
                        result[b, protein_indices[:n_align]] = scale_factor * text_emb

        return result


def build_conditioner(
    conditioning_method: str,
    c_atom_encoder: int = 384,
    c_text: int = 3584,
    use_zero_init: bool = True,
    use_learnable_scale: bool = True,
    confidence_scaler_type: Optional[str] = None,
    confidence_scaler_params: Optional[Dict[str, Any]] = None,
    use_layernorm: bool = False,
) -> Optional[nn.Module]:
    """
    Factory function to build a conditioner based on configuration.

    Parameters
    ----------
    conditioning_method : str
        The conditioning method to use:
        - "none": No conditioning (returns None)
        - "add_embed": CDR-aligned add-embed conditioner
    c_atom_encoder : int
        Dimension of AtomAttentionEncoder output (default 384).
        This is the target dimension for text_conditioning projection.
    c_text : int
        Dimension of text hidden states.
    use_zero_init : bool
        Whether to zero-initialize for backward compatibility.
    use_learnable_scale : bool
        Whether to use a learnable scale parameter (default True).
        This is INDEPENDENT of confidence scaling.
    confidence_scaler_type : Optional[str]
        Type of confidence-based dynamic scaling. Options:
        - None or "none": No confidence scaling
        - "identity": scale = confidence
        - "threshold": scale = 1 if conf >= threshold else 0
        - "power": scale = conf^alpha
    confidence_scaler_params : Optional[Dict[str, Any]]
        Strategy-specific parameters for confidence scaler.
        - threshold: {"threshold": 0.9}
        - power: {"alpha": 2.0}
    use_layernorm : bool
        Whether to apply LayerNorm to text hidden states before projection.
        Normalizes input text_conditioning [B, L_text, c_text] before the proj layer.
        Default False.

    Returns
    -------
    Optional[nn.Module]
        The conditioner module, or None if conditioning_method is "none".

    Raises
    ------
    ValueError
        If conditioning_method is not recognized.
    """
    if conditioning_method == "none":
        return None
    elif conditioning_method == "add_embed":
        return CDRAlignedConditioner(
            c_atom_encoder=c_atom_encoder,
            c_text=c_text,
            use_zero_init=use_zero_init,
            use_learnable_scale=use_learnable_scale,
            confidence_scaler_type=confidence_scaler_type,
            confidence_scaler_params=confidence_scaler_params,
            use_layernorm=use_layernorm,
        )
    else:
        raise ValueError(
            f"Unknown conditioning_method: {conditioning_method}. "
            f"Expected one of: 'none', 'add_embed'"
        )
