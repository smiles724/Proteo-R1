"""
Custom vLLM Model for PLLM (Protein Language Model)

This integrates the full PLLM model (with protein encoders) into vLLM for efficient inference.

Architecture:
    Protein Sequence → Protein Encoder → Prefix Embeddings → LLM → Output
    Structure Sequence → Structure Encoder ↗

Based on vLLM's multimodal model integration guide:
https://docs.vllm.ai/en/stable/contributing/model/multimodal.html
"""

import os
import sys
from typing import Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm.attention import AttentionMetadata
from vllm.config import CacheConfig, MultiModalConfig
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors

# Add parent directory to path to import PLLM modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proteinLLM_pllm import PLLM, PrefixProjector
import protein_encoder as protein_encoder_mod
import structure_encoder as structure_encoder_mod


class PLLMConfig(PretrainedConfig):
    """Configuration for PLLM model."""
    
    model_type = "pllm"
    
    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        protein_config: str = None,
        structure_config: str = None,
        hidden_size: int = 896,
        prefix_len: int = 4,
        proj_hid: int = 1024,
        single_token_prefix: bool = False,
        train_encoders: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path
        self.protein_config = protein_config
        self.structure_config = structure_config
        self.hidden_size = hidden_size
        self.prefix_len = prefix_len
        self.proj_hid = proj_hid
        self.single_token_prefix = single_token_prefix
        self.train_encoders = train_encoders


class PLLMForCausalLM(nn.Module):
    """
    PLLM model for vLLM with protein and structure encoders.
    
    This wraps the base LLM with protein-specific components:
    - Protein encoder (ESM-2)
    - Structure encoder (Foldseek)
    - Prefix MLP for projection
    """
    
    def __init__(
        self,
        config: PLLMConfig,
        multimodal_config: MultiModalConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[dict] = None,
    ):
        super().__init__()
        self.config = config
        self.multimodal_config = multimodal_config
        
        # Load the base LLM (Qwen2.5)
        from transformers import AutoConfig
        llm_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
        self.language_model = Qwen2ForCausalLM(llm_config, cache_config, quant_config)
        
        # Initialize protein and structure encoders
        if config.protein_config:
            self.protein_encoder = protein_encoder_mod.ProteinEncoder(
                config.protein_config, 
                out_dim=config.proj_hid, 
                load_pretrained=False
            )
        else:
            self.protein_encoder = None
            
        if config.structure_config:
            self.structure_encoder = structure_encoder_mod.StructureEncoder(
                config.structure_config,
                out_dim=config.proj_hid,
                load_pretrained=False
            )
        else:
            self.structure_encoder = None
        
        # Prefix MLP to project protein embeddings to LLM hidden space
        self.prefix_len = 1 if config.single_token_prefix else config.prefix_len
        self.prefix_mlp = PrefixProjector(
            in_dim=config.proj_hid,
            mid_dim=config.proj_hid,
            out_hidden=config.hidden_size,
            dropout=0.1,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        # Set encoder training mode
        if self.protein_encoder:
            self.protein_encoder.train(config.train_encoders)
            if not config.train_encoders:
                for param in self.protein_encoder.parameters():
                    param.requires_grad = False
                    
        if self.structure_encoder:
            self.structure_encoder.train(config.train_encoders)
            if not config.train_encoders:
                for param in self.structure_encoder.parameters():
                    param.requires_grad = False
    
    def _encode_protein_prefix(
        self,
        protein_sequences: Optional[List[str]] = None,
        structure_sequences: Optional[List[str]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Encode protein and structure sequences into prefix embeddings.
        
        Args:
            protein_sequences: List of amino acid sequences
            structure_sequences: List of structure sequences (3Di tokens)
            
        Returns:
            Prefix embeddings of shape [batch_size, prefix_len, hidden_size]
        """
        if not protein_sequences and not structure_sequences:
            return None
        
        batch_size = len(protein_sequences) if protein_sequences else len(structure_sequences)
        device = next(self.parameters()).device
        
        # Encode protein sequences
        protein_embeds = None
        if protein_sequences and self.protein_encoder:
            protein_embeds = self.protein_encoder(protein_sequences)  # [B, D]
            
        # Encode structure sequences
        structure_embeds = None
        if structure_sequences and self.structure_encoder:
            structure_embeds = self.structure_encoder(structure_sequences)  # [B, D]
        
        # Combine embeddings
        if protein_embeds is not None and structure_embeds is not None:
            combined = protein_embeds + structure_embeds
        elif protein_embeds is not None:
            combined = protein_embeds
        elif structure_embeds is not None:
            combined = structure_embeds
        else:
            return None
        
        # Expand to prefix length and project
        # combined: [B, D] -> [B, prefix_len, D]
        combined = combined.unsqueeze(1).expand(-1, self.prefix_len, -1)
        
        # Project to LLM hidden size: [B, prefix_len, D] -> [B, prefix_len, H]
        prefix_embeds = self.prefix_mlp(combined)
        
        return prefix_embeds
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        protein_sequences: Optional[List[str]] = None,
        structure_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with protein encoding.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            positions: Position IDs
            kv_caches: KV caches for attention
            attn_metadata: Attention metadata
            protein_sequences: Optional protein sequences for encoding
            structure_sequences: Optional structure sequences for encoding
            
        Returns:
            Hidden states from the language model
        """
        # Encode protein prefix if provided
        prefix_embeds = None
        if protein_sequences or structure_sequences:
            prefix_embeds = self._encode_protein_prefix(
                protein_sequences=protein_sequences,
                structure_sequences=structure_sequences,
            )
        
        # If we have prefix embeddings, we need to prepend them to the input
        if prefix_embeds is not None:
            # Get input embeddings from LLM
            inputs_embeds = self.language_model.model.embed_tokens(input_ids)
            
            # Prepend protein prefix: [B, prefix_len, H] + [B, seq_len, H]
            inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)
            
            # Adjust positions to account for prefix
            batch_size, prefix_len = prefix_embeds.shape[:2]
            prefix_positions = torch.arange(
                prefix_len, 
                device=positions.device
            ).unsqueeze(0).expand(batch_size, -1)
            positions = torch.cat([prefix_positions, positions + prefix_len], dim=1)
            
            # Forward through LLM with embeddings
            hidden_states = self.language_model.model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **kwargs,
            )
        else:
            # Standard forward without protein prefix
            hidden_states = self.language_model(
                input_ids=input_ids,
                positions=positions,
                kv_caches=kv_caches,
                attn_metadata=attn_metadata,
                intermediate_tensors=intermediate_tensors,
                **kwargs,
            )
        
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        """Compute logits from hidden states."""
        return self.language_model.compute_logits(hidden_states, sampling_metadata)
    
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        """Sample from logits."""
        return self.language_model.sample(logits, sampling_metadata)
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """
        Load weights from the saved PLLM model.
        
        The weights include:
        - Base LLM weights (from ./pllm/llm/)
        - Protein encoder weights (from model.safetensors)
        - Structure encoder weights (from model.safetensors)
        - Prefix MLP weights (from model.safetensors)
        """
        params_dict = dict(self.named_parameters())
        
        for name, loaded_weight in weights:
            # Map weight names
            if name.startswith("protein_encoder."):
                param_name = name
            elif name.startswith("structure_encoder."):
                param_name = name
            elif name.startswith("prefix_mlp."):
                param_name = name
            elif name.startswith("language_model."):
                # LLM weights
                param_name = name
            elif name.startswith("model."):
                # Map to language_model
                param_name = f"language_model.{name}"
            else:
                # Try to map to language_model
                param_name = f"language_model.model.{name}"
            
            if param_name in params_dict:
                param = params_dict[param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


# Register the model configuration
from transformers import AutoConfig
AutoConfig.register("pllm", PLLMConfig)


# Input processing for protein sequences
@INPUT_REGISTRY.register_input_processor(
    lambda cfg: cfg.model_type == "pllm",
)
def pllm_input_processor(ctx: InputContext, llm_inputs: LLMInputs) -> LLMInputs:
    """
    Process inputs for PLLM model.
    
    Extracts protein and structure sequences from the prompt and passes them
    as additional inputs to the model.
    """
    # Extract protein sequences from multi_modal_data if present
    multi_modal_data = llm_inputs.get("multi_modal_data", {})
    
    # The protein sequences should be passed in multi_modal_data
    # Format: {"protein": ["MALVFV...", ...], "structure": ["ACDEF...", ...]}
    protein_sequences = multi_modal_data.get("protein", None)
    structure_sequences = multi_modal_data.get("structure", None)
    
    # Add to prompt_adapter_request or as additional kwargs
    if protein_sequences or structure_sequences:
        llm_inputs["protein_sequences"] = protein_sequences
        llm_inputs["structure_sequences"] = structure_sequences
    
    return llm_inputs


# Register multimodal input mapper
@MULTIMODAL_REGISTRY.register_input_mapper(
    lambda cfg: cfg.model_type == "pllm",
)
def pllm_input_mapper(ctx: InputContext, data: dict) -> dict:
    """
    Map protein/structure sequences to model inputs.
    """
    return {
        "protein_sequences": data.get("protein", None),
        "structure_sequences": data.get("structure", None),
    }


# Register the maximum number of tokens for protein sequences
@MULTIMODAL_REGISTRY.register_max_multimodal_tokens(
    lambda cfg: cfg.model_type == "pllm",
)
def pllm_max_multimodal_tokens(ctx: InputContext) -> int:
    """
    Return the maximum number of tokens for protein prefix.
    """
    # The prefix length is configurable, default is 4
    return ctx.model_config.hf_config.prefix_len if hasattr(ctx.model_config.hf_config, "prefix_len") else 4

