from transformers import AutoConfig, AutoModelForCausalLM

from .configuration_pllm import PLLMConfig
from .modeling_pllm import PLLM
from .protein_encoder import ProteinEncoder
from .structure_encoder import StructureEncoder

AutoConfig.register("pllm", PLLMConfig)
AutoModelForCausalLM.register(PLLMConfig, PLLM)

__all__ = [
    "PLLMConfig",
    "PLLM",
    "ProteinEncoder",
    "StructureEncoder",
]
