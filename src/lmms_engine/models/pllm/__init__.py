from .configuration_pllm import PLLMConfig
from .modeling_pllm import PLLM
from .processing_pllm import PLLMProcessor
from ...mapping_func import register_model


register_model(
    "pllm",
    PLLMConfig,
    PLLM,
)


__all__ = [
    "PLLMConfig",
    "PLLM",
    "PLLMProcessor",
]