from proteor1._mapping import register_model

from .configuration import ProteoR1GenerateConfig
from .modeling import ProteoR1GenerateModel
from .modeling_boltz import Boltz1

register_model("proteor1_generate", ProteoR1GenerateConfig, ProteoR1GenerateModel, "general")

__all__ = [
    "ProteoR1GenerateConfig",
    "ProteoR1GenerateModel",
    "Boltz1",
]
