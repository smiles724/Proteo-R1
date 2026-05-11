from transformers import AutoProcessor

from proteor1._mapping import register_model

from .configuration import ProteoR1UnderstandConfig
from .data_collator import ProteoR1UnderstandDataCollator, move_protenix_features_to_device, prepare_batch_for_model
from .modeling import PrefixProjector, ProteoR1UnderstandModel
from .processing import ProteoR1UnderstandProcessor

register_model(
    model_type="proteor1_understand",
    model_config=ProteoR1UnderstandConfig,
    model_class=ProteoR1UnderstandModel,
    model_general_type="causal_lm",
)
AutoProcessor.register(ProteoR1UnderstandConfig, ProteoR1UnderstandProcessor, exist_ok=True)

__all__ = [
    "ProteoR1UnderstandConfig",
    "ProteoR1UnderstandProcessor",
    "ProteoR1UnderstandModel",
    "PrefixProjector",
    "ProteoR1UnderstandDataCollator",
    "move_protenix_features_to_device",
    "prepare_batch_for_model",
]
