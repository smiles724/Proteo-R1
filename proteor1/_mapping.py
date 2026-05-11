from typing import Literal

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForMaskedLM
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig


AUTO_REGISTER_MODEL_MAPPING = {
    "causal_lm": AutoModelForCausalLM,
    "masked_lm": AutoModelForMaskedLM,
    "image_text_to_text": AutoModelForImageTextToText,
    "general": AutoModel,
}


def register_model(
    model_type: str,
    model_config: type[PretrainedConfig],
    model_class: type[PreTrainedModel],
    model_general_type: Literal["causal_lm", "masked_lm", "image_text_to_text", "general"] = "causal_lm",
) -> None:
    AutoConfig.register(model_type, model_config, exist_ok=True)
    AUTO_REGISTER_MODEL_MAPPING[model_general_type].register(model_config, model_class, exist_ok=True)
