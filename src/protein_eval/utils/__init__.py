import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from transformers import AutoTokenizer


def extract_model_size(model_id: str) -> Optional[int]:
    """
    Extract model size from model_id string.
    Looks for patterns like: 2B, 2b, 7B, 7b, 32B, 32b, 72B, 72b
    Returns the size as an integer (in billions) or None if not found.

    Examples:
        "Qwen/Qwen2-VL-2B-Instruct" -> 2
        "Qwen/Qwen2-VL-7B" -> 7
        "qwen2-vl-72b-instruct" -> 72
        "model-3b-test" -> 3
    """
    # Pattern to match: one or more digits followed by B or b
    # Can be preceded by - or _
    pattern = r'[-_]?(\d+)[Bb]'
    match = re.search(pattern, model_id)

    if match:
        return int(match.group(1))
    return None



class ModelConfig(ABC):
    default_engine_args: Dict = dict(generation_config="auto")
    model_id: str = None

    @abstractmethod
    def get_prompt_from_question(self, messages: List[Dict], **kwargs):
        raise NotImplementedError

    def update_chat_template(self, chat_template: str):
        raise NotImplementedError


class Qwen(ModelConfig):
    def __init__(self, model_id: str, max_model_len: int = None, max_tokens: int = None):
        if "qwen" not in model_id.lower():
            raise NotImplementedError("")

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.default_engine_args["model"] = model_id

        model_size = extract_model_size(model_id)

        if model_size <= 14:
            self.default_engine_args["tensor_parallel_size"] = 1
        elif model_size <= 32:
            self.default_engine_args["tensor_parallel_size"] = 2
        elif model_size <= 72:
            self.default_engine_args["tensor_parallel_size"] = 4
        else:
            self.default_engine_args["tensor_parallel_size"] = 8

        if max_tokens is not None:
            self.default_engine_args["override_generation_config"] = {"max_tokens": max_tokens}

        if max_model_len is not None:
            self.default_engine_args.update(
                max_model_len=max_model_len,
                max_num_batched_tokens=max_model_len,
            )
            if ("qwen" in model_id.lower() and ("2.5" in model_id or "25" in model_id) and "math" in model_id.lower()) and max_model_len > 4096:
                self.default_engine_args["hf_overrides"] = {
                    "max_position_embeddings": max_model_len,
                }

    def get_prompt_from_question(self, messages: List[Dict], enable_thinking: bool = False):
        kwargs = {}
        if "qwen3" in self.model_id.lower():
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **kwargs
        )


def build_model_config(model_id: str, **kwargs) -> ModelConfig:
    model_config = Qwen(model_id=model_id, **kwargs)
    engine_args = model_config.default_engine_args
    if "model" not in engine_args:
        engine_args["model"] = model_id
    return model_config

