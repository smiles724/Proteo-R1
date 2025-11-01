from typing import List, Optional
import torch

from lmms_engine.models.pllm.processing_pllm import PLLMProcessor, PLLMProcessorKwargs

from lmms_engine.mapping_func import register_processor
from . import BaseQwen2_5_DataProcessor


@register_processor("pllm_qwen25")
class PLLMQwen2_5_DataProcessor(BaseQwen2_5_DataProcessor):
    """
    PLLM DataProcessor wrapper for Qwen2.5 models.

    Responsibilities:
    1. Build HF PLLMProcessor
    2. Process HF messages format
    3. Implement complete label masking
    """

    def _build_processor(self):
        """Build HF PLLMProcessor from pretrained path (includes all 3 tokenizers)."""
        processor = PLLMProcessor.from_pretrained(self.config.processor_name)
        return processor

    def build(self):
        super().build()
        self.processor.tokenizer.chat_template = self.chat_template_no_system

    def process(
        self,
        hf_messages,
        aa_seq: Optional[List[str]] = None,
        stru_str: Optional[List[str]] = None,
        system_message: str = "You are a helpful assistant",
        add_system_prompt=True,
        add_generation_prompt=False,  # Whether add a generation prompt at the end
        **kwargs,
    ):
        prompt_text = self.processor.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        protein_inputs = self.processor(
            prompt_text,
            aa_seq=aa_seq,
            stru_str=stru_str,
            return_tensors="pt",
        )

        # Create input_ids and labels (simplified - no expansion needed)
        # Messages already expanded by PLLMPlugin
        inputs = self.get_qwen_template_labels(
            hf_messages=hf_messages,
            system_message=system_message,
            add_system_prompt=add_system_prompt,
            add_generation_prompt=add_generation_prompt,
        )

        inputs.update(
            protein_input_ids=protein_inputs["protein_input_ids"],
            protein_attention_mask=protein_inputs["protein_attention_mask"],
            structure_input_ids=protein_inputs["structure_input_ids"],
            structure_attention_mask=protein_inputs["structure_attention_mask"],
        )

        return inputs

    def get_qwen_template_labels(
        self,
        hf_messages,
        system_message: str = "You are a helpful assistant",
        add_system_prompt: bool = True,
        add_generation_prompt: bool = False,
        **kwargs,
    ):
        special_tokens = self.processor.tokenizer.additional_special_tokens
        special_tokens.extend(["<|im_start|>", "<|im_end|>"])
        unmask_tokens_idx = [self.processor.tokenizer.convert_tokens_to_ids(t) for t in special_tokens]
        input_id, target = [], []

        if add_system_prompt and hf_messages[0]["role"] != "system":
            input_id += self.processor.tokenizer.apply_chat_template(
                [{"role": "system", "content": system_message}],
            )
            target += [-100] * len(input_id)

        for message in hf_messages:
            role = message["role"]
            # Cautions, qwen2_5 vl tokenizer wrap into a list
            encode_id = self.processor.tokenizer.apply_chat_template([message], tokenize=True)
            if isinstance(encode_id, list) and len(encode_id) > 0 and isinstance(encode_id[0], list):
                encode_id = encode_id[0]

            input_id += encode_id
            if role in ["user", "system"]:
                target += [-100] * len(encode_id)
            else:
                # Mask out the assistant prefix (first 3 tokens)
                encode_id[:3] = [-100] * 3
                target += encode_id

        if add_generation_prompt:
            generation_tokens = self.processor.tokenizer.encode("<|im_start|>assistant\n")
            input_id += generation_tokens
            target += [-100] * len(generation_tokens)

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        # Unmask special tokens + Mask multimodal tokens
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == self.seq_token_id:
                target[idx] = -100
            if encode_id == self.struct_token_id:
                target[idx] = -100

        input_id = torch.tensor(input_id, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return dict(
            input_ids=input_id,
            labels=target,
        )

    @property
    def image_token_id(self):
        """PLLM does not use image tokens."""
        return None

    @property
    def video_token_id(self):
        """PLLM does not use video tokens."""
        return None

    @property
    def audio_token_id(self):
        """PLLM does not use audio tokens."""
        return None

    @property
    def seq_token_id(self):
        """Protein sequence token ID (forwarded to inner processor)."""
        return self.processor.seq_token_id

    @property
    def struct_token_id(self):
        """Structure sequence token ID (forwarded to inner processor)."""
        return self.processor.struct_token_id

    @property
    def protein_tokenizer(self):
        return self.processor.protein_tokenizer

    @property
    def structure_tokenizer(self):
        return self.processor.structure_tokenizer
