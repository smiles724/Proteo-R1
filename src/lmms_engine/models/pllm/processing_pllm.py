import os
from os.path import dirname
from typing import List, Optional, Union
from transformers import ProcessorMixin, BatchFeature, AutoTokenizer, EsmTokenizer
from transformers.processing_utils import ProcessingKwargs
from transformers.utils import logging

import lmms_engine

logger = logging.get_logger(__name__)


PROTEIN_PLACEHOLDER = "<aa_placeholder>"
STRUCTURE_PLACEHOLDER = "<struct_placeholder>"


class PLLMProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        }
    }


class PLLMProcessor(ProcessorMixin):
    """
    PLLM multimodal processor integrating 3 tokenizers.

    Tokenizers:
    - tokenizer: Text tokenizer (Qwen/LLaMA etc.)
    - protein_tokenizer: ESM tokenizer for amino acid sequences
    - structure_tokenizer: ESM tokenizer for structure (Foldseek 3Di)

    Core functionality:
    1. Tokenize all modalities
    2. Calculate sequence lengths
    3. Build input_ids with special token placeholders
    4. Generate masks and index mappings for embedding
    """

    attributes = ["tokenizer"]
    valid_kwargs = ["chat_template"]

    tokenizer_class = "AutoTokenizer"

    add_token_list = [
        "<chain>", "</chain>",
        "<seq>", "<aa_seq>", "</seq>",
        "<struct>", "<3d_struct>", "</struct>",
        "<think>", "</think>", "<answer>", "</answer>"
    ]

    def __init__(
        self,
        tokenizer=None,
        protein_tokenizer=None,
        structure_tokenizer=None,
        chat_template=None,
        **kwargs
    ):
        added_token_num = tokenizer.add_tokens(self.add_token_list)
        if added_token_num > 0:
            logger.warning(
                f"{added_token_num} new tokens have been added into vocabulary of your tokenizer! "
                f"Please don't forget to call `resize_token_embeddings` of your model to expand the token embeddings!"
            )

        self.seq_token = "<aa_seq>"
        self.struct_token = "<3d_struct>"
        self.seq_token_id = tokenizer.convert_tokens_to_ids(self.seq_token)
        self.struct_token_id = tokenizer.convert_tokens_to_ids(self.struct_token)

        # Manually set protein and structure tokenizers since they're not in attributes
        self.protein_tokenizer = protein_tokenizer
        self.structure_tokenizer = structure_tokenizer

        if chat_template is None:
            chat_template = tokenizer.chat_template
        super().__init__(tokenizer, chat_template=chat_template)

    def __call__(
            self,
            text: Union[str, List[str]] = None,
            aa_seq: Optional[Union[str, List[str]]] = None,
            stru_str: Optional[Union[str, List[str]]] = None,
            return_tensors: str = "pt",
            padding: bool = True,
            truncation: bool = False,
            add_special_tokens: bool = True,
            **kwargs
    ) -> BatchFeature:

        # Step 1: Normalize inputs
        if isinstance(text, str):
            text = [text]
        batch_size = len(text)

        text = text.copy()  # Avoid modifying original input

        # Step 2: Normalize aa_seq and stru_str to flat List[str] format
        # API design: aa_seq and stru_str are flat lists containing all chains from all samples
        if aa_seq is None:
            aa_seq_flat = None
        elif isinstance(aa_seq, str):
            aa_seq_flat = [aa_seq]
        elif isinstance(aa_seq, list):
            aa_seq_flat = aa_seq
        else:
            raise ValueError(f"Unsupported aa_seq type: {type(aa_seq)}")

        if stru_str is None:
            stru_str_flat = None
        elif isinstance(stru_str, str):
            stru_str_flat = [stru_str]
        elif isinstance(stru_str, list):
            stru_str_flat = stru_str
        else:
            raise ValueError(f"Unsupported stru_str type: {type(stru_str)}")

        # Step 3: Tokenize protein/structure sequences
        protein_tokens = None
        structure_tokens = None
        protein_lengths = []
        structure_lengths = []

        if aa_seq_flat is not None and len(aa_seq_flat) > 0:
            # 批量tokenize所有protein chains
            protein_tokens = self.protein_tokenizer(
                aa_seq_flat,
                add_special_tokens=True, # always add bos and eos
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
            )
            protein_lengths = protein_tokens["attention_mask"].sum(dim=1).tolist()

        if stru_str_flat is not None and len(stru_str_flat) > 0:
            # 批量tokenize所有structure chains
            structure_tokens = self.structure_tokenizer(
                stru_str_flat,
                add_special_tokens=True, # always add bos and eos
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors,
            )
            structure_lengths = structure_tokens["attention_mask"].sum(dim=1).tolist()

        # IMPORTANT: -2 to remove bos and eos features
        expected_protein_total_len = sum(protein_lengths) - 2 * len(protein_lengths)
        expected_structure_total_len = sum(structure_lengths) - 2 * len(structure_lengths)

        input_protein_total_len = 0
        input_structure_total_len = 0
        for t in text:
            input_protein_total_len += t.count(self.seq_token)
            input_structure_total_len += t.count(self.struct_token)

        protein_satisfy = input_protein_total_len == expected_protein_total_len
        structure_satisfy = input_structure_total_len == expected_structure_total_len

        # Check if expansion is needed
        if protein_satisfy ^ structure_satisfy:
            # One expanded, one not - inconsistent state
            raise RuntimeError(
                f"Inconsistent expansion state: "
                f"protein {'already expanded' if protein_satisfy else 'needs expansion'} "
                f"(expected {expected_protein_total_len} tokens, got {input_protein_total_len}), "
                f"structure {'already expanded' if structure_satisfy else 'needs expansion'} "
                f"(expected {expected_structure_total_len} tokens, got {input_structure_total_len}). "
                f"Both should be in the same state."
            )
        elif input_protein_total_len > expected_protein_total_len:
            raise RuntimeError(
                f"Protein token count exceeds expected. "
                f"Expected {expected_protein_total_len}, got {input_protein_total_len}"
            )
        elif input_structure_total_len > expected_structure_total_len:
            raise RuntimeError(
                f"Structure token count exceeds expected. "
                f"Expected {expected_structure_total_len}, got {input_structure_total_len}"
            )

        # If both not expanded, perform expansion
        if not protein_satisfy or not structure_satisfy:
            # Step 3: Expand special tokens to placeholders
            # Benefit: When tokenize=False (e.g., apply_chat_template), can use text with special tokens directly
            seq_counter = 0
            struct_counter = 0

            for i in range(batch_size):
                if protein_tokens is not None:
                    while self.seq_token in text[i]:
                        if seq_counter < len(protein_tokens["input_ids"]):
                            valid_len = protein_tokens["attention_mask"][seq_counter].sum().item()
                            # IMPORTANT: -2 to remove bos and eos features
                            text[i] = text[i].replace(self.seq_token, PROTEIN_PLACEHOLDER * (valid_len - 2), 1)
                            seq_counter += 1
                        else:
                            # No more proteins available
                            break

                    text[i] = text[i].replace(PROTEIN_PLACEHOLDER, self.seq_token)

                if structure_tokens is not None:
                    while self.struct_token in text[i]:
                        if struct_counter < len(structure_tokens["input_ids"]):
                            valid_len = structure_tokens["attention_mask"][struct_counter].sum().item()
                            # IMPORTANT: -2 to remove bos and eos features
                            text[i] = text[i].replace(self.struct_token, STRUCTURE_PLACEHOLDER * (valid_len - 2), 1)
                            struct_counter += 1
                        else:
                            # No more structures available
                            break

                    text[i] = text[i].replace(STRUCTURE_PLACEHOLDER, self.struct_token)

        # Step 4: Tokenize expanded text and locate special tokens
        text_tokens = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            return_tensors=return_tensors,
            truncation=truncation,
            **kwargs
        )

        # Step 5: Build return result
        result = {
            "input_ids": text_tokens.input_ids,
            "attention_mask": text_tokens.attention_mask,
        }
        if protein_tokens is not None:
            result["protein_input_ids"] = protein_tokens["input_ids"]
            result["protein_attention_mask"] = protein_tokens["attention_mask"]

        if structure_tokens is not None:
            result["structure_input_ids"] = structure_tokens["input_ids"]
            result["structure_attention_mask"] = structure_tokens["attention_mask"]

        return BatchFeature(result)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return [
            "input_ids",
            "attention_mask",
            "protein_input_ids",
            "protein_attention_mask",
            "structure_input_ids",
            "structure_attention_mask",
        ]

    def save_pretrained(self, save_directory, **kwargs):
        """Save processor and all tokenizers."""
        super().save_pretrained(save_directory, **kwargs)
        self.protein_tokenizer.save_pretrained(f"{save_directory}/protein_tokenizer")
        self.structure_tokenizer.save_pretrained(f"{save_directory}/structure_tokenizer")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        # Use super() to load the main processor (which loads tokenizer and processor_config.json)
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Load protein_tokenizer and structure_tokenizer from subdirectories
        # This works for both local paths and HuggingFace Hub IDs

        # Handle both local path and HuggingFace Hub ID
        if os.path.isdir(pretrained_model_name_or_path):
            # Local path
            protein_tokenizer_path = os.path.join(pretrained_model_name_or_path, "protein_tokenizer")
            structure_tokenizer_path = os.path.join(pretrained_model_name_or_path, "structure_tokenizer")
            processor.protein_tokenizer = EsmTokenizer.from_pretrained(protein_tokenizer_path)
            processor.structure_tokenizer = EsmTokenizer.from_pretrained(structure_tokenizer_path)

            return processor
        else:
            # Hub ID - let from_pretrained handle the download
            # HuggingFace Hub supports subfolder parameter
            protein_tokenizer_path = pretrained_model_name_or_path
            structure_tokenizer_path = pretrained_model_name_or_path
            kwargs["subfolder"] = "protein_tokenizer"
            processor.protein_tokenizer = EsmTokenizer.from_pretrained(
                protein_tokenizer_path, **kwargs
            )
            kwargs["subfolder"] = "structure_tokenizer"
            processor.structure_tokenizer = EsmTokenizer.from_pretrained(
                structure_tokenizer_path, **kwargs
            )
            return processor


def test():
    pretrained_root = f"{dirname(lmms_engine.__file__)}/../../pretrained"
    temp_root = f"{dirname(lmms_engine.__file__)}/../../temp"

    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True)
    protein_tokenizer = EsmTokenizer.from_pretrained(f"{pretrained_root}/ProTrek_650M/esm2_t33_650M_UR50D")
    structure_tokenizer = EsmTokenizer.from_pretrained(f"{pretrained_root}/ProTrek_650M/foldseek_t30_150M")

    processor = PLLMProcessor(
        tokenizer=llm_tokenizer,
        protein_tokenizer=protein_tokenizer,
        structure_tokenizer=structure_tokenizer
    )
    processor.save_pretrained(f"{temp_root}/pllm_processor")

    # Test local loading
    processor_loaded = PLLMProcessor.from_pretrained(f"{temp_root}/pllm_processor")
    print("✅ Local loading successful!")
    print(f"Tokenizer vocab size: {len(processor_loaded.tokenizer)}")
    print(f"Protein tokenizer vocab size: {len(processor_loaded.protein_tokenizer)}")
    print(f"Structure tokenizer vocab size: {len(processor_loaded.structure_tokenizer)}")

    # Test if tokenizers work
    test_result = processor_loaded(
        text="Test <aa_seq> protein",
        aa_seq=["MKLL"],
        stru_str=["aabb"],
    )
    print(f"✅ Processor works! input_ids shape: {test_result.input_ids.shape}")


if __name__ == '__main__':
    test()

