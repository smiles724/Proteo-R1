"""
ProteoR1UnderstandProcessor - protein multimodal processor.

Combines the LLM tokenizer and ProtenixProcessor; implements Qwen2VL-style placeholder expansion.

Core features:
1. Process text input, supporting the <protein> placeholder.
2. Process protein structure (JSON entry) through ProtenixProcessor to obtain Protenix features.
3. Expand <protein> into N_token placeholders (N_token comes from the Protenix features).
4. Output text tokens together with protenix features.

Usage:
    processor = ProteoR1UnderstandProcessor(
        tokenizer=AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"),
        protenix_processor=ProtenixProcessor.from_pretrained("pretrained/protenix_encoder"),
    )

    # Multimodal input
    result = processor(
        text="Analyze this protein: <protein>",
        protein_json=json_entry,
    )

    # result contains:
    # - input_ids, attention_mask (text)
    # - protenix_input_feature_dict, protenix_atom_array, protenix_token_array (protein)
"""

import os
from typing import Any, Dict, List, Optional, Union

from transformers import AutoTokenizer, BatchFeature, ProcessorMixin
from transformers.utils import logging

from proteor1.understand.protenix_encoder.processing_protenix_encoder import (
    ProtenixProcessor,
    ProtenixProcessorOutput,
)

logger = logging.get_logger(__name__)

# Internal placeholder used when expanding the <protein> token.
PROTEIN_PLACEHOLDER = "<prot_placeholder>"


class ProteoR1UnderstandProcessor(ProcessorMixin):
    """
    ProteoR1Understand multimodal processor.

    Combines:
    - tokenizer: LLM tokenizer (Qwen3)
    - protenix_processor: Protenix protein processor

    Core features:
    1. Tokenize text.
    2. Process the protein-structure JSON to obtain Protenix features.
    3. Expand the <protein> placeholder into N_token tokens (Qwen2VL-style).
    4. Produce the mask needed to fill in embeddings.
    """

    attributes = ["tokenizer"]
    valid_kwargs = ["chat_template"]
    tokenizer_class = "AutoTokenizer"

    # Special tokens added to the tokenizer.
    # Note: Protenix encodes multi-chain directly and produces (N_token, embed); no <chain> marker required.
    add_token_list = [
        "<protein>",
        "<protein_start>",  # protein-region start marker
        "<protein_end>",    # protein-region end marker
        # "<think>", "</think>", # think tags are already in Qwen3's vocab
        # "<answer>", "</answer>", # express the answer tags as a pair of new tokens may not be a good choice
    ]

    def __init__(
            self,
            tokenizer=None,
            protenix_encoder_path: Optional[str] = None,
            protein_token: str = "<protein>",
            protein_start_token: str = "<protein_start>",
            protein_end_token: str = "<protein_end>",
            chat_template: Optional[str] = None,
            **kwargs
    ):
        """
        Initialize ProteoR1UnderstandProcessor.

        Args:
            tokenizer: LLM tokenizer (Qwen3)
            protenix_encoder_path: pretrained directory of the Protenix encoder.
                When provided, a ProtenixProcessor is created automatically.
            protein_token: protein placeholder token; default "<protein>"
            protein_start_token: protein-region start marker; default "<protein_start>"
            protein_end_token: protein-region end marker; default "<protein_end>"
            chat_template: optional chat template
        """
        # Add special tokens.
        added_token_num = 0
        for add_token in self.add_token_list:
            if add_token not in tokenizer.get_vocab():
                added_token_num += tokenizer.add_tokens(add_token)

        if added_token_num > 0:
            logger.warning(
                f"{added_token_num} new tokens have been added to the tokenizer vocabulary! "
                f"Don't forget to call `model.resize_token_embeddings(len(processor.tokenizer))` "
                f"to expand the token embeddings."
            )

        # Protein placeholder token.
        self.protein_token = protein_token
        self.protein_token_id = tokenizer.convert_tokens_to_ids(protein_token)

        # Protein boundary marker tokens.
        self.protein_start_token = protein_start_token
        self.protein_start_token_id = tokenizer.convert_tokens_to_ids(protein_start_token)
        self.protein_end_token = protein_end_token
        self.protein_end_token_id = tokenizer.convert_tokens_to_ids(protein_end_token)

        # Log token IDs for verification
        logger.info(
            f"ProteoR1UnderstandProcessor initialized with "
            f"protein_token_id={self.protein_token_id}, "
            f"protein_start_token_id={self.protein_start_token_id}, "
            f"protein_end_token_id={self.protein_end_token_id}"
        )

        # Stored for serialization and deserialization.
        self.protenix_encoder_path = protenix_encoder_path

        # Create ProtenixProcessor from the path.
        if protenix_encoder_path is not None and os.path.isdir(protenix_encoder_path):
            self.protenix_processor = ProtenixProcessor.from_pretrained(
                protenix_encoder_path,
                init_esm_tokenizer=True,
            )
        else:
            self.protenix_processor = None
            if protenix_encoder_path is not None:
                logger.warning(
                    f"protenix_encoder_path '{protenix_encoder_path}' is not a valid directory. "
                    f"ProtenixProcessor not loaded."
                )

        if chat_template is None:
            chat_template = tokenizer.chat_template
        super().__init__(tokenizer, chat_template=chat_template)

    def __call__(
            self,
            text: Union[str, List[str]],
            protein_json: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
            return_tensors: str = "pt",
            padding: bool = True,
            truncation: bool = False,
            add_special_tokens: bool = True,
            **kwargs
    ) -> BatchFeature:
        """
        Process multimodal input.

        Qwen2VL-style design:
        1. If protein_json is provided, run ProtenixProcessor first to obtain N_token.
        2. Expand <protein> in the text into N_token placeholders.
        3. Tokenize the expanded text.
        4. Return text tokens plus protenix features.

        Args:
            text: text input, may contain the <protein> placeholder
                - str: single text
                - List[str]: batched text (note: Protenix only supports batch_size=1)
            protein_json: Protenix JSON-format protein structure
                - dict: single JSON entry
                - List[dict]: batched JSON entries (note: only batch_size=1 is supported)
            return_tensors: return tensor type
            padding: whether to pad
            truncation: whether to truncate
            add_special_tokens: whether to add special tokens

        Returns:
            BatchFeature containing:
                - input_ids: [B, L] text token IDs
                - attention_mask: [B, L]
                - protenix_input_feature_dict: Protenix feature dict (when protein input present)
                - protenix_atom_array: biotite AtomArray
                - protenix_token_array: TokenArray
                - protenix_metadata: metadata
        """
        # ============ Step 1: normalize input ============
        if isinstance(text, str):
            text = [text]
        batch_size = len(text)

        # Protenix only supports batch_size=1 (O(N^2) memory for pair features).
        if batch_size > 1:
            raise NotImplementedError(
                f"batch_size > 1 is not supported for ProteoR1UnderstandProcessor. "
                f"Got batch_size={batch_size}. "
                f"Protenix uses O(N²) memory for pair features, use batch_size=1."
            )

        text = [t for t in text]  # copy to avoid mutating the caller's input

        # Normalize protein_json.
        if protein_json is None:
            protein_json_list = [None] * batch_size
        elif isinstance(protein_json, dict):
            protein_json_list = [protein_json]
        elif isinstance(protein_json, list):
            protein_json_list = protein_json
        else:
            raise ValueError(f"Unsupported protein_json type: {type(protein_json)}")

        if len(protein_json_list) != batch_size:
            raise ValueError(
                f"Mismatch: {batch_size} texts but {len(protein_json_list)} protein_json entries"
            )

        # ============ Step 2: process protein structure and obtain N_token ============
        protenix_outputs: List[Optional[ProtenixProcessorOutput]] = []
        n_tokens_list: List[int] = []

        for i, pj in enumerate(protein_json_list):
            if pj is not None and self.protenix_processor is not None:
                # Process the protein structure.
                protenix_output = self.protenix_processor(
                    json_entry=pj,
                    return_atom_array=True,
                    return_token_array=True,
                )
                protenix_outputs.append(protenix_output)
                n_tokens_list.append(protenix_output.metadata["N_token"])
            else:
                protenix_outputs.append(None)
                n_tokens_list.append(0)

        # ============ Step 3: check and expand <protein> placeholders ============
        # Expected total protein-token count (from Protenix N_token).
        expected_protein_tokens = sum(n_tokens_list)
        # Current protein-token count in the text.
        input_protein_tokens = sum(t.count(self.protein_token) for t in text)

        # Detect whether the text was already expanded.
        already_expanded = (input_protein_tokens == expected_protein_tokens) and expected_protein_tokens > 0

        if input_protein_tokens > expected_protein_tokens:
            # Token count in text exceeds the expected count — input data is inconsistent.
            raise RuntimeError(
                f"Protein token count in text ({input_protein_tokens}) exceeds "
                f"expected count from Protenix N_token ({expected_protein_tokens}). "
                f"Check your input data."
            )

        # When not yet expanded, expand now.
        # Expansion format: <protein> -> <protein_start><protein>*N<protein_end>
        if not already_expanded and expected_protein_tokens > 0:
            for i in range(batch_size):
                original_count = text[i].count(self.protein_token)

                if n_tokens_list[i] > 0:
                    # The original text must contain exactly one <protein>.
                    if original_count != 1:
                        raise RuntimeError(
                            f"Text[{i}] contains {original_count} <protein> tokens, "
                            f"but exactly 1 is required for expansion. "
                            f"Each text should have exactly one <protein> placeholder."
                        )

                    # Expand: replace <protein> with <protein_start><protein>*N<protein_end>.
                    n_token = n_tokens_list[i]
                    # Build the expanded string.
                    expanded = (
                        self.protein_start_token +
                        self.protein_token * n_token +
                        self.protein_end_token
                    )
                    text[i] = text[i].replace(self.protein_token, expanded, 1)

                elif original_count != 0:
                    # n_tokens_list[i] == 0 yet the text contains <protein>: no protein_json for this sample.
                    raise RuntimeError(
                        f"Text[{i}] contains {original_count} <protein> token(s), "
                        f"but no protein_json was provided for this sample. "
                        f"Either remove <protein> from text or provide protein_json."
                    )

        # Final check: counts must match after expansion.
        final_protein_tokens = sum(t.count(self.protein_token) for t in text)
        if expected_protein_tokens > 0 and final_protein_tokens != expected_protein_tokens:
            raise RuntimeError(
                f"Protein token count mismatch after expansion: "
                f"expected {expected_protein_tokens}, got {final_protein_tokens}."
            )

        # ============ Step 4: tokenize the expanded text ============
        text_tokens = self.tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            return_tensors=return_tensors,
            truncation=truncation,
            **kwargs
        )

        # ============ Step 5: compute position_ids ============
        # batch_size=1 was checked in step 1, so handle the first sample directly.
        input_ids_1d = text_tokens.input_ids.squeeze(0)  # [L]
        position_ids = self.compute_compressed_position_ids(input_ids_1d)  # [L]
        position_ids = position_ids.unsqueeze(0)  # [1, L]

        # ============ Step 6: build the return result ============
        result = {
            "input_ids": text_tokens.input_ids,
            "attention_mask": text_tokens.attention_mask,
            "position_ids": position_ids,
        }

        # Append Protenix features (batch_size=1 was checked in step 1).
        if protenix_outputs[0] is not None:
            po = protenix_outputs[0]
            result["protenix_input_feature_dict"] = po.input_feature_dict
            result["protenix_atom_array"] = po.atom_array
            result["protenix_token_array"] = po.token_array
            result["protenix_metadata"] = po.metadata

        return BatchFeature(result)

    def compute_compressed_position_ids(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        """
        Compute compressed position_ids so the protein region shares a single position_id.

        Rules (Bagel-style):
        1. Regular token: position_id increments.
        2. <protein_start>: enters the protein region; uses the current position_id.
        3. <protein> tokens: share the protein region's position_id.
        4. <protein_end>: still inside the protein region; shares the same position_id.
        5. After <protein_end>: position_id + 1 then continues to increment.

        Example:
            input_ids: [<protein_start>, prot, prot, prot, <protein_end>, text, text]
            position:  [0,               0,    0,    0,    0,             1,    2   ]

        Args:
            input_ids: [L] token IDs

        Returns:
            position_ids: [L] compressed position IDs
        """
        import torch

        L = input_ids.shape[0]
        position_ids = torch.zeros(L, dtype=torch.long)

        current_pos = 0
        in_protein_region = False

        for i in range(L):
            token_id = input_ids[i].item()

            if token_id == self.protein_start_token_id:
                # <protein_start>: enter the protein region.
                position_ids[i] = current_pos
                in_protein_region = True
                # Do not increment current_pos; share with the protein region.

            elif token_id == self.protein_end_token_id:
                # <protein_end>: exit the protein region; still shares the same position.
                position_ids[i] = current_pos
                in_protein_region = False
                current_pos += 1  # increment after the protein region ends

            elif in_protein_region and token_id == self.protein_token_id:
                # <protein> token: share the protein-region position.
                position_ids[i] = current_pos

            else:
                # Regular token: position increments.
                position_ids[i] = current_pos
                current_pos += 1

        return position_ids

    def get_position_info(self, protein_json: dict) -> dict:
        """
        Return only the position info (used by protenix_pos_embed).

        Lightweight helper used in precomputed mode to obtain residue_index and asym_id.
        It does not run an Encoder forward; it only does featurization.

        Args:
            protein_json: Protenix JSON-format protein structure

        Returns:
            {
                "residue_index": [N_token] residue index within the chain
                "asym_id": [N_token] chain ID
                "n_token": int, number of protein tokens
            }
        """
        if self.protenix_processor is None:
            raise RuntimeError(
                "protenix_processor is not initialized. "
                "Please provide protenix_encoder_path when creating ProteoR1UnderstandProcessor."
            )
        return self.protenix_processor.get_position_info(protein_json)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return [
            "input_ids",
            "attention_mask",
            "position_ids",
            "protenix_input_feature_dict",
            "protenix_atom_array",
            "protenix_token_array",
        ]

    def to_dict(self, legacy_serialization=True):
        """
        Serialize to a dict, excluding the non-serializable protenix_processor.
        """
        output = super().to_dict(legacy_serialization=legacy_serialization)
        # protenix_processor cannot be JSON-serialized; it is reloaded from protenix_encoder_path.
        output.pop("protenix_processor", None)
        return output

    def save_pretrained(self, save_directory, **kwargs):
        """Save the processor and its tokenizer."""
        super().save_pretrained(save_directory, **kwargs)
        # ProtenixProcessor does not need to be saved separately; it loads from the protenix_encoder directory.

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            protenix_encoder_path: Optional[str] = None,
            **kwargs
    ):
        """
        Load the processor from a pretrained directory.

        Args:
            pretrained_model_name_or_path: processor save directory
            protenix_encoder_path: pretrained directory of the Protenix encoder.
                When None, use the path stored at save time (loaded from processor_config.json).
                When provided, override the saved path.

        Returns:
            ProteoR1UnderstandProcessor instance.
        """
        # If protenix_encoder_path is passed explicitly, override the value in the config.
        if protenix_encoder_path is not None:
            kwargs["protenix_encoder_path"] = protenix_encoder_path

        # Load the processor (tokenizer + processor_config.json).
        # __init__ will create ProtenixProcessor automatically based on protenix_encoder_path.
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


def test():
    """Smoke-test ProteoR1UnderstandProcessor."""
    import shutil

    protenix_path = "pretrained/protenix_mini_ism_v0.5.0"

    # Create the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

    # Create the processor; passing the path auto-creates ProtenixProcessor.
    processor = ProteoR1UnderstandProcessor(
        tokenizer=tokenizer,
        protenix_encoder_path=protenix_path,
    )

    print(f"protein_token: {processor.protein_token}")
    print(f"protein_token_id: {processor.protein_token_id}")
    print(f"protenix_encoder_path: {processor.protenix_encoder_path}")
    print(f"Tokenizer vocab size: {len(processor.tokenizer)}")
    print(f"ProtenixProcessor loaded: {processor.protenix_processor is not None}")

    test_prot_json = {
        "name": "P0C9F0",
        "sequences": [
            {
                "proteinChain": {
                    "sequence": "MVRLFYNPIKYLFYRRSCKKRLRKALKKLNFYHPPKECCQIYRLLENAPGGTYFITENMTNELIMIAKDPVDKKIKSVKLYLTGNYIKINQHYYINIYMYLMRYNQIYKYPLICFSKYSKIL",
                    "count": 1
                }
            }
        ],
        "_metadata": {
            "source": "uniprot",
            "caption": "Plays a role in virus cell tropism, and may be required for efficient virus replication in macrophages. Belongs to the asfivirus MGF 100 family.",
            "seq_len": 122,
            "caption_len": 144,
            "caption_key": "plays a role in virus cell tropism, and may be required for efficient virus replication in macrophages. belongs to the asfivirus mgf 100 family.",
            "dup_count": 4
        }
    }

    # Multimodal smoke test.
    result = processor(
        text="Analyze this protein: <protein>",
        protein_json=test_prot_json,
    )
    print(f"\nMultimodal test:")
    print(f"  input_ids shape: {result.input_ids.shape}")
    print(f"  N_token from metadata: {result.get('protenix_metadata', {}).get('N_token', 'N/A')}")

    # Save and reload smoke test.
    test_dir = "temp/test_protenix_qwen_processor"
    shutil.rmtree(test_dir, ignore_errors=True)
    processor.save_pretrained(test_dir)
    print(f"\nSaved processor to {test_dir}")

    # Reload (without passing protenix_encoder_path, use the saved path).
    loaded = ProteoR1UnderstandProcessor.from_pretrained(test_dir)
    print(f"Loaded processor (using saved path):")
    print(f"  protein_token: {loaded.protein_token}")
    print(f"  protein_token_id: {loaded.protein_token_id}")
    print(f"  protenix_encoder_path: {loaded.protenix_encoder_path}")
    print(f"  ProtenixProcessor loaded: {loaded.protenix_processor is not None}")

    # Reload (with a new protenix_encoder_path that overrides).
    loaded2 = ProteoR1UnderstandProcessor.from_pretrained(
        test_dir,
        protenix_encoder_path=protenix_path,  # explicit override
    )
    print(f"\nLoaded processor (with override path):")
    print(f"  protenix_encoder_path: {loaded2.protenix_encoder_path}")
    print(f"  ProtenixProcessor loaded: {loaded2.protenix_processor is not None}")

    # Cleanup.
    shutil.rmtree(test_dir, ignore_errors=True)
    print("\nTest passed!")


if __name__ == "__main__":
    test()
