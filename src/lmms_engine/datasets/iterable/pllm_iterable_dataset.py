"""
PLLM Iterable Dataset for protein language models.

Features:
- Inherits from MultiModalIterableDataset
- Uses HuggingFace messages format
- Complete label masking (supervises assistant only)
- Multi-turn conversation support
- Prepared for future packing implementation
"""
import string
import copy
from dataclasses import dataclass
from typing import Dict, List

from .multimodal_iterable_dataset import MultiModalIterableDataset

from lmms_engine.mapping_func import register_dataset
from lmms_engine.datasets.collator.pllm_collator import PLLMCollator
from ...models.aero.modeling_aero import logger

CHAIN_ID_CANDIDATES = string.ascii_uppercase + string.digits + string.ascii_lowercase


MM_PROMPT_DICT = {
    "vanilla": "{question}\n\nInputs:".strip(),

    "open-ended": """
You are a professional protein biologist.
You will receive a multi-chain protein in which each chain is described by two aligned tracks: (1) its amino-acid sequence (primary structure) and (2) a residue-wise discrete 3D structural track capturing local tertiary context.

Use both the sequence evidence and the structural track in a chain-aware manner to answer the open-ended question below:
{question}

First, reason step by step inside a <think></think> block, explicitly tying sequence-derived cues to structural signals.
Then, provide your final response inside an <answer></answer> block.

Multi-chain Protein Input:
""".strip(),

    "yes–no": """
You are a professional protein biologist.
You will receive a multi-chain protein in which each chain is described by two aligned tracks: (1) its amino-acid sequence (primary structure) and (2) a residue-wise discrete 3D structural track capturing local tertiary context.

Use both the sequence evidence and the structural track in a chain-aware manner to answer the yes-no question below:
{question}

First, reason step by step inside a <think></think> block, explicitly tying sequence-derived cues to structural signals.
Then, provide your final answer ("Yes" or "No") inside an <answer></answer> block.

Multi-chain Protein Input:
    """.strip(),

    "multiple-choice": """
You are a professional protein biologist.
You will receive a multi-chain protein in which each chain is described by two aligned tracks: (1) its amino-acid sequence (primary structure) and (2) a residue-wise discrete 3D structural track capturing local tertiary context.

Use both the sequence evidence and the structural track in a chain-aware manner to answer the multiple-choice question below:
{question}

First, reason step by step inside a <think></think> block, explicitly tying sequence-derived cues to structural signals.
Then, provide your final answer (the option letter of the correct one) inside an <answer></answer> block.

Multi-chain Protein Input:
    """.strip(),

    "zero-shot": """
You are a professional protein biologist.
You will receive a multi-chain protein in which each chain is described by its amino-acid sequence (primary structure).

Use both the sequence evidence in a chain-aware manner to answer the multiple-choice question below:
{question}

Provide your final answer (the option letter of the correct one) inside an <answer></answer> block.

Multi-chain Protein Input:
    """.strip(),
}


def parse_chains(chains: Dict):
    aa_seq_list = []
    stru_str_list = []
    chain_prompt = ""

    for chain_id in CHAIN_ID_CANDIDATES:
        if chain_id not in chains:
            continue
        if chains[chain_id] is None:
            continue

        if "aa_seq" not in chains[chain_id] or "threeDi_seq" not in chains[chain_id]:
            raise RuntimeError(
                f"Missing required fields in chain {chain_id}. "
                f"Expected 'aa_seq' and 'threeDi_seq', got keys: {list(chains[chain_id].keys())}"
            )

        aa_seq_list.append(chains[chain_id]["aa_seq"])
        stru_str_list.append(chains[chain_id]["threeDi_seq"].lower())
        # No colon after chain letter to prevent tokenization differences
        chain_prompt += f"\n<chain>Chain {chain_id}<seq><aa_seq></seq><struct><3d_struct></struct></chain>"

    return (aa_seq_list, stru_str_list), chain_prompt.strip()


def parse_sft_doc_by_keys(doc: Dict):
    """Parse SFT document into user prompt, assistant response, and protein info."""
    if "chains" in doc:
        if doc.get("response") is not None:
            question = doc["question"]
            assistant_response = doc["response"]
            doc_type = "open-ended"
            user_prompt = MM_PROMPT_DICT[doc_type].format(question=question)

        elif doc.get("agent2_qa_list") is not None:
            if len(doc["agent2_qa_list"]) != 1:
                raise NotImplementedError(
                    f"Only single QA pair supported, got {len(doc['agent2_qa_list'])} pairs"
                )

            question = doc["agent2_qa_list"][0]["Q"]

            think = doc["agent2_qa_list"][0]["Ans"]["think"]
            answer = doc["agent2_qa_list"][0]["Ans"]["answer"]
            assistant_response = "<think>\n" + think + "\n</think>\n\n<answer>\n" + answer + "\n</answer>"

            if "type" in doc["agent2_qa_list"][0]:
                doc_type = doc["agent2_qa_list"][0]["type"]
            else:
                doc_type = "multiple-choice" if "options" in doc["agent2_qa_list"][0] else "yes-no"

            if doc_type == "yes–no":
                user_prompt = MM_PROMPT_DICT["multiple-choice"].format(question=question + "  A: Yes  B: No")

            elif doc_type == "multiple-choice":
                options = doc["agent2_qa_list"][0]["options"]
                options_str = ""
                for opt_idx in range(len(options)):
                    opt_letter = string.ascii_uppercase[opt_idx]
                    opt = options[opt_idx]
                    options_str += f"{opt_letter}: {opt}  "  # Two spaces after each option
                options_str = "  " + options_str.strip()  # Two spaces before first option
                user_prompt = MM_PROMPT_DICT["multiple-choice"].format(question=question + options_str)

            else:
                raise ValueError(f"Unknown question type: {doc_type}")

        else:
            raise RuntimeError(
                f"Unknown document format. Expected 'response' or 'agent2_qa_list', "
                f"got keys: {list(doc.keys())}, doc={doc}"
            )

        chains = doc["chains"]
        if len(chains) == 0:
            raise RuntimeError(f"Empty chains for {doc['pdb_id']}")

        protein_info, chain_prompt = parse_chains(chains)
        user_prompt += "\n" + chain_prompt

    # Text-only training data (future support)
    else:
        raise NotImplementedError("Text-only training data not yet supported")

    return user_prompt, assistant_response, protein_info



@register_dataset("pllm_iterable")
class PLLMIterableDataset(MultiModalIterableDataset):
    def build(self):
        super().build()
        self.pllm_plugin = get_pllm_plugin(
            seq_token=self.processor.processor.seq_token,
            struct_token=self.processor.processor.struct_token,
            expand_mm_tokens=True
        )

        error_num = 0
        for doc in self.data_list:
            has_response = doc.get("response") is not None and doc.get("response") != ""
            has_question = doc.get("question") is not None and doc.get("question") != ""
            has_agent2_qa = doc.get("agent2_qa_list") is not None and len(doc.get("agent2_qa_list", [])) > 0
            if (not has_response or not has_question) and not has_agent2_qa:
                error_num += 1
        logger.warning(f"{error_num} out of {len(self.data_list)} samples have error!")

    def load_from_json(self, data, data_folder=None):
        doc = copy.deepcopy(data)

        user_prompt, assistant_response, protein_info = parse_sft_doc_by_keys(doc)
        aa_seq_list, stru_str_list = protein_info

        messages = [
            dict(role="user", content=user_prompt),
            dict(role="assistant", content=assistant_response)
        ]

        expanded_messages = self.pllm_plugin.process_messages(
            messages=messages,
            proteins=aa_seq_list,
            structures=stru_str_list,
            processor=self.processor.processor
        )

        inputs = self.processor.process(hf_messages=expanded_messages, aa_seq=aa_seq_list, stru_str=stru_str_list)
        return inputs

    def get_collator(self):
        return PLLMCollator(self.processor)


PROTEIN_PLACEHOLDER = "<aa_placeholder>"
STRUCTURE_PLACEHOLDER = "<struct_placeholder>"


@dataclass
class PLLMPlugin:
    """
    PLLM Plugin for protein language model multimodal training.

    Similar to Qwen2VLPlugin design:
    1. process_messages: Expand protein/structure tokens before tokenization
    2. process_token_ids: Process token ids after tokenization (usually not needed)
    3. get_mm_inputs: Batch process protein data in DataCollator

    Args:
        seq_token: Special token for protein sequences (default: "<aa_seq>")
        struct_token: Special token for structure sequences (default: "<3d_struct>")
        expand_mm_tokens: Whether to expand multimodal tokens
    """
    seq_token: str = "<aa_seq>"
    struct_token: str = "<3d_struct>"
    expand_mm_tokens: bool = True

    def _validate_input(
        self,
        processor,
        proteins: List[str],
        structures: List[str],
    ) -> None:
        """Validate input arguments."""
        if (len(proteins) > 0 or len(structures) > 0) and processor is None:
            raise ValueError("Processor was not found for protein input.")

        if processor is not None:
            if not hasattr(processor, "protein_tokenizer") or processor.protein_tokenizer is None:
                raise ValueError("Protein tokenizer was not found in processor.")

            if not hasattr(processor, "structure_tokenizer") or processor.structure_tokenizer is None:
                raise ValueError("Structure tokenizer was not found in processor.")

    def _validate_messages(
        self,
        messages: list[dict[str, str]],
        proteins: List[str],
        structures: List[str],
    ) -> None:
        """Validate that placeholder counts match protein/structure data."""
        for msg in messages:
            if msg["content"] is None:
                logger.warning(f"None is found in the given messages: {messages}")
                break

        num_protein_tokens = sum(msg["content"].count(self.seq_token) for msg in messages if msg.get("content"))
        num_structure_tokens = sum(msg["content"].count(self.struct_token) for msg in messages if msg.get("content"))

        if len(proteins) != num_protein_tokens:
            raise ValueError(
                f"The number of proteins ({len(proteins)}) does not match the number of "
                f"{self.seq_token} tokens ({num_protein_tokens}) in messages."
            )

        if len(structures) != num_structure_tokens:
            raise ValueError(
                f"The number of structures ({len(structures)}) does not match the number of "
                f"{self.struct_token} tokens ({num_structure_tokens}) in messages."
            )

    def process_messages(
        self,
        messages: list[dict[str, str]],
        proteins: List[str],
        structures: List[str],
        processor,
    ) -> list[dict[str, str]]:
        """
        Process messages before tokenization, expanding protein/structure placeholders.

        Similar to Qwen2VLPlugin.process_messages.

        Args:
            messages: Conversation messages with potential protein/structure placeholders
            proteins: Protein sequence data
            structures: Structure sequence data
            processor: PLLM processor

        Returns:
            Processed messages with expanded special tokens
        """
        self._validate_input(processor, proteins, structures)
        self._validate_messages(messages, proteins, structures)

        messages = copy.deepcopy(messages)
        if not self.expand_mm_tokens:
            return messages

        # Tokenize to get sequence lengths
        protein_lengths = []
        if len(proteins) > 0:
            tokens = processor.protein_tokenizer(
                proteins,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt"
            )
            protein_lengths = tokens["attention_mask"].sum(dim=1).tolist()

        structure_lengths = []
        if len(structures) > 0:
            tokens = processor.structure_tokenizer(
                structures,
                add_special_tokens=True,
                padding=True,
                truncation=False,
                return_tensors="pt"
            )
            structure_lengths = tokens["attention_mask"].sum(dim=1).tolist()

        # IMPORTANT: -2 to remove bos and eos features
        expected_protein_total_len = sum(protein_lengths) - 2 * len(protein_lengths)
        expected_structure_total_len = sum(structure_lengths) - 2 * len(structure_lengths)

        input_protein_total_len = 0
        input_structure_total_len = 0
        for msg in messages:
            content = msg["content"]
            input_protein_total_len += content.count(self.seq_token)
            input_structure_total_len += content.count(self.struct_token)

        protein_satisfy = input_protein_total_len == expected_protein_total_len
        structure_satisfy = input_structure_total_len == expected_structure_total_len

        if protein_satisfy and structure_satisfy:
            return messages
        elif protein_satisfy ^ structure_satisfy:
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

        protein_counter = 0
        structure_counter = 0
        for message in messages:
            content = message["content"]

            while self.seq_token in content:
                if protein_counter < len(protein_lengths):
                    seq_len = protein_lengths[protein_counter]
                    # IMPORTANT: -2 to remove bos and eos features
                    expanded = PROTEIN_PLACEHOLDER * (seq_len - 2)
                    content = content.replace(self.seq_token, expanded, 1)
                    protein_counter += 1
                else:
                    break

            content = content.replace(PROTEIN_PLACEHOLDER, self.seq_token)

            while self.struct_token in content:
                if structure_counter < len(structure_lengths):
                    seq_len = structure_lengths[structure_counter]
                    # IMPORTANT: -2 to remove bos and eos features
                    expanded = STRUCTURE_PLACEHOLDER * (seq_len - 2)
                    content = content.replace(self.struct_token, expanded, 1)
                    structure_counter += 1
                else:
                    break

            content = content.replace(STRUCTURE_PLACEHOLDER, self.struct_token)

            message["content"] = content

        return messages


def get_pllm_plugin(
    seq_token: str = "<aa_seq>",
    struct_token: str = "<3d_struct>",
    expand_mm_tokens: bool = True,
) -> PLLMPlugin:

    return PLLMPlugin(
        seq_token=seq_token,
        struct_token=struct_token,
        expand_mm_tokens=expand_mm_tokens
    )

