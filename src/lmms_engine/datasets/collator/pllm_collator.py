import collections
from dataclasses import dataclass
from typing import Sequence, Dict

import torch

from lmms_engine.protocol import Processable


@dataclass
class PLLMCollator:
    """
    Data collator for PLLM that handles protein and structure sequences.

    Flattens multi-chain protein data and pads all sequences in a batch.
    """
    processor: Processable

    def pad_sequence(self, input_ids, batch_first, padding_value=None, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.processor.tokenizer
        if padding_value is None:
            padding_value = tokenizer.pad_token_id

        if tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if isinstance(instances[0], list):
            instances = [inst for instance in instances for inst in instance]
        inputs = collections.defaultdict(list)
        for instance in instances:
            for key, values in instance.items():
                inputs[key].append(values)

        batched_inputs = {}
        if "input_ids" in inputs.keys():
            input_ids = inputs.pop("input_ids")
            input_ids = self.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            )
            batched_inputs["input_ids"] = input_ids
        if "labels" in inputs.keys():
            labels = inputs.pop("labels")
            labels = self.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            )
            batched_inputs["labels"] = labels

        if "attention_mask" in inputs.keys():
            inputs.pop("attention_mask")

        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id).long()
        batched_inputs["attention_mask"] = attention_mask

        # protein data collation
        # Each item in inputs["protein_input_ids"] is a 2D tensor [num_chains, seq_len]
        # Flatten all chains from all samples into a single list of 1D tensors
        protein_input_ids = []
        for item in inputs["protein_input_ids"]:  # item: [num_chains, seq_len]
            for i in range(len(item)):  # iterate over chains
                protein_input_ids.append(item[i])  # append 1D tensor [seq_len]

        protein_input_ids = self.pad_sequence(
            protein_input_ids, batch_first=True, tokenizer=self.processor.protein_tokenizer
        )
        protein_attention_mask = protein_input_ids.ne(self.processor.protein_tokenizer.pad_token_id).long()
        batched_inputs.update(
            protein_input_ids=protein_input_ids,
            protein_attention_mask=protein_attention_mask,
        )
        inputs.pop("protein_input_ids")
        inputs.pop("protein_attention_mask")

        # structure data collation
        # Each item in inputs["structure_input_ids"] is a 2D tensor [num_chains, seq_len]
        # Flatten all chains from all samples into a single list of 1D tensors
        structure_input_ids = []
        for item in inputs["structure_input_ids"]:  # item: [num_chains, seq_len]
            for i in range(len(item)):  # iterate over chains
                structure_input_ids.append(item[i])  # append 1D tensor [seq_len]

        structure_input_ids = self.pad_sequence(
            structure_input_ids, batch_first=True, tokenizer=self.processor.structure_tokenizer
        )
        structure_attention_mask = structure_input_ids.ne(self.processor.structure_tokenizer.pad_token_id).long()
        batched_inputs.update(
            structure_input_ids=structure_input_ids,
            structure_attention_mask=structure_attention_mask,
        )
        inputs.pop("structure_input_ids")
        inputs.pop("structure_attention_mask")

        # Handle remaining keys
        for key, values in inputs.items():
            # Handle scalar/boolean values ( use_audio_in_video)
            if isinstance(values[0], bool) or (
                    isinstance(values[0], (int, float)) and not isinstance(values[0], torch.Tensor)
            ):
                batched_inputs[key] = values[0]
            else:
                batched_inputs[key] = torch.concatenate(values, dim=0)
        return batched_inputs
