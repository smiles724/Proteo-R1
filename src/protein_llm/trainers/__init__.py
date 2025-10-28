import torch

from typing import Optional

from torch.utils.data import Dataset, RandomSampler
from transformers import Trainer
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length


class ProteinLLMTrainer(Trainer):
    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None:
            train_dataset = self.train_dataset
        if train_dataset is None or not has_length(train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            lengths = getattr(train_dataset, "lengths", None)
            if lengths is None:
                raise RuntimeError("Please register `lengths` as List[int] in your `train_dataset`")

            print("[INFO] group_by_length is activated by train_dataset.lengths!")
            sorted_lengths = sorted(lengths)
            n = len(sorted_lengths)
            median = sorted_lengths[n//2] if n % 2 == 1 else (sorted_lengths[n//2-1] + sorted_lengths[n//2]) / 2
            q25 = sorted_lengths[n//4]
            q75 = sorted_lengths[3*n//4]
            print(f"[INFO] lengths statistics: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.2f}, median={median:.2f}, q25={q25}, q75={q75}, total_samples={len(lengths)}")

            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        else:
            return RandomSampler(train_dataset)

