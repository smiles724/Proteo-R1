from dataclasses import dataclass

import numpy as np
import torch
from transformers import DataCollatorForSeq2Seq


@dataclass
class ProteinLLMChainDataCollator(DataCollatorForSeq2Seq):
    return_metadata: bool = False
    return_protein_idx: bool = False

    def __call__(self, features, return_tensors=None):
        hf_features = []
        other_features = []
        for feat in features:
            hf_features.append({})
            other_features.append({})
            for key in feat:
                if key in ["input_ids", "attention_mask"]:
                    hf_features[-1][key] = feat[key]
                elif key in ["labels"]:
                    key_feat = feat[key]
                    # avoid slow convertion from List[np.ndarray] to torch.Tensor in super()
                    if isinstance(key_feat, (np.ndarray, torch.Tensor)):
                        key_feat = key_feat.tolist()
                    hf_features[-1][key] = key_feat
                else:
                    other_features[-1][key] = feat[key]

        batch = super(ProteinLLMChainDataCollator, self).__call__(features=hf_features, return_tensors=return_tensors)
        for key in other_features[0].keys():
            if key == "metadata":
                if not self.return_metadata:
                    continue
                batch[key] = []
                for feat in other_features:
                    batch[key].append(feat[key])

            # None means text-only data
            elif key in ["aa_seq", "stru_str"]:
                batch[key] = []
                for feat in other_features:
                    # None means text-only data
                    if feat[key] is None:
                        raise NotImplementedError
                    batch[key].extend(feat[key])

            else:
                raise ValueError(key)

        if self.return_protein_idx:
            batch["protein_idx"] = []
            for i, feat in enumerate(other_features):
                assert len(feat["aa_seq"]) == len(feat["stru_str"]), (len(feat["aa_seq"]), len(feat["stru_str"]))
                batch["protein_idx"].extend([i] * len(feat["aa_seq"]))
            batch["protein_idx"] = torch.LongTensor(batch["protein_idx"])

        return batch
