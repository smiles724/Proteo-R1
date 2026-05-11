# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ProtenixEncoder - end-to-end protein sequence encoder.

Has two parts:
1. ESM Encoder: ESM2-3B-ISM, produces token embedding (2560-dim) from amino-acid sequences.
2. Protenix Encoder: Pairformer, turns token embedding into structure-aware s embedding (384-dim).

Usage:
    from temp import ProtenixEncoder, ProteinLLMProjector

    # Load from a pretrained directory (contains config.json, protenix.pt, esm.pt)
    encoder = ProtenixEncoder.from_pretrained(
        pretrained_path="path/to/pretrained_dir",
    )

    # Encode directly from sequences
    sequences = ["MKFLILLFNILC", "MKTVRQERLKS"]  # multi-chain sequences
    s, esm_embedding = encoder(sequences)
    # s: [N_token, 384], esm_embedding: [N_token, 2560]

    # Or skip ESM and pass a precomputed ESM embedding
    s, _ = encoder(sequences=None, esm_embedding=precomputed_esm, input_feature_dict=...)

    # Project into the LLM embedding space
    projector = ProteinLLMProjector(protenix_dim=384, llm_dim=4096)
    llm_input = projector(s)  # [N_token, 4096]
"""

from proteor1.understand.protenix_encoder.modeling_protenix_encoder import (
    ProtenixEncoder,
    ESMEncoder,
    ProteinLLMProjector,
)
from proteor1.understand.protenix_encoder.processing_protenix_encoder import (
    ProtenixProcessor
)


__all__ = [
    "ProtenixEncoder",
    "ESMEncoder",
    "ProteinLLMProjector",
    "ProtenixProcessor",
]
