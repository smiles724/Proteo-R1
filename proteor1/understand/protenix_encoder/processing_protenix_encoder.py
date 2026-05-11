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
ProtenixProcessor - HuggingFace-style protein-sequence processor.

Produces a full input_feature_dict containing:
1. Protenix features (from SampleDictToFeatures).
2. ESM tokenization results (esm_input_ids, esm_attention_mask, etc.).

Usage:
    processor = ProtenixProcessor.from_pretrained("pretrained/protenix_encoder")

    # Process a JSON entry
    input_feature_dict = processor(json_entry)

    # Batch collate
    batch = protenix_collate_fn([input_feature_dict1, input_feature_dict2])

    # Feed into ProtenixEncoder
    s, esm_emb = encoder(batch["input_feature_dict"])
"""
import json
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from protenix.data.data_pipeline import DataPipeline
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.utils import data_type_transform, make_dummy_feature


@dataclass
class ProtenixProcessorOutput:
    """Processor output."""
    input_feature_dict: dict[str, torch.Tensor]
    atom_array: Any  # biotite AtomArray
    token_array: Any  # TokenArray
    metadata: dict[str, Any] = field(default_factory=dict)


class ProtenixProcessor:
    """
    Protenix sequence processor.

    Converts a JSON entry into an input_feature_dict that ProtenixEncoder can consume directly.

    input_feature_dict contains:
    - All Protenix features (token_index, asym_id, ref_pos, ...).
    - ESM tokenization (esm_input_ids, esm_attention_mask, esm_sequence_lengths, ...).
    - ESM auxiliary info (esm_entity_id_to_sequence, esm_unique_sequences, ...).
    """

    # Prefix for ESM-related keys; avoids collisions with Protenix features.
    ESM_PREFIX = "esm_"

    def __init__(
        self,
        esm_embedding_dim: int = 2560,
        esm_truncation_seq_length: int = 4094,
        esm_alphabet=None,
    ):
        """
        Initialize the Processor.

        Args:
            esm_embedding_dim: ESM embedding dimensionality.
            esm_truncation_seq_length: ESM sequence truncation length.
            esm_alphabet: ESM alphabet (used for tokenization).
        """
        self.esm_embedding_dim = esm_embedding_dim
        self.esm_truncation_seq_length = esm_truncation_seq_length
        self.esm_alphabet = esm_alphabet
        self.esm_batch_converter = None

        if esm_alphabet is not None:
            self.esm_batch_converter = esm_alphabet.get_batch_converter(
                esm_truncation_seq_length
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        init_esm_tokenizer: bool = True,
    ) -> "ProtenixProcessor":
        """
        Load the Processor from a pretrained directory.

        Uses the same directory layout as ProtenixEncoder.from_pretrained():
            pretrained_path/
            ├── config.json
            ├── protenix_mini_ism_v0.5.0.pt
            └── esm2_t36_3B_UR50D_ism.pt

        Args:
            pretrained_path: pretrained directory path
            init_esm_tokenizer: whether to initialize the ESM tokenizer

        Returns:
            ProtenixProcessor instance
        """
        if not os.path.isdir(pretrained_path):
            raise FileNotFoundError(f"Pretrained directory not found: {pretrained_path}")

        config_path = os.path.join(pretrained_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = json.load(f)

        esm_config = config_dict.get("esm", {})
        esm_embedding_dim = esm_config.get("embedding_dim", 2560)

        # Initialize the ESM tokenizer.
        esm_alphabet = None
        if init_esm_tokenizer:
            try:
                import esm
                esm_alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
                print(f"Initialized ESM tokenizer (vocab_size={len(esm_alphabet)})")
            except ImportError:
                print("Warning: esm package not found, ESM tokenization disabled")

        processor = cls(
            esm_embedding_dim=esm_embedding_dim,
            esm_truncation_seq_length=4094,
            esm_alphabet=esm_alphabet,
        )

        print(f"Loaded ProtenixProcessor from {pretrained_path}")
        return processor

    def _extract_protein_info(
        self,
        json_entry: dict[str, Any],
    ) -> tuple[list[str], dict[str, str]]:
        """
        Extract protein-sequence info from a JSON entry.

        Returns:
            unique_sequences: list of unique protein sequences
            entity_id_to_sequence: entity_id -> sequence mapping
        """
        entity_id_to_sequence = {}

        for i, entity_wrapper in enumerate(json_entry["sequences"]):
            entity_id = str(i + 1)
            entity_type = list(entity_wrapper.keys())[0]
            entity_info = entity_wrapper[entity_type]

            if entity_type == "proteinChain":
                entity_id_to_sequence[entity_id] = entity_info["sequence"]

        # Order-preserving unique sequences.
        unique_sequences = list(dict.fromkeys(entity_id_to_sequence.values()))

        return unique_sequences, entity_id_to_sequence

    def _tokenize_sequences(
        self,
        sequences: list[str],
    ) -> dict[str, Any]:
        """
        ESM tokenization.

        Args:
            sequences: list of protein sequences

        Returns:
            {
                "input_ids": Tensor [n_seq, max_len+2],
                "attention_mask": Tensor [n_seq, max_len+2],
                "sequence_lengths": list[int],
            }
        """
        if len(sequences) == 0:
            return {
                "input_ids": torch.zeros(0, 2, dtype=torch.long),
                "attention_mask": torch.zeros(0, 2, dtype=torch.long),
                "sequence_lengths": [],
            }

        if self.esm_batch_converter is None:
            raise RuntimeError("ESM tokenizer not initialized")

        labels = [f"seq_{i}" for i in range(len(sequences))]
        _, strs, tokens = self.esm_batch_converter(list(zip(labels, sequences)))

        attention_mask = (tokens != self.esm_alphabet.padding_idx).long()
        sequence_lengths = [
            min(self.esm_truncation_seq_length, len(s)) for s in sequences
        ]

        return {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "sequence_lengths": sequence_lengths,
        }

    def _compute_input_feature_dict(
        self,
        json_entry: dict[str, Any],
        ref_pos_augment: bool = True,
    ) -> tuple[dict[str, torch.Tensor], Any, Any, dict]:
        """
        Compute the Protenix input_feature_dict.

        **Fully aligned with the upstream InferenceDataset.process_one().**

        Args:
            json_entry: Protenix JSON-format entry
            ref_pos_augment: Whether to apply random augmentation to reference positions.
                When True (default), random rotation and translation are applied.
                When False, only centering is applied (deterministic).
        """
        # Step 1: SampleDictToFeatures
        sample2feat = SampleDictToFeatures(json_entry, ref_pos_augment=ref_pos_augment)
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()

        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()

        entity_poly_type = sample2feat.entity_poly_type

        # Step 2: entity_to_asym_id mapping.
        entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)

        # Step 3: Dummy features.
        dummy_feats = ["template", "msa"]
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # Step 4: Data-type conversion.
        features_dict = data_type_transform(feat_or_label_dict=features_dict)

        extra_info = {
            "entity_poly_type": entity_poly_type,
            "entity_to_asym_id": dict(entity_to_asym_id),
        }

        return features_dict, atom_array, token_array, extra_info

    def get_position_info(
        self,
        json_entry: dict[str, Any],
        ref_pos_augment: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Return only the position info (used by protenix_pos_embed).

        This lightweight helper only computes the position-related fields of the Protenix features.
        It skips ESM tokenization, which is what you want in precomputed mode to obtain
        residue_index and asym_id.

        Args:
            json_entry: Protenix JSON-format entry
            ref_pos_augment: Whether to apply random augmentation to reference positions.

        Returns:
            {
                "residue_index": [N_token] residue index within the chain
                "asym_id": [N_token] chain ID
                "n_token": int, number of protein tokens
            }
        """
        features_dict, _, _, _ = self._compute_input_feature_dict(
            json_entry, ref_pos_augment=ref_pos_augment
        )

        return {
            "residue_index": features_dict["residue_index"],  # [N_token]
            "asym_id": features_dict["asym_id"],  # [N_token]
            "n_token": features_dict["residue_index"].shape[0],
        }

    def process_from_cif(
        self,
        cif_path: str,
        assembly_id: str = "1",
        ref_pos_augment: bool = False,
        return_atom_array: bool = True,
        return_token_array: bool = True,
        use_bioassembly: bool = True,
        protein_only: bool = False,
    ) -> ProtenixProcessorOutput:
        """
        Process a protein structure directly from a CIF file (used for feature dumping).

        This is the recommended path for feature dumping because it:
        1. Uses the crystal structure's actual GT coordinates.
        2. Guarantees atom-order consistency.
        3. Avoids coordinate-alignment issues between the JSON and CIF pipelines.

        Pipeline:
            CIF file -> MMCIFParser.get_bioassembly() / DistillationMMCIFParser.get_structure_dict()
                     -> atom_array (GT coords)
                     -> AtomArrayTokenizer -> token_array
                     -> Featurizer -> features_dict

        Args:
            cif_path: CIF file path (supports .gz)
            assembly_id: Bioassembly ID (default "1"), only effective when use_bioassembly=True
            ref_pos_augment: Whether to apply random augmentation to ref_pos.
                False (default) means centering only (deterministic), recommended for feature dumping.
            return_atom_array: whether to return atom_array
            return_token_array: whether to return token_array
            use_bioassembly: whether to use bioassembly mode (default True)
                True: use MMCIFParser.get_bioassembly() to expand symmetric chains and apply
                      more filters (water, hydrogen, unknown residues, far-away C-alpha, crystallization
                      auxiliaries); has chain-count limits (max 20 chains, 5120 tokens).
                False: use DistillationMMCIFParser.get_structure_dict() to read the asymmetric unit
                       only with minimal filters (fix_arginine, add_missing_atoms); no chain-count limit,
                       suitable for distillation data.
            protein_only: whether to keep only protein atoms (default False)
                True: filter out non-protein atoms (ligands, DNA/RNA, ions, etc.) before featurization.
                      Raises ValueError when no protein atoms remain.
                False: keep every atom type.

        Returns:
            ProtenixProcessorOutput:
                - input_feature_dict: Protenix features + ESM tokenization
                - atom_array: Biotite AtomArray
                - token_array: TokenArray
                - metadata: metadata dict
        """
        from protenix.data.parser import DistillationMMCIFParser, MMCIFParser
        from protenix.data.tokenizer import AtomArrayTokenizer
        from protenix.data.featurizer import Featurizer
        from protenix.data.utils import data_type_transform, make_dummy_feature

        # Step 1: Parse the CIF file (MMCIFParser natively supports .gz).
        # Pick the parser and method based on use_bioassembly.
        if use_bioassembly:
            parser = MMCIFParser(cif_path)
            bioassembly = parser.get_bioassembly(assembly_id=assembly_id, expand_assembly=False)
            actual_assembly_id = assembly_id
        else:
            parser = DistillationMMCIFParser(mmcif_file=cif_path)
            bioassembly = parser.get_structure_dict()
            actual_assembly_id = None  # DistillationMMCIFParser does not use assembly_id

        atom_array = bioassembly.get("atom_array")
        if atom_array is None or len(atom_array) == 0:
            raise ValueError(f"No valid atom_array from CIF: {cif_path}")

        # Early protein filtering (before featurization)
        if protein_only:
            # Prefer chain_mol_type (semantic molecule type) over hetero (HETATM flag)
            # chain_mol_type is set by Protenix parser and is consistent with ESMFeaturizer
            if hasattr(atom_array, "chain_mol_type"):
                protein_mask = atom_array.chain_mol_type == "protein"
            else:
                # Fallback to hetero for non-Protenix parsed structures
                protein_mask = ~atom_array.hetero
            atom_array = atom_array[protein_mask]
            if len(atom_array) == 0:
                raise ValueError(f"No protein atoms found after filtering in CIF: {cif_path}")

        # Step 2: Build token_array from atom_array.
        tokenizer = AtomArrayTokenizer(atom_array)
        token_array = tokenizer.get_token_array()

        # Step 3: Extract features via Featurizer (matches the training pipeline).
        # NOTE: include_discont_poly_poly_bonds=False matches the training default in dataset.py.
        featurizer = Featurizer(
            token_array,
            atom_array,
            ref_pos_augment=ref_pos_augment,
            include_discont_poly_poly_bonds=False,
        )
        features_dict = featurizer.get_all_input_features()

        # Step 4: Create dummy features for template and MSA.
        dummy_feats = ["template", "msa"]
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # Step 5: Data-type conversion.
        features_dict = data_type_transform(feat_or_label_dict=features_dict)

        # Step 6: Extract protein sequences from bioassembly + entity_poly_type.
        entity_id_to_sequence = {}
        sequences = bioassembly.get("sequences", {})
        entity_poly_type = parser.entity_poly_type

        for entity_id, seq in sequences.items():
            # Only protein chains carry ESM embeddings.
            if entity_id in entity_poly_type and "polypeptide" in entity_poly_type.get(entity_id, ""):
                entity_id_to_sequence[entity_id] = seq

        # Order-preserving unique sequences.
        unique_sequences = list(dict.fromkeys(entity_id_to_sequence.values()))

        # Step 7: ESM tokenization (reuses self._tokenize_sequences()).
        if self.esm_batch_converter is not None and len(unique_sequences) > 0:
            esm_tokens = self._tokenize_sequences(unique_sequences)

            features_dict["esm_input_ids"] = esm_tokens["input_ids"]
            features_dict["esm_attention_mask"] = esm_tokens["attention_mask"]
            features_dict["esm_sequence_lengths"] = esm_tokens["sequence_lengths"]
            features_dict["esm_unique_sequences"] = unique_sequences
            features_dict["esm_entity_id_to_sequence"] = entity_id_to_sequence
        else:
            features_dict["esm_input_ids"] = None
            features_dict["esm_attention_mask"] = None
            features_dict["esm_sequence_lengths"] = []
            features_dict["esm_unique_sequences"] = []
            features_dict["esm_entity_id_to_sequence"] = {}

        features_dict["esm_embedding_dim"] = self.esm_embedding_dim

        # Step 8: Build metadata.
        N_token = features_dict["token_index"].shape[0]
        N_atom = features_dict["atom_to_token_idx"].shape[0]
        N_msa = features_dict["msa"].shape[0]
        N_asym = len(torch.unique(features_dict["asym_id"]))

        # Extract a name from the CIF filename.
        import os
        basename = os.path.basename(cif_path)
        name = basename.split("-")[0].split(".")[0].upper()

        metadata = {
            "name": name,
            "cif_path": cif_path,
            "assembly_id": actual_assembly_id,
            "use_bioassembly": use_bioassembly,
            "protein_only": protein_only,
            "N_token": N_token,
            "N_atom": N_atom,
            "N_msa": N_msa,
            "N_asym": N_asym,
            "N_protein_sequences": len(unique_sequences),
            "entity_poly_type": entity_poly_type,
            "coord_source": "cif_gt",
        }

        return ProtenixProcessorOutput(
            input_feature_dict=features_dict,
            atom_array=atom_array if return_atom_array else None,
            token_array=token_array if return_token_array else None,
            metadata=metadata,
        )

    def __call__(
        self,
        json_entry: dict[str, Any],
        return_atom_array: bool = True,
        return_token_array: bool = True,
        ref_pos_augment: bool = True,
    ) -> ProtenixProcessorOutput:
        """
        Process a JSON entry and produce a complete input_feature_dict.

        Args:
            json_entry: Protenix JSON-format entry
            return_atom_array: whether to return atom_array
            return_token_array: whether to return token_array
            ref_pos_augment: Whether to apply random augmentation to reference positions.
                When True (default), random rotation and translation are applied.
                When False, only centering is applied (deterministic), recommended
                for feature dumping where reproducibility is required.

        Returns:
            ProtenixProcessorOutput:
                - input_feature_dict: Protenix features + ESM tokenization
                - atom_array: Biotite AtomArray
                - token_array: TokenArray
                - metadata: metadata dict
        """
        # 1. Compute Protenix features.
        features_dict, atom_array, token_array, extra_info = \
            self._compute_input_feature_dict(json_entry, ref_pos_augment=ref_pos_augment)

        # 2. Extract protein sequences and tokenize.
        unique_sequences, entity_id_to_sequence = self._extract_protein_info(json_entry)

        if self.esm_batch_converter is not None and len(unique_sequences) > 0:
            esm_tokens = self._tokenize_sequences(unique_sequences)

            # Add to features_dict (esm_ prefix).
            features_dict["esm_input_ids"] = esm_tokens["input_ids"]
            features_dict["esm_attention_mask"] = esm_tokens["attention_mask"]
            features_dict["esm_sequence_lengths"] = esm_tokens["sequence_lengths"]  # list, not a tensor
            features_dict["esm_unique_sequences"] = unique_sequences  # list
            features_dict["esm_entity_id_to_sequence"] = entity_id_to_sequence  # dict
        else:
            # No ESM tokenizer, or no protein sequences.
            features_dict["esm_input_ids"] = None
            features_dict["esm_attention_mask"] = None
            features_dict["esm_sequence_lengths"] = []
            features_dict["esm_unique_sequences"] = []
            features_dict["esm_entity_id_to_sequence"] = {}

        # 3. Add auxiliary info for ESM embedding fill.
        # ProtenixEncoder uses this to map ESM embeddings to the right token positions.
        features_dict["esm_embedding_dim"] = self.esm_embedding_dim

        # 4. Metadata.
        N_token = features_dict["token_index"].shape[0]
        N_atom = features_dict["atom_to_token_idx"].shape[0]
        N_msa = features_dict["msa"].shape[0]
        N_asym = len(torch.unique(features_dict["asym_id"]))

        metadata = {
            "name": json_entry["name"],
            "N_token": N_token,
            "N_atom": N_atom,
            "N_msa": N_msa,
            "N_asym": N_asym,
            "N_protein_sequences": len(unique_sequences),
            "entity_poly_type": extra_info["entity_poly_type"],
            "entity_to_asym_id": extra_info["entity_to_asym_id"],
        }

        return ProtenixProcessorOutput(
            input_feature_dict=features_dict,
            atom_array=atom_array if return_atom_array else None,
            token_array=token_array if return_token_array else None,
            metadata=metadata,
        )


def protenix_collate_fn(
    batch: list[ProtenixProcessorOutput],
) -> dict[str, Any]:
    """
    Batch collate function.

    Matches the upstream Protenix collate_fn_first: does not add a batch dim.
    Protenix indexes with [..., N_token, ...] and supports arbitrary leading dims.

    Note: Protenix's pair features are O(N^2), so this supports only batch_size=1.

    Args:
        batch: list of ProtenixProcessorOutput

    Returns:
        {
            "input_feature_dict": features (no added batch dim),
            "atom_array": AtomArray,
            "token_array": TokenArray,
            "metadata": metadata dict,
        }
    """
    if len(batch) == 0:
        raise ValueError("Empty batch")

    if len(batch) != 1:
        raise NotImplementedError(
            "Batch size > 1 is not supported for Protenix due to O(N²) memory. "
            "Use batch_size=1."
        )

    # Matches the upstream collate_fn_first: return the first sample directly, no added batch dim.
    item = batch[0]
    return {
        "input_feature_dict": item.input_feature_dict,
        "atom_array": item.atom_array,
        "token_array": item.token_array,
        "metadata": item.metadata,
    }
