"""
ProteoR1 generate inference dataset.

This module wraps the underlying PredictionDataset and adds
text inputs (input_ids, chain_type_ids, cdr_region_type_ids) for the generate model.

The CDR sequences are extracted from:
- chain_infos["seq_gt"]: Ground truth H+L chain sequence
- chain_infos["spec_mask"]: Mask string where '1' indicates CDR positions
- features["region_type"]: Region labels (2=CDR1, 4=CDR2, 6=CDR3)
- features["chain_type"]: Chain labels (1=H, 2=L, 3=other)

Supports two modes:
1. Normal mode: Tokenize CDR sequences on-the-fly
2. Precomputed mode: Load precomputed CDR hidden states from .safetensors files
   - Files should be named {record_id}_SAMPLE_N.safetensors
   - Contains: cdr_hidden_states, cdr_chain_type, cdr_region_type, cdr_confidence
"""

import logging
import os
import re
from glob import glob
from typing import Any, Dict, List, Optional

import torch
from safetensors.torch import load_file as load_safetensors
from torch import Tensor
from torch.utils.data import Dataset

CDR_REGION_TYPES = {
    2: "CDR1",
    4: "CDR2",
    6: "CDR3",
}
CHAIN_TYPE_HEAVY = 1
CHAIN_TYPE_LIGHT = 2

logger = logging.getLogger(__name__)


class BagelPredictionDataset(Dataset):
    """
    ProteoR1 generate inference dataset.

    Wraps PredictionDataset and adds text inputs for the generate model.
    Extracts CDR sequences from chain_infos using spec_mask and entity_to_gt.

    Supports two modes:
    1. Normal mode (precomputed_cdr_dir=None): Tokenize CDR sequences on-the-fly
    2. Precomputed mode (precomputed_cdr_dir set): Load precomputed hidden states

    Parameters
    ----------
    prediction_dataset : Dataset
        The underlying PredictionDataset
    chain_infos : Dict[str, Any]
        Dictionary mapping record_id to chain info containing:
        - seq_gt: Ground truth H+L chain sequence
        - spec_mask: Mask string where '1' indicates CDR positions
        - entity_to_gt: Dictionary mapping entity_id to ground truth sequence
    processor : Any
        Qwen3 tokenizer/processor for encoding amino acids
    use_text_conditioning : bool
        Whether to include text conditioning. If False, returns empty text inputs.
    precomputed_cdr_dir : Optional[str]
        Path to precomputed CDR hidden states directory.
        Files should be named {record_id}_SAMPLE_N.safetensors.
        When provided, enables precomputed mode and filters to only records with files.
    """

    def __init__(
        self,
        prediction_dataset: Any,
        chain_infos: Dict[str, Any],
        processor: Any,
        use_text_conditioning: bool = True,
        precomputed_cdr_dir: Optional[str] = None,
    ) -> None:
        self.prediction_dataset = prediction_dataset
        self.chain_infos = chain_infos
        self.processor = processor
        self.use_text_conditioning = use_text_conditioning
        self.precomputed_cdr_dir = precomputed_cdr_dir

        # Precomputed mode state
        self._precomputed_file_map: Optional[Dict[str, List[str]]] = None
        self._filtered_indices: Optional[List[int]] = None

        # Initialize precomputed mode if directory is provided
        if precomputed_cdr_dir is not None:
            self._build_precomputed_mode()

    def _build_precomputed_mode(self) -> None:
        """
        Build precomputed mode by scanning for safetensors files and filtering records.

        This method:
        1. Scans precomputed_cdr_dir for {record_id}_SAMPLE_N.safetensors files
        2. Builds a mapping from record_id to list of file paths
        3. Filters prediction_dataset indices to only those with precomputed files

        Raises
        ------
        FileNotFoundError
            If precomputed_cdr_dir does not exist
        ValueError
            If no records have precomputed files
        """
        if self.precomputed_cdr_dir is None:
            return

        if not os.path.isdir(self.precomputed_cdr_dir):
            raise FileNotFoundError(
                f"Precomputed CDR directory does not exist: {self.precomputed_cdr_dir}"
            )

        # Scan for safetensors files
        pattern = os.path.join(self.precomputed_cdr_dir, "*.safetensors")
        all_files = sorted(glob(pattern))

        # Build file map: record_id -> list of file paths
        # Only support {record_id}_SAMPLE_N.safetensors format
        sample_pattern = re.compile(r"^(.+?)_SAMPLE_(\d+)\.safetensors$")
        self._precomputed_file_map = {}

        for file_path in all_files:
            filename = os.path.basename(file_path)
            match = sample_pattern.match(filename)
            if match:
                record_id = match.group(1)
                # Normalize to lowercase for consistent matching
                record_id_lower = record_id.lower()
                if record_id_lower not in self._precomputed_file_map:
                    self._precomputed_file_map[record_id_lower] = []
                self._precomputed_file_map[record_id_lower].append(file_path)

        logger.info(
            f"Precomputed CDR mode: found {len(self._precomputed_file_map)} unique record_ids "
            f"with {sum(len(v) for v in self._precomputed_file_map.values())} total files"
        )

        # Filter indices to only those with precomputed files
        self._filtered_indices = []
        for idx in range(len(self.prediction_dataset)):
            # Get record_id from the underlying dataset
            # Note: We access manifest.records directly to avoid loading full features
            record = self.prediction_dataset.manifest.records[idx]
            record_id_lower = record.id.lower()

            if record_id_lower in self._precomputed_file_map:
                self._filtered_indices.append(idx)

        records_before = len(self.prediction_dataset)
        records_after = len(self._filtered_indices)

        logger.info(
            f"Precomputed CDR filtering: {records_before} -> {records_after} records "
            f"({records_before - records_after} records filtered out)"
        )

        if records_after == 0:
            raise ValueError(
                f"No records remaining after precomputed CDR filtering. "
                f"Check that precomputed files ({len(self._precomputed_file_map)} record_ids) "
                f"match record IDs in manifest ({records_before} records)."
            )

    def _build_precomputed_item(self, record_id: str, sample_idx: int = 0) -> Dict[str, Tensor]:
        """
        Load precomputed CDR hidden states from safetensors file.

        Parameters
        ----------
        record_id : str
            The record identifier (e.g., "7df1_H_L_D")
        sample_idx : int
            Which sample file to load (default 0 = first/only sample)

        Returns
        -------
        Dict[str, Tensor]
            Dictionary containing:
            - precomputed_hidden_states: [N_cdr, hidden_dim]
            - attention_mask: [N_cdr]
            - chain_type_ids: [N_cdr]
            - cdr_region_type_ids: [N_cdr]

        Raises
        ------
        KeyError
            If record_id is not in the precomputed file map
        IndexError
            If sample_idx is out of range for this record
        RuntimeError
            If the precomputed file is corrupted or missing required keys
        """
        record_id_lower = record_id.lower()

        if self._precomputed_file_map is None:
            raise RuntimeError("Precomputed file map not initialized")

        if record_id_lower not in self._precomputed_file_map:
            raise KeyError(
                f"Record '{record_id}' not found in precomputed file map. "
                f"This indicates a bug in filtering logic."
            )

        file_list = self._precomputed_file_map[record_id_lower]
        if sample_idx >= len(file_list):
            raise IndexError(
                f"Sample index {sample_idx} out of range for record '{record_id}' "
                f"(only {len(file_list)} samples available)"
            )

        file_path = file_list[sample_idx]
        logger.debug(f"Loading precomputed CDR from: {file_path}")

        # Load with error handling
        try:
            data = load_safetensors(file_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load precomputed CDR file '{file_path}': {e}"
            ) from e

        # Validate required keys
        required_keys = ["cdr_hidden_states", "cdr_chain_type", "cdr_region_type", "cdr_confidence"]
        missing_keys = [k for k in required_keys if k not in data]
        if missing_keys:
            raise RuntimeError(
                f"Precomputed CDR file '{file_path}' is missing required keys: {missing_keys}"
            )

        hidden_states = data["cdr_hidden_states"]
        n_cdr = hidden_states.shape[0]

        return {
            "precomputed_hidden_states": hidden_states,
            "attention_mask": torch.ones(n_cdr, dtype=torch.long),
            "chain_type_ids": data["cdr_chain_type"].long(),
            "cdr_region_type_ids": data["cdr_region_type"].long(),
            "cdr_confidence": data["cdr_confidence"].float(),
        }

    def __len__(self) -> int:
        # In precomputed mode, return filtered length
        if self._filtered_indices is not None:
            return len(self._filtered_indices)
        return len(self.prediction_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get an item from the dataset.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - text: Dict with either:
              - Normal mode: input_ids, attention_mask, chain_type_ids, cdr_region_type_ids
              - Precomputed mode: precomputed_hidden_states, attention_mask, chain_type_ids, cdr_region_type_ids
            - boltz: Dict with original PredictionDataset features
        """
        # In precomputed mode, map idx through filtered indices
        if self._filtered_indices is not None:
            actual_idx = self._filtered_indices[idx]
        else:
            actual_idx = idx

        # Get original boltz features
        boltz_features = self.prediction_dataset[actual_idx]
        record = boltz_features["record"]
        record_id = record.id

        if not self.use_text_conditioning:
            # Return empty text inputs if text conditioning is disabled
            return {
                "text": self._build_empty_text_inputs(),
                "boltz": boltz_features,
            }

        # Check if using precomputed mode
        if self._precomputed_file_map is not None:
            # Load precomputed CDR hidden states
            text_inputs = self._build_precomputed_item(record_id, sample_idx=0)
            return {
                "text": text_inputs,
                "boltz": boltz_features,
            }

        # Normal mode: tokenize CDR sequences on-the-fly

        # Get chain info for this record
        if record_id not in self.chain_infos:
            logger.warning(f"Record {record_id} not found in chain_infos, using empty text inputs")
            return {
                "text": self._build_empty_text_inputs(),
                "boltz": boltz_features,
            }

        info = self.chain_infos[record_id]
        seq_gt = info.get("seq_gt")
        spec_mask = info.get("spec_mask")

        if seq_gt is None or spec_mask is None:
            logger.warning(f"Record {record_id} missing seq_gt or spec_mask, using empty text inputs")
            return {
                "text": self._build_empty_text_inputs(),
                "boltz": boltz_features,
            }

        # Extract CDR sequences using spec_mask + region_type + chain_type
        cdr_sequences = self._extract_cdr_from_gt(
            seq_gt=seq_gt,
            spec_mask=spec_mask,
            region_type=boltz_features["region_type"],
            chain_type=boltz_features["chain_type"],
        )

        # Build text inputs
        text_inputs = self._build_text_inputs(cdr_sequences)

        return {
            "text": text_inputs,
            "boltz": boltz_features,
        }

    def _extract_cdr_from_gt(
        self,
        seq_gt: str,
        spec_mask: str,
        region_type: Tensor,
        chain_type: Tensor,
    ) -> Dict[str, str]:
        """
        Extract CDR sequences from ground truth sequence using spec_mask.

        The spec_mask marks CDR positions with '1'. We use region_type and chain_type
        from features to determine which CDR region each position belongs to.

        Parameters
        ----------
        seq_gt : str
            Ground truth H+L chain concatenated sequence
        spec_mask : str
            Mask string where '1' indicates CDR positions
        region_type : Tensor
            Region labels tensor (2=CDR1, 4=CDR2, 6=CDR3)
        chain_type : Tensor
            Chain labels tensor (1=H, 2=L, 3=other)

        Returns
        -------
        Dict[str, str]
            Dictionary with keys: HCDR1, HCDR2, HCDR3, LCDR1, LCDR2, LCDR3
        """
        cdr_sequences = {
            "HCDR1": "",
            "HCDR2": "",
            "HCDR3": "",
            "LCDR1": "",
            "LCDR2": "",
            "LCDR3": "",
        }

        # Validate lengths
        # Note: spec_mask and seq_gt are for H+L chains only (not antigen)
        # region_type and chain_type include all tokens (H+L+antigen)
        seq_len = len(seq_gt)

        if len(spec_mask) != seq_len:
            logger.warning(
                f"spec_mask length ({len(spec_mask)}) != seq_gt length ({seq_len}), "
                "CDR extraction may be incorrect"
            )

        # Count antibody tokens (H chain + L chain only)
        ab_token_count = 0
        for i in range(len(chain_type)):
            if chain_type[i].item() in [CHAIN_TYPE_HEAVY, CHAIN_TYPE_LIGHT]:
                ab_token_count += 1

        if ab_token_count != seq_len:
            logger.debug(
                f"Antibody token count ({ab_token_count}) != seq_gt length ({seq_len}), "
                "this is expected if features are padded or cropped differently"
            )

        # Build position mapping: for each position in spec_mask/seq_gt,
        # find the corresponding position in region_type/chain_type
        ab_idx = 0  # Index into seq_gt / spec_mask
        for feat_idx in range(len(chain_type)):
            chain = chain_type[feat_idx].item()

            # Skip non-antibody chains
            if chain not in [CHAIN_TYPE_HEAVY, CHAIN_TYPE_LIGHT]:
                continue

            # Check if we've exhausted seq_gt positions
            if ab_idx >= seq_len:
                break

            # Check if this position is a CDR (marked in spec_mask)
            if spec_mask[ab_idx] == '1':
                aa = seq_gt[ab_idx]
                region = region_type[feat_idx].item()

                # Determine CDR name from region_type
                region_int = int(region)
                if region_int in CDR_REGION_TYPES:
                    cdr_name = CDR_REGION_TYPES[region_int]  # CDR1, CDR2, or CDR3
                    chain_prefix = "H" if chain == CHAIN_TYPE_HEAVY else "L"
                    full_cdr_name = f"{chain_prefix}{cdr_name}"
                    cdr_sequences[full_cdr_name] += aa

            ab_idx += 1

        return cdr_sequences

    def _build_text_inputs(self, cdr_sequences: Dict[str, str]) -> Dict[str, Tensor]:
        """
        Build text inputs from CDR sequences.

        This follows the same chain-info filtering logic used during training.

        Parameters
        ----------
        cdr_sequences : Dict[str, str]
            Dictionary with CDR sequences (HCDR1, HCDR2, HCDR3, LCDR1, LCDR2, LCDR3)

        Returns
        -------
        Dict[str, Tensor]
            Dictionary containing:
            - input_ids: Token IDs for CDR amino acids
            - attention_mask: Attention mask (all 1s for valid positions)
            - chain_type_ids: Chain type for each position (1=H, 2=L)
            - cdr_region_type_ids: CDR region type for each position (2=CDR1, 4=CDR2, 6=CDR3)
        """
        input_ids = []
        attention_mask = []
        chain_type_ids = []
        cdr_region_type_ids = []

        # CDR region type mapping (inverse of CDR_REGION_TYPES)
        cdr_name_to_region = {"CDR1": 2, "CDR2": 4, "CDR3": 6}

        for key in ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]:
            cdr_seq = cdr_sequences.get(key, "")
            if not cdr_seq:
                continue

            cdr_len = len(cdr_seq)

            # Determine chain type
            if key[0] == "H":
                chain_type_ids.extend([CHAIN_TYPE_HEAVY] * cdr_len)
            else:
                chain_type_ids.extend([CHAIN_TYPE_LIGHT] * cdr_len)

            # Determine CDR region type
            cdr_region = cdr_name_to_region[key[1:]]  # Extract "CDR1", "CDR2", or "CDR3"
            cdr_region_type_ids.extend([cdr_region] * cdr_len)

            # Attention mask
            attention_mask.extend([1] * cdr_len)

            # Tokenize each amino acid
            for aa in cdr_seq:
                aa_token_ids = self.processor.encode(aa, add_special_tokens=False)
                if len(aa_token_ids) >= 1:
                    input_ids.append(aa_token_ids[0])
                else:
                    logger.warning(f"Tokenizer produced no tokens for '{aa}'")
                    # Use a fallback - pad token or first token
                    input_ids.append(self.processor.pad_token_id or 0)

        # Handle empty case
        if len(input_ids) == 0:
            return self._build_empty_text_inputs()

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "chain_type_ids": torch.tensor(chain_type_ids, dtype=torch.long),
            "cdr_region_type_ids": torch.tensor(cdr_region_type_ids, dtype=torch.long),
        }

    def _build_empty_text_inputs(self) -> Dict[str, Tensor]:
        """Build empty text inputs when text conditioning is disabled."""
        return {
            "input_ids": torch.tensor([], dtype=torch.long),
            "attention_mask": torch.tensor([], dtype=torch.long),
            "chain_type_ids": torch.tensor([], dtype=torch.long),
            "cdr_region_type_ids": torch.tensor([], dtype=torch.long),
        }
