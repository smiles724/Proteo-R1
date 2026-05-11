#!/usr/bin/env python3
"""
CDR Evaluation Pipeline - Stage 2: CIF to JSON + CDR Masking

This module handles:
1. Converting CIF files to Protenix input JSON format
2. Identifying CDR regions using abnumber library (Chothia scheme)
3. Masking CDR residues in antibody chains for structure prediction

Usage:
    from proteor1.cdr_eval.cdr_masking import (
        cif_to_protenix_json,
        get_cdr_indices_from_sequence,
        mask_sequence,
        extract_and_mask_cdr,
    )

Notes:
    - Uses Chothia numbering scheme for CDR definition
    - Handles both heavy and light chains
    - Preserves antigen chains without modification
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from tqdm import tqdm

from proteor1.cdr_eval import load_entries_for_split, EntryInfo

from protenix.data.json_maker import cif_to_input_json
from protenix.data.utils import pdb_to_cif

logger = logging.getLogger(__name__)


# Chothia CDR region names
CHOTHIA_CDR_REGIONS = ["cdr1", "cdr2", "cdr3"]


class CDRRegionInfo(TypedDict):
    """Information about a single CDR region."""
    region_name: str
    indices: list[int]
    sequence: str


class ChainCDRInfo(TypedDict):
    """CDR information for a single antibody chain."""
    chain_type: str  # "H" or "L"
    original_seq: str
    masked_seq: str
    cdr_indices: list[int]  # Full sequence indices (0-based)
    domain_start: int  # offset of the variable domain in the full sequence (used to convert to variable-domain indices)
    cdr1: CDRRegionInfo
    cdr2: CDRRegionInfo
    cdr3: CDRRegionInfo
    error: str | None


@dataclass
class CDRMaskingResult:
    """Result of CDR masking operation on an entry."""
    entry_name: str
    json_data: list[dict]
    heavy_chain_info: ChainCDRInfo | None = None
    light_chain_info: ChainCDRInfo | None = None
    matched_heavy_entity: int | None = None
    matched_light_entity: int | None = None
    success: bool = True
    error_message: str | None = None


def cif_to_protenix_json(
    cif_path: str | Path,
    sample_name: str | None = None,
    save_entity_and_asym_id: bool = True,
    get_entity_seq_with_coords: bool = False
) -> dict:
    """
    Convert CIF file to Protenix input JSON format.

    Args:
        cif_path: Path to the CIF file
        sample_name: Name for the sample in JSON (default: CIF filename stem)
        save_entity_and_asym_id: Whether to save entity and chain ID mappings

    Returns:
        Protenix input JSON dictionary

    Raises:
        FileNotFoundError: If CIF file does not exist
        RuntimeError: If conversion fails
    """
    cif_path = Path(cif_path)

    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file not found: {cif_path}")

    if sample_name is None:
        sample_name = cif_path.stem

    if str(cif_path).endswith(".pdb"):
        with tempfile.NamedTemporaryFile(suffix=".cif") as tmp:
            tmp_cif_file = tmp.name
            pdb_to_cif(str(cif_path), tmp_cif_file)
            json_dict = cif_to_input_json(
                tmp_cif_file,
                sample_name=sample_name,
                save_entity_and_asym_id=save_entity_and_asym_id,
                get_entity_seq_with_coords=get_entity_seq_with_coords
            )
            return json_dict
    else:
        try:
            json_dict = cif_to_input_json(
                mmcif_file=str(cif_path),
                sample_name=sample_name,
                save_entity_and_asym_id=save_entity_and_asym_id,
                get_entity_seq_with_coords=get_entity_seq_with_coords
            )
            return json_dict
        except Exception as e:
            raise RuntimeError(f"Failed to convert CIF to JSON: {cif_path}") from e


def _get_entity_count(seq_item: dict) -> int:
    """Get the count of an entity from a sequence item."""
    for etype in ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]:
        if etype in seq_item:
            return seq_item[etype].get("count", 1)
    return 1


def filter_json_by_chains(
    json_data: dict,
    chain_ids: list[str],
    protein_only: bool = True,
    case_insensitive: bool = True,
) -> dict:
    """
    Filter JSON sequences to only keep entities containing specified chain IDs.

    When a PDB structure contains multiple antibody-antigen complexes,
    this function extracts only the chains relevant to a specific entry.

    Args:
        json_data: Protenix JSON dictionary (must have save_entity_and_asym_id=True)
        chain_ids: List of chain IDs to keep (e.g., ["D", "E", "G"])
        protein_only: If True, only keep proteinChain entities, filtering out
            ligands, ions, dnaSequence, and rnaSequence (default: True)
        case_insensitive: If True, perform case-insensitive chain ID matching (default: True)

    Returns:
        Filtered JSON dictionary with only matching sequences

    Notes:
        - Matches only against auth_asym_id (PDB author chain IDs)
        - User-provided chain_ids are expected to be PDB author chain IDs
        - For kept entities, the count is adjusted to reflect only matching chains
        - Covalent bonds are filtered to only include bonds between kept entities
        - Bond copy indices are remapped when entity copies are filtered
    """
    import copy

    # Build case-normalized set for matching
    if case_insensitive:
        chain_id_set = set(c.upper() for c in chain_ids)
    else:
        chain_id_set = set(chain_ids)
    filtered_sequences = []
    kept_entity_indices = []  # Track which entities are kept (for bond filtering)

    # Track copy index remapping for each kept entity
    # Maps: (old_entity_idx_1based, old_copy_idx_1based) -> new_copy_idx_1based
    entity_copy_remap: dict[tuple[int, int], int] = {}

    for idx, seq_item in enumerate(json_data.get("sequences", [])):
        # Get the entity type (proteinChain, dnaSequence, rnaSequence, ligand, ion)
        entity_type = None
        entity_data = None
        for etype in ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]:
            if etype in seq_item:
                entity_type = etype
                entity_data = seq_item[etype]
                break

        if entity_data is None:
            continue

        # Filter out non-protein entities if protein_only is True
        if protein_only and entity_type != "proteinChain":
            continue

        # Get auth_asym_id and label_asym_id
        auth_asym_ids = entity_data.get("auth_asym_id", [])
        label_asym_ids = entity_data.get("label_asym_id", [])

        # If no auth_asym_id is available, keep the entity as-is
        if not auth_asym_ids:
            logger.warning(f"Entity {idx} has no auth_asym_id, keeping it")
            filtered_sequences.append(copy.deepcopy(seq_item))
            kept_entity_indices.append(idx)
            # For entities without chain IDs, assume single copy maps to itself
            entity_copy_remap[(idx + 1, 1)] = 1
            continue

        # Find matching chains - only match against auth_asym_id
        # (user-provided chain_ids are PDB author chain IDs)
        matching_indices = []
        for i, chain_id in enumerate(auth_asym_ids):
            compare_id = chain_id.upper() if case_insensitive else chain_id
            if compare_id in chain_id_set:
                matching_indices.append(i)

        if matching_indices:
            # Keep this entity, but adjust count to reflect only matching chains
            new_seq_item = copy.deepcopy(seq_item)
            new_seq_item[entity_type]["count"] = len(matching_indices)

            # Update both auth_asym_id and label_asym_id lists
            if auth_asym_ids:
                new_seq_item[entity_type]["auth_asym_id"] = [auth_asym_ids[i] for i in matching_indices]
            if label_asym_ids:
                new_seq_item[entity_type]["label_asym_id"] = [label_asym_ids[i] for i in matching_indices]

            filtered_sequences.append(new_seq_item)
            kept_entity_indices.append(idx)

            # Build copy index remapping for this entity
            # Original: copies [1, 2, 3, ...]
            # Filtered: matching copies with new indices [1, 2, ...]
            old_entity_id = idx + 1  # 1-based entity ID
            for new_copy_idx, old_chain_idx in enumerate(matching_indices):
                old_copy_idx = old_chain_idx + 1  # Original copy index (1-based)
                entity_copy_remap[(old_entity_id, old_copy_idx)] = new_copy_idx + 1

    # Create filtered JSON
    filtered_json = copy.deepcopy(json_data)
    filtered_json["sequences"] = filtered_sequences

    # Filter covalent bonds if present
    if "covalent_bonds" in filtered_json and kept_entity_indices:
        # Map old entity indices (1-based in bonds) to new indices
        old_to_new_entity = {}
        for new_idx, old_idx in enumerate(kept_entity_indices):
            old_to_new_entity[old_idx + 1] = new_idx + 1  # Bonds use 1-based entity IDs

        filtered_bonds = []
        for bond in filtered_json.get("covalent_bonds", []):
            # Handle entity1/entity2 type inconsistency (may be int or string)
            entity1 = bond.get("entity1")
            entity2 = bond.get("entity2")
            if isinstance(entity1, str):
                entity1 = int(entity1)
            if isinstance(entity2, str):
                entity2 = int(entity2)

            # Keep bond only if both entities are in the filtered set
            if entity1 in old_to_new_entity and entity2 in old_to_new_entity:
                new_bond = copy.deepcopy(bond)
                new_bond["entity1"] = old_to_new_entity[entity1]
                new_bond["entity2"] = old_to_new_entity[entity2]

                # Remap copy1 and copy2 if present
                copy1 = bond.get("copy1")
                copy2 = bond.get("copy2")

                if copy1 is not None:
                    if isinstance(copy1, str):
                        copy1 = int(copy1)
                    remap_key = (entity1, copy1)
                    if remap_key in entity_copy_remap:
                        new_bond["copy1"] = entity_copy_remap[remap_key]
                    else:
                        # If no explicit remap found, the bond references a filtered-out copy
                        # Skip this bond
                        continue

                if copy2 is not None:
                    if isinstance(copy2, str):
                        copy2 = int(copy2)
                    remap_key = (entity2, copy2)
                    if remap_key in entity_copy_remap:
                        new_bond["copy2"] = entity_copy_remap[remap_key]
                    else:
                        # If no explicit remap found, the bond references a filtered-out copy
                        # Skip this bond
                        continue

                # When bond has no copy1/copy2, it applies to all corresponding copies
                # (copy1-copy1, copy2-copy2, ...). Per Protenix spec, both entities must
                # have equal counts. After filtering, we need to verify this still holds.
                if copy1 is None and copy2 is None:
                    new_entity1_idx = old_to_new_entity[entity1] - 1  # Convert to 0-based
                    new_entity2_idx = old_to_new_entity[entity2] - 1

                    count1 = _get_entity_count(filtered_sequences[new_entity1_idx])
                    count2 = _get_entity_count(filtered_sequences[new_entity2_idx])

                    if count1 != count2:
                        logger.warning(
                            f"Skipping covalent bond without copy specifiers: "
                            f"entity1={new_bond['entity1']} (count={count1}) and "
                            f"entity2={new_bond['entity2']} (count={count2}) have unequal counts "
                            f"after filtering. Original bond: {bond}"
                        )
                        continue

                filtered_bonds.append(new_bond)

        if filtered_bonds:
            filtered_json["covalent_bonds"] = filtered_bonds
        else:
            # Remove empty covalent_bonds
            filtered_json.pop("covalent_bonds", None)

    logger.debug(
        f"Filtered JSON: {len(json_data.get('sequences', []))} -> "
        f"{len(filtered_sequences)} sequences for chains {chain_ids}"
    )

    return filtered_json


def reorder_sequences_by_chain_ids(
    json_data: dict,
    ordered_chain_ids: list[str],
    case_insensitive: bool = True,
) -> dict:
    """
    Reorder sequences in JSON to match the specified chain ID order.

    This function reorders the sequences list so that entities appear in the
    same order as the provided chain IDs. Entities are matched by auth_asym_id.

    Args:
        json_data: Protenix JSON dictionary with sequences
        ordered_chain_ids: List of chain IDs in the desired order
            (e.g., ["D", "E", "G"] for heavy, light, antigen)
        case_insensitive: If True, perform case-insensitive chain ID matching (default: True)

    Returns:
        JSON dictionary with reordered sequences

    Notes:
        - Only reorders entities that match auth_asym_id in ordered_chain_ids
        - Entities not in ordered_chain_ids are appended at the end
        - Does NOT remap covalent bonds (assumes this is called after filtering)
    """
    import copy

    sequences = json_data.get("sequences", [])
    if not sequences:
        return json_data

    # Build a mapping from auth_asym_id (normalized) to sequence item
    # For entities with multiple chains, use the first auth_asym_id
    chain_to_seq: dict[str, dict] = {}
    unmatched_sequences: list[dict] = []

    for seq_item in sequences:
        entity_data = None
        for etype in ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]:
            if etype in seq_item:
                entity_data = seq_item[etype]
                break

        if entity_data is None:
            unmatched_sequences.append(seq_item)
            continue

        auth_asym_ids = entity_data.get("auth_asym_id", [])
        if not auth_asym_ids:
            unmatched_sequences.append(seq_item)
            continue

        # Use the first auth_asym_id as the key (normalized for case-insensitive matching)
        first_chain_id = auth_asym_ids[0]
        key = first_chain_id.upper() if case_insensitive else first_chain_id
        chain_to_seq[key] = seq_item

    # Build reordered sequence list
    reordered_sequences: list[dict] = []
    seen_chains: set[str] = set()

    for chain_id in ordered_chain_ids:
        key = chain_id.upper() if case_insensitive else chain_id
        if key in chain_to_seq and key not in seen_chains:
            reordered_sequences.append(chain_to_seq[key])
            seen_chains.add(key)

    # Append any remaining sequences not in ordered_chain_ids
    for key, seq_item in chain_to_seq.items():
        if key not in seen_chains:
            reordered_sequences.append(seq_item)

    reordered_sequences.extend(unmatched_sequences)

    # Create result JSON
    result_json = copy.deepcopy(json_data)
    result_json["sequences"] = reordered_sequences

    return result_json


def get_entry_chain_ids(
    heavy_chain: str,
    light_chain: str | None,
    antigen_chains: list[str],
) -> list[str]:
    """
    Get all chain IDs from an entry's chain information.

    Args:
        heavy_chain: Heavy chain ID (e.g., "D")
        light_chain: Light chain ID (e.g., "E") or None for nanobody
        antigen_chains: List of antigen chain IDs (e.g., ["G"])

    Returns:
        List of all chain IDs (e.g., ["D", "E", "G"])
    """
    chain_ids = [heavy_chain]
    if light_chain:
        chain_ids.append(light_chain)
    chain_ids.extend(antigen_chains)
    return chain_ids


def _extract_cdr_info_from_chain(ab_chain) -> tuple[list[int], dict[str, CDRRegionInfo]]:
    """
    Extract CDR indices and region info from an abnumber Chain object.

    Args:
        ab_chain: abnumber Chain object

    Returns:
        Tuple of (cdr_indices, region_info)
    """
    # Get all positions in sequence order
    all_positions = list(ab_chain)

    # Collect CDR positions using *_dict attributes
    # Note: abnumber uses cdr1_dict, cdr2_dict, cdr3_dict (OrderedDicts)
    cdr_positions = set()
    region_info: dict[str, CDRRegionInfo] = {}

    for region_name in CHOTHIA_CDR_REGIONS:
        # Get CDR dict (e.g., cdr1_dict, cdr2_dict, cdr3_dict)
        region_dict = getattr(ab_chain, f"{region_name}_dict", {})
        region_positions = set(region_dict.keys())
        region_seq = getattr(ab_chain, f"{region_name}_seq", "")

        # Add to all CDR positions
        cdr_positions.update(region_positions)

        # Map positions to sequence indices
        region_indices = []
        for idx, (pos, aa) in enumerate(all_positions):
            if pos in region_positions:
                region_indices.append(idx)

        region_info[region_name] = CDRRegionInfo(
            region_name=region_name,
            indices=region_indices,
            sequence=region_seq,
        )

    # Get all CDR indices (sorted by position in sequence)
    cdr_indices = []
    for idx, (pos, aa) in enumerate(all_positions):
        if pos in cdr_positions:
            cdr_indices.append(idx)

    return cdr_indices, region_info


def get_cdr_indices_from_sequence(
    sequence: str,
    scheme: str = "chothia",
    expected_chain_type: str | None = None,
) -> tuple[list[int], dict[str, CDRRegionInfo], int] | tuple[None, str, None]:
    """
    Get CDR residue indices from an antibody sequence.

    Uses abnumber library to identify CDR regions based on the specified
    numbering scheme. Handles both single-domain antibody chains and
    multi-domain sequences like ScFv (single-chain variable fragment).

    Args:
        sequence: Antibody chain sequence (heavy or light, or ScFv)
        scheme: Numbering scheme (default: "chothia")
        expected_chain_type: Expected chain type ("H" for heavy, "L" for light).
            Used to select the correct domain when parsing ScFv sequences.
            If None, returns the first successfully parsed domain.

    Returns:
        Tuple of (cdr_indices, region_info, domain_start) if successful:
            - cdr_indices: CDR indices in the full sequence (0-based)
            - region_info: per-CDR-region details
            - domain_start: offset of the variable domain in the full sequence
        Tuple of (None, error_message, None) if CDR detection fails

    Notes:
        - Returns zero-based indices relative to the full sequence
        - CDR indices are sorted in ascending order
        - region_info contains per-CDR details (cdr1, cdr2, cdr3)
        - For ScFv sequences, extracts CDR from both heavy and light domains
        - domain_start is used to map full-sequence indices back to variable-domain indices
    """
    try:
        from abnumber import Chain
    except ImportError:
        return None, "abnumber library not installed", None

    # First, try standard single-domain parsing
    single_chain_error = None
    try:
        ab_chain = Chain(sequence, scheme=scheme)
        cdr_indices, region_info = _extract_cdr_info_from_chain(ab_chain)

        # Fix: compute the offset of the variable domain inside the original sequence.
        # abnumber.Chain automatically extracts the VH/VL variable domain, stripping
        # signal peptides and other leading prefixes. The indices returned by
        # _extract_cdr_info_from_chain are relative to ab_chain.seq, so we add the
        # offset to get indices relative to the original full sequence.
        domain_start = sequence.find(ab_chain.seq)
        if domain_start == -1:
            return None, "Could not locate variable domain in original sequence", None

        # If the variable domain is not at the start (e.g. a signal-peptide prefix exists), shift all indices.
        if domain_start > 0:
            cdr_indices = [idx + domain_start for idx in cdr_indices]
            for region_name in CHOTHIA_CDR_REGIONS:
                region_info[region_name]["indices"] = [
                    idx + domain_start for idx in region_info[region_name]["indices"]
                ]

        # Return full-sequence indices and the variable-domain start offset.
        return cdr_indices, region_info, domain_start
    except Exception as e:
        # Save the error for potential reporting later
        single_chain_error = str(e)

    # Fallback: try multi-domain parsing (handles ScFv and other multi-domain sequences)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            chains_list = Chain.multiple_domains(sequence, scheme=scheme)
        except Exception as e:
            # Both single-chain and multi-domain parsing failed
            return None, f"Failed to parse antibody sequence. Single-chain error: {single_chain_error}; Multi-domain error: {e}", None

    if not chains_list:
        # Multi-domain returned empty, report original single-chain error
        return None, f"Failed to parse antibody sequence: {single_chain_error}", None

    # For ScFv sequences, we need to combine CDRs from all domains
    # The indices need to be mapped back to the original sequence
    all_cdr_indices = []
    combined_region_info: dict[str, CDRRegionInfo] = {
        "cdr1": CDRRegionInfo(region_name="cdr1", indices=[], sequence=""),
        "cdr2": CDRRegionInfo(region_name="cdr2", indices=[], sequence=""),
        "cdr3": CDRRegionInfo(region_name="cdr3", indices=[], sequence=""),
    }

    # Track the start offset of the matched variable domain (used to convert back to variable-domain indices).
    # For ScFv we only return the start offset of the domain matching expected_chain_type.
    matched_domain_start: int | None = None

    # chains_list contains Chain objects for each domain
    # We need to find the offset of each domain in the original sequence
    for domain_idx, ab_chain in enumerate(chains_list):
        # Get the domain sequence
        domain_seq = ab_chain.seq

        # Find where this domain starts in the original sequence
        domain_start = sequence.find(domain_seq)
        if domain_start == -1:
            logger.warning(f"Could not locate domain {domain_idx} in original sequence")
            continue

        # If expected_chain_type is specified, only process matching domain
        if expected_chain_type:
            chain_type = ab_chain.chain_type  # "H" or "L" or "K"
            # Normalize: K (kappa) and L (lambda) are both light chains
            if chain_type in ("K", "L") and expected_chain_type == "L":
                pass  # Match
            elif chain_type == "H" and expected_chain_type == "H":
                pass  # Match
            else:
                continue  # Skip non-matching domain

        # Record the start offset of the first matched domain (an ScFv typically has one H and one L domain).
        if matched_domain_start is None:
            matched_domain_start = domain_start

        # Extract CDR info from this domain
        domain_cdr_indices, domain_region_info = _extract_cdr_info_from_chain(ab_chain)

        # Offset indices to original sequence positions
        offset_indices = [idx + domain_start for idx in domain_cdr_indices]
        all_cdr_indices.extend(offset_indices)

        # Combine region info
        for region_name in CHOTHIA_CDR_REGIONS:
            domain_region = domain_region_info[region_name]
            offset_region_indices = [idx + domain_start for idx in domain_region["indices"]]

            # Append to combined info
            combined_region_info[region_name]["indices"].extend(offset_region_indices)
            if domain_region["sequence"]:
                existing_seq = combined_region_info[region_name]["sequence"]
                if existing_seq:
                    combined_region_info[region_name]["sequence"] = existing_seq + "+" + domain_region["sequence"]
                else:
                    combined_region_info[region_name]["sequence"] = domain_region["sequence"]

    if not all_cdr_indices:
        return None, f"No CDR regions found in ScFv sequence (expected_chain_type={expected_chain_type})", None

    # Sort indices
    all_cdr_indices.sort()
    for region_name in CHOTHIA_CDR_REGIONS:
        combined_region_info[region_name]["indices"].sort()

    # Return full-sequence indices and the start offset of the matched variable domain.
    return all_cdr_indices, combined_region_info, matched_domain_start if matched_domain_start is not None else 0


def mask_sequence(
    sequence: str,
    indices: list[int],
    mask_token: str = "X",
) -> str:
    """
    Replace residues at specified indices with mask token.

    Args:
        sequence: Original sequence string
        indices: Zero-based indices to mask
        mask_token: Character to use for masking (default: "X")

    Returns:
        Masked sequence string

    Raises:
        ValueError: If any index is out of bounds
    """
    if not indices:
        return sequence

    seq_list = list(sequence)
    for idx in indices:
        if idx < 0 or idx >= len(seq_list):
            raise ValueError(f"Index {idx} out of bounds for sequence length {len(seq_list)}")
        seq_list[idx] = mask_token

    return "".join(seq_list)


def find_matching_entity_by_sequence(
    json_data: dict,
    target_sequence: str,
    min_match_ratio: float = 0.95,
) -> tuple[int | None, str | None]:
    """
    Find the entity in JSON that contains a matching sequence.

    Since CIF and PDB may have different chain IDs, we match by sequence.

    Args:
        json_data: Protenix JSON dictionary
        target_sequence: Sequence to match (can be a subsequence)
        min_match_ratio: Minimum ratio of target sequence that must match

    Returns:
        Tuple of (entity_index, full_sequence) if found, (None, None) otherwise
    """
    target_len = len(target_sequence)

    for idx, seq_item in enumerate(json_data.get("sequences", [])):
        if "proteinChain" not in seq_item:
            continue

        entity_seq = seq_item["proteinChain"].get("sequence", "")

        # Check if target sequence is contained in entity sequence
        if target_sequence in entity_seq:
            return idx, entity_seq

        # Check if entity sequence is contained in target (for truncated chains)
        if entity_seq in target_sequence:
            return idx, entity_seq

        # Partial match: check if significant portion overlaps
        # This handles cases where sequences differ slightly
        if len(entity_seq) >= target_len * 0.8:
            # Simple overlap check
            match_count = sum(1 for a, b in zip(target_sequence, entity_seq) if a == b)
            if match_count >= target_len * min_match_ratio:
                return idx, entity_seq

    return None, None


def _chain_id_matches(target: str, candidate: str, case_insensitive: bool = True) -> bool:
    """Check if two chain IDs match, optionally case-insensitive."""
    if case_insensitive:
        return target.upper() == candidate.upper()
    return target == candidate


def find_entity_by_chain_id(
    json_data: dict,
    chain_id: str,
    case_insensitive: bool = True,
) -> tuple[int | None, str | None]:
    """
    Find an entity in JSON by chain ID.

    Searches both auth_asym_id (PDB chain ID) and label_asym_id (CIF chain ID).
    Priority: auth_asym_id first, then label_asym_id.

    Args:
        json_data: Protenix JSON dictionary (must have save_entity_and_asym_id=True)
        chain_id: Chain ID to find (e.g., "H", "L", "A")
        case_insensitive: If True, perform case-insensitive matching (default: True)

    Returns:
        Tuple of (entity_index, sequence) if found, (None, None) otherwise
    """
    for idx, seq_item in enumerate(json_data.get("sequences", [])):
        if "proteinChain" not in seq_item:
            continue

        entity_data = seq_item["proteinChain"]

        # First try auth_asym_id (PDB original chain ID)
        auth_asym_ids = entity_data.get("auth_asym_id", [])
        for auth_id in auth_asym_ids:
            if _chain_id_matches(chain_id, auth_id, case_insensitive):
                return idx, entity_data.get("sequence", "")

        # Then try label_asym_id (CIF standard chain ID)
        label_asym_ids = entity_data.get("label_asym_id", [])
        for label_id in label_asym_ids:
            if _chain_id_matches(chain_id, label_id, case_insensitive):
                return idx, entity_data.get("sequence", "")

    return None, None


def remove_invalid_covalent_bonds(
    json_data: dict,
    masked_positions: list[tuple[int, int]],
) -> dict:
    """
    Remove covalent bonds that involve masked positions.

    When CDR residues are masked (replaced with 'X'), any disulfide bonds
    (or other covalent bonds) involving those positions become invalid since
    the masked residue 'X' has no sulfur atom (SG). This function removes
    such bonds to prevent Protenix inference errors.

    Args:
        json_data: Protenix JSON dictionary (will be modified in-place)
        masked_positions: List of (entity_id, position) tuples for masked residues.
            entity_id is 1-based (as used in JSON covalent_bonds).
            position is 1-based residue position in the sequence.

    Returns:
        The modified json_data with invalid bonds removed
    """
    if not masked_positions or "covalent_bonds" not in json_data:
        return json_data

    # Convert to a set for O(1) lookup
    masked_set = set(masked_positions)

    original_bonds = json_data.get("covalent_bonds", [])
    valid_bonds = []

    for bond in original_bonds:
        # Get entity and position for both ends of the bond
        # Note: entity1/entity2 may be int or string in JSON
        entity1 = bond.get("entity1")
        entity2 = bond.get("entity2")
        position1 = bond.get("position1")
        position2 = bond.get("position2")

        if isinstance(entity1, str):
            entity1 = int(entity1)
        if isinstance(entity2, str):
            entity2 = int(entity2)
        if isinstance(position1, str):
            position1 = int(position1)
        if isinstance(position2, str):
            position2 = int(position2)

        # Check if either end of the bond is at a masked position
        if (entity1, position1) in masked_set or (entity2, position2) in masked_set:
            logger.debug(
                f"Removing covalent bond involving masked position: "
                f"entity1={entity1}, pos1={position1}, entity2={entity2}, pos2={position2}"
            )
            continue

        valid_bonds.append(bond)

    # Update or remove covalent_bonds
    if valid_bonds:
        json_data["covalent_bonds"] = valid_bonds
    else:
        json_data.pop("covalent_bonds", None)

    removed_count = len(original_bonds) - len(valid_bonds)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} covalent bond(s) involving masked positions")

    return json_data


def extract_and_mask_cdr(
    json_data: dict,
    heavy_chain_id: str,
    light_chain_id: str | None,
    mask_token: str = "X",
) -> CDRMaskingResult:
    """
    Extract antibody chains from JSON by chain ID and mask CDR regions.

    This function:
    1. Finds antibody chains by chain ID (auth_asym_id or label_asym_id)
    2. Identifies CDR regions using abnumber
    3. Masks CDR residues in the JSON data

    Note:
        This function creates a deep copy of json_data internally to avoid
        modifying the original input.

    Args:
        json_data: Protenix JSON dictionary (must have save_entity_and_asym_id=True)
        heavy_chain_id: Heavy chain ID (e.g., "H", "D")
        light_chain_id: Light chain ID (e.g., "L", "E") or None for nanobody
        mask_token: Character to use for masking CDRs

    Returns:
        CDRMaskingResult with modified JSON and CDR information
    """
    import copy

    # Create a deep copy to avoid modifying the original input
    json_data = copy.deepcopy(json_data)

    entry_name = json_data.get("name", "unknown")
    result = CDRMaskingResult(
        entry_name=entry_name,
        json_data=[json_data],  # Wrap in list for Protenix format
    )

    # Track masked positions for covalent bond cleanup
    # Format: list of (entity_id_1based, position_1based)
    masked_positions: list[tuple[int, int]] = []

    # Process heavy chain
    entity_idx, entity_seq = find_entity_by_chain_id(json_data, heavy_chain_id)

    if entity_idx is not None:
        result.matched_heavy_entity = entity_idx
        # Pass expected_chain_type="H" to handle ScFv sequences correctly
        cdr_result = get_cdr_indices_from_sequence(entity_seq, expected_chain_type="H")

        if cdr_result[0] is not None:
            # Unpack: cdr_indices are full-sequence indices; domain_start is used for later conversion.
            cdr_indices, region_info, domain_start = cdr_result
            masked_seq = mask_sequence(entity_seq, cdr_indices, mask_token)

            # Update JSON
            json_data["sequences"][entity_idx]["proteinChain"]["sequence"] = masked_seq

            # Record masked positions for covalent bond cleanup
            # entity_idx is 0-based, convert to 1-based entity_id
            # cdr_indices are 0-based, convert to 1-based positions
            entity_id_1based = entity_idx + 1
            for idx in cdr_indices:
                masked_positions.append((entity_id_1based, idx + 1))

            result.heavy_chain_info = ChainCDRInfo(
                chain_type="H",
                original_seq=entity_seq,
                masked_seq=masked_seq,
                cdr_indices=cdr_indices,
                domain_start=domain_start,  # save variable-domain start offset
                cdr1=region_info["cdr1"],
                cdr2=region_info["cdr2"],
                cdr3=region_info["cdr3"],
                error=None,
            )
        else:
            error_msg = cdr_result[1]
            result.success = False
            result.error_message = f"Heavy chain CDR detection failed: {error_msg}"
            result.heavy_chain_info = ChainCDRInfo(
                chain_type="H",
                original_seq=entity_seq,
                masked_seq=entity_seq,  # Keep original if CDR detection fails
                cdr_indices=[],
                domain_start=0,  # default value
                cdr1=CDRRegionInfo(region_name="cdr1", indices=[], sequence=""),
                cdr2=CDRRegionInfo(region_name="cdr2", indices=[], sequence=""),
                cdr3=CDRRegionInfo(region_name="cdr3", indices=[], sequence=""),
                error=error_msg,
            )
            logger.warning(f"{entry_name}: {result.error_message}")
    else:
        result.success = False
        result.error_message = f"Could not find heavy chain '{heavy_chain_id}' in JSON"
        logger.error(f"{entry_name}: {result.error_message}")

    # Process light chain (if not a nanobody)
    if light_chain_id:
        entity_idx, entity_seq = find_entity_by_chain_id(json_data, light_chain_id)

        if entity_idx is not None:
            result.matched_light_entity = entity_idx
            # Pass expected_chain_type="L" to handle ScFv sequences correctly
            cdr_result = get_cdr_indices_from_sequence(entity_seq, expected_chain_type="L")

            if cdr_result[0] is not None:
                # Unpack: cdr_indices are full-sequence indices; domain_start is used for later conversion.
                cdr_indices, region_info, domain_start = cdr_result
                masked_seq = mask_sequence(entity_seq, cdr_indices, mask_token)

                # Update JSON
                json_data["sequences"][entity_idx]["proteinChain"]["sequence"] = masked_seq

                # Record masked positions for covalent bond cleanup
                # entity_idx is 0-based, convert to 1-based entity_id
                # cdr_indices are 0-based, convert to 1-based positions
                entity_id_1based = entity_idx + 1
                for idx in cdr_indices:
                    masked_positions.append((entity_id_1based, idx + 1))

                result.light_chain_info = ChainCDRInfo(
                    chain_type="L",
                    original_seq=entity_seq,
                    masked_seq=masked_seq,
                    cdr_indices=cdr_indices,
                    domain_start=domain_start,  # save variable-domain start offset
                    cdr1=region_info["cdr1"],
                    cdr2=region_info["cdr2"],
                    cdr3=region_info["cdr3"],
                    error=None,
                )
            else:
                error_msg = cdr_result[1]
                result.success = False
                result.error_message = f"Light chain CDR detection failed: {error_msg}"
                result.light_chain_info = ChainCDRInfo(
                    chain_type="L",
                    original_seq=entity_seq,
                    masked_seq=entity_seq,
                    cdr_indices=[],
                    domain_start=0,  # default value
                    cdr1=CDRRegionInfo(region_name="cdr1", indices=[], sequence=""),
                    cdr2=CDRRegionInfo(region_name="cdr2", indices=[], sequence=""),
                    cdr3=CDRRegionInfo(region_name="cdr3", indices=[], sequence=""),
                    error=error_msg,
                )
                logger.warning(f"{entry_name}: {result.error_message}")
        else:
            result.success = False
            result.error_message = f"Could not find light chain '{light_chain_id}' in JSON"
            logger.error(f"{entry_name}: {result.error_message}")

    # Remove covalent bonds involving masked positions (e.g., disulfide bonds
    # where one cysteine was in a CDR region and is now masked as 'X')
    if masked_positions:
        json_data = remove_invalid_covalent_bonds(json_data, masked_positions)
        # Update result.json_data to reflect the modification (for clarity)
        result.json_data = [json_data]

    return result


def process_entry_cdr_masking(
    entry: EntryInfo,
    output_json_path: str | Path | None = None,
    output_cif_path: str | Path | None = None,
    output_cdr_info_path: str | Path | None = None,
    mask_token: str = "X",
) -> CDRMaskingResult:
    """
    Full pipeline to process a single entry for CDR masking.

    This function:
    1. Converts CIF to Protenix JSON format
    2. Filters JSON to keep only the protein chains from this entry
       (removes ligands, ions, and other non-protein entities)
    3. Reorders sequences to: heavy chain -> light chain -> antigen chains
    4. Identifies and masks CDR regions in antibody chains
    5. Optionally extracts chains to a ground truth CIF file
    6. Optionally saves CDR info and chain mapping for lDDT calculation

    Args:
        entry: Entry information from load_entries_for_split()
        output_json_path: Optional path to save the masked JSON
        output_cif_path: Optional path to save the extracted chains as CIF
                         (ground truth for offline LDDT evaluation)
        output_cdr_info_path: Optional path to save CDR info and chain mapping JSON
                              (needed for offline lDDT and CIF post-processing)
        mask_token: Character to use for masking CDRs (default: "X")

    Returns:
        CDRMaskingResult with modified JSON and CDR information
    """
    cif_path = entry["cif_path"]
    entry_name = entry["entry"]
    heavy_chain_id = entry["heavy_chain"]
    light_chain_id = entry["light_chain"]

    # Step 1: Convert CIF to JSON
    try:
        json_data = cif_to_protenix_json(cif_path, sample_name=entry_name, get_entity_seq_with_coords=False)
    except Exception as e:
        return CDRMaskingResult(
            entry_name=entry_name,
            json_data=[],
            success=False,
            error_message=f"CIF to JSON conversion failed: {e}",
        )

    # Step 2: Filter JSON to keep only the protein chains from this entry
    chain_ids = get_entry_chain_ids(
        heavy_chain=heavy_chain_id,
        light_chain=light_chain_id,
        antigen_chains=entry["antigen_chains"],
    )
    json_data = filter_json_by_chains(json_data, chain_ids, protein_only=True)

    # DO NOT reorder sequences to match: heavy -> light -> antigen order

    # Step 3: Extract and mask CDRs
    result = extract_and_mask_cdr(
        json_data=json_data,
        heavy_chain_id=heavy_chain_id,
        light_chain_id=light_chain_id,
        mask_token=mask_token,
    )

    # Step 4: Save JSON if requested
    if output_json_path and result.success:
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_json_path, "w") as f:
            json.dump(result.json_data, f, indent=2)

        logger.info(f"Saved masked JSON to: {output_json_path}")

    # Step 5: Extract and save CIF if requested (ground truth for offline LDDT)
    if output_cif_path and result.success:
        from proteor1.cdr_eval.chain_extractor import extract_chains_from_cif

        extraction_result = extract_chains_from_cif(
            cif_path=cif_path,
            heavy_chain=heavy_chain_id,
            light_chain=light_chain_id,
            antigen_chains=entry["antigen_chains"],
            output_path=output_cif_path,
            entry_name=entry_name,
            apply_filters=True,
        )

        if extraction_result.success:
            logger.info(f"Saved extracted chains CIF to: {output_cif_path}")
        else:
            logger.warning(
                f"CIF extraction failed for {entry_name}: {extraction_result.error_message}"
            )

    # Step 6: Save CDR info and chain mapping if requested
    if output_cdr_info_path and result.success:
        save_cdr_info_and_chain_mapping(
            result=result,
            output_path=output_cdr_info_path,
            heavy_chain_id=heavy_chain_id,
            light_chain_id=light_chain_id,
        )

    return result


def get_cdr_summary(result: CDRMaskingResult) -> dict:
    """
    Generate a summary dictionary from CDR masking result.

    Useful for logging and aggregating results.

    Args:
        result: CDRMaskingResult from processing

    Returns:
        Summary dictionary with key statistics
    """
    summary = {
        "entry": result.entry_name,
        "success": result.success,
        "error": result.error_message,
    }

    if result.heavy_chain_info:
        h_info = result.heavy_chain_info
        summary["heavy_chain"] = {
            "num_cdr_residues": len(h_info["cdr_indices"]),
            "cdr1_len": len(h_info["cdr1"]["indices"]),
            "cdr2_len": len(h_info["cdr2"]["indices"]),
            "cdr3_len": len(h_info["cdr3"]["indices"]),
            "cdr_error": h_info["error"],
        }

    if result.light_chain_info:
        l_info = result.light_chain_info
        summary["light_chain"] = {
            "num_cdr_residues": len(l_info["cdr_indices"]),
            "cdr1_len": len(l_info["cdr1"]["indices"]),
            "cdr2_len": len(l_info["cdr2"]["indices"]),
            "cdr3_len": len(l_info["cdr3"]["indices"]),
            "cdr_error": l_info["error"],
        }

    return summary


def get_chain_mapping_from_json(json_data: list[dict]) -> dict:
    """
    Extract chain ID mapping from Protenix JSON data.

    Protenix assigns chain IDs (A, B, C, ...) based on the order of sequences
    in the input JSON. This function extracts the original auth_asym_id for each
    entity to build a mapping.

    Args:
        json_data: Protenix JSON data (list with one dict)

    Returns:
        Dictionary with chain mapping information:
        {
            "protenix_to_original": {"A": "E", "B": "C", "C": "D"},
            "original_to_protenix": {"E": "A", "C": "B", "D": "C"},
            "sequence_order": [
                {"protenix_chain": "A", "original_chain": "E", "seq_len": 260},
                ...
            ]
        }
    """
    if not json_data or not isinstance(json_data, list) or len(json_data) == 0:
        return {}

    data = json_data[0]
    sequences = data.get("sequences", [])

    protenix_to_original = {}
    original_to_protenix = {}
    sequence_order = []

    # Protenix assigns chain IDs starting from A, incrementing for each chain
    # based on the order in sequences array
    chain_idx = 0
    for seq_item in sequences:
        for entity_type in ["proteinChain", "dnaSequence", "rnaSequence", "ligand", "ion"]:
            if entity_type not in seq_item:
                continue

            entity_data = seq_item[entity_type]
            count = entity_data.get("count", 1)
            auth_asym_ids = entity_data.get("auth_asym_id", [])
            sequence = entity_data.get("sequence", "")

            for copy_idx in range(count):
                # Protenix uses int_to_letters: 1->A, 2->B, ...
                protenix_chain = _int_to_letters(chain_idx + 1)

                # Get original chain ID if available
                if copy_idx < len(auth_asym_ids):
                    original_chain = auth_asym_ids[copy_idx]
                else:
                    original_chain = protenix_chain  # fallback

                protenix_to_original[protenix_chain] = original_chain
                original_to_protenix[original_chain] = protenix_chain

                sequence_order.append({
                    "protenix_chain": protenix_chain,
                    "original_chain": original_chain,
                    "seq_len": len(sequence) if sequence else 0,
                    "entity_type": entity_type,
                })

                chain_idx += 1
            break  # Only process one entity type per seq_item

    return {
        "protenix_to_original": protenix_to_original,
        "original_to_protenix": original_to_protenix,
        "sequence_order": sequence_order,
    }


def _int_to_letters(n: int) -> str:
    """
    Convert int to letters (same as Protenix's int_to_letters).

    Args:
        n: int number (1-based)

    Returns:
        Letters: 1->A, 2->B, ..., 26->Z, 27->AA, 28->AB, ...
    """
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def get_cdr_info_for_lddt(result: CDRMaskingResult) -> dict:
    """
    Generate a CDR info dictionary for downstream lDDT-style scoring.

    The CDR indices are converted from full-sequence indices back to
    variable-domain (VH/VL) indices.

    The variable_domain_start_res_id field preserves exact residue-ID mapping
    when the ground-truth CIF has extra residues before the variable domain.

    Args:
        result: CDRMaskingResult from CDR masking

    Returns:
        Dictionary with CDR info for lDDT calculation:
        {
            "H_chain": {
                "cdr_indices": [...],  # variable-domain indices (0-based)
                "variable_domain_start_res_id": int,  # variable-domain start res_id (1-based)
                "cdr1": {"indices": [...]},
                "cdr2": {"indices": [...]},
                "cdr3": {"indices": [...]}
            },
            "L_chain": {...}  # or None for nanobody
        }
    """
    cdr_info = {}

    if result.heavy_chain_info and not result.heavy_chain_info.get("error"):
        h_info = result.heavy_chain_info
        # Read variable-domain start offset, used to convert full-sequence indices to variable-domain indices.
        domain_start = h_info.get("domain_start", 0)

        # Convert to variable-domain indices and also record variable_domain_start_res_id for precise mapping.
        # res_id is 1-based, so use domain_start + 1.
        cdr_info["H_chain"] = {
            "cdr_indices": [i - domain_start for i in h_info["cdr_indices"]],
            "variable_domain_start_res_id": domain_start + 1,  # variable-domain start res_id (1-based)
            "cdr1": {"indices": [i - domain_start for i in h_info["cdr1"]["indices"]]},
            "cdr2": {"indices": [i - domain_start for i in h_info["cdr2"]["indices"]]},
            "cdr3": {"indices": [i - domain_start for i in h_info["cdr3"]["indices"]]},
        }

    if result.light_chain_info and not result.light_chain_info.get("error"):
        l_info = result.light_chain_info
        # Read variable-domain start offset, used to convert full-sequence indices to variable-domain indices.
        domain_start = l_info.get("domain_start", 0)

        # Convert to variable-domain indices and also record variable_domain_start_res_id for precise mapping.
        cdr_info["L_chain"] = {
            "cdr_indices": [i - domain_start for i in l_info["cdr_indices"]],
            "variable_domain_start_res_id": domain_start + 1,  # variable-domain start res_id (1-based)
            "cdr1": {"indices": [i - domain_start for i in l_info["cdr1"]["indices"]]},
            "cdr2": {"indices": [i - domain_start for i in l_info["cdr2"]["indices"]]},
            "cdr3": {"indices": [i - domain_start for i in l_info["cdr3"]["indices"]]},
        }

    return cdr_info


def save_cdr_info_and_chain_mapping(
    result: CDRMaskingResult,
    output_path: str | Path,
    heavy_chain_id: str,
    light_chain_id: str | None,
) -> None:
    """
    Save CDR info and chain mapping to a JSON file.

    This file is needed for:
    1. Downstream scoring utilities
    2. Post-processing Protenix output CIF to restore original chain IDs

    Args:
        result: CDRMaskingResult from CDR masking
        output_path: Path to save the info JSON file
        heavy_chain_id: Original heavy chain ID
        light_chain_id: Original light chain ID (None for nanobody)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get chain mapping
    chain_mapping = get_chain_mapping_from_json(result.json_data)

    # Get CDR info for lDDT
    cdr_info = get_cdr_info_for_lddt(result)

    # Build complete info dict
    info = {
        "entry_name": result.entry_name,
        "heavy_chain_id": heavy_chain_id,
        "light_chain_id": light_chain_id,
        "chain_mapping": chain_mapping,
        "cdr_info": cdr_info,
    }

    with open(output_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved CDR info and chain mapping to: {output_path}")


if __name__ == "__main__":
    # Simple test when run directly
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="CDR Masking Tool")
    parser.add_argument("--cif_dir", type=str, default="data/cif", help="Directory containing CIF files")
    parser.add_argument("--split_json", type=str, required=True, help="Path to split JSON file")
    parser.add_argument("--output_dir", type=str, default="data/cdr_masking", help="Output directory for masked JSON files")
    args = parser.parse_args()

    entries = load_entries_for_split(args.split_json, args.cif_dir)

    output_json_dir = f"{args.output_dir}/protenix_json"
    output_cif_dir = f"{args.output_dir}/cif"
    output_cdr_info_dir = f"{args.output_dir}/cdr_info"

    for entry in tqdm(entries):
        # if entry['entry'] != "8e1m_H_L_C":
        #     continue

        output_json_path=f"{output_json_dir}/{entry['entry']}.json"
        output_cif_path = f"{output_cif_dir}/{entry['entry']}.cif"
        output_cdr_info_path = f"{output_cdr_info_dir}/{entry['entry']}_cdr_info.json"

        result = process_entry_cdr_masking(
            entry=entry,
            output_json_path=output_json_path,
            output_cif_path=output_cif_path,
            output_cdr_info_path=output_cdr_info_path,
        )
