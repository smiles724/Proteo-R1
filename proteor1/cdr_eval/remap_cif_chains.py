#!/usr/bin/env python3
"""
Post-processing script to remap chain IDs in Protenix output CIF files.

Protenix assigns chain IDs (A, B, C, ...) based on the order of sequences in
the input JSON. This script restores the original chain IDs using the chain
mapping saved during CDR masking.

Usage:
    python -m proteor1.cdr_eval.remap_cif_chains \
        --pred_cif path/to/prediction.cif \
        --cdr_info path/to/cdr_info.json \
        --output path/to/remapped.cif

    # Batch processing:
    python -m proteor1.cdr_eval.remap_cif_chains \
        --pred_cif_dir path/to/predictions/ \
        --cdr_info_dir path/to/cdr_info/ \
        --output_dir path/to/remapped/
"""

from __future__ import annotations

import copy
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import biotite.structure as struc
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class RemapResult:
    """Result of CIF chain ID remapping."""
    entry_name: str
    success: bool
    output_path: str | None = None
    error_message: str | None = None
    chains_remapped: dict[str, str] | None = None


def load_chain_mapping(cdr_info_path: str | Path) -> dict[str, str]:
    """
    Load chain mapping from CDR info JSON file.

    Args:
        cdr_info_path: Path to CDR info JSON file

    Returns:
        Dictionary mapping Protenix chain IDs to original chain IDs
        e.g., {"A": "E", "B": "C", "C": "D"}
    """
    with open(cdr_info_path, "r") as f:
        cdr_info = json.load(f)

    chain_mapping = cdr_info.get("chain_mapping", {})
    protenix_to_original = chain_mapping.get("protenix_to_original", {})

    return protenix_to_original


def _build_two_stage_mapping(
    chain_mapping: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """
    Build a two-stage mapping to handle chain ID conflicts.

    When we have cyclic mappings like A->B, B->A, directly applying
    the mapping would cause conflicts. We solve this by:
    1. First mapping all chains to unique temporary IDs
    2. Then mapping temporary IDs to final target IDs

    Args:
        chain_mapping: Original mapping from source to target chain IDs

    Returns:
        Tuple of (stage1_mapping, stage2_mapping)
        - stage1_mapping: source_id -> temp_id
        - stage2_mapping: temp_id -> target_id
    """
    stage1_mapping = {}  # source -> temp
    stage2_mapping = {}  # temp -> target

    for i, (source, target) in enumerate(chain_mapping.items()):
        # Create a unique temporary ID
        temp_id = f"__TEMP_{i}__"
        stage1_mapping[source] = temp_id
        stage2_mapping[temp_id] = target

    return stage1_mapping, stage2_mapping


def _apply_mapping_to_array(
    arr: np.ndarray,
    mapping: dict[str, str],
) -> tuple[np.ndarray, dict[str, str]]:
    """
    Apply chain ID mapping to a numpy array of chain IDs.

    Args:
        arr: Numpy array of chain ID strings
        mapping: Dictionary mapping old IDs to new IDs

    Returns:
        Tuple of (modified array, dict of chains that were actually remapped)
    """
    # Determine the max string length needed for new IDs
    max_len = max(
        max((len(v) for v in mapping.values()), default=1),
        max((len(s) for s in arr), default=1),
    )
    # Convert to object dtype to allow variable-length strings,
    # then convert back to fixed-width unicode string with sufficient length
    result = arr.astype(f"<U{max_len}")
    chains_remapped = {}

    for old_id, new_id in mapping.items():
        mask = arr == old_id
        if np.any(mask):
            result[mask] = new_id
            chains_remapped[old_id] = new_id

    return result, chains_remapped


def remap_cif_chain_ids(
    input_cif_path: str | Path,
    output_cif_path: str | Path,
    chain_mapping: dict[str, str],
) -> RemapResult:
    """
    Remap chain IDs in a CIF file using MMCIFParser and save_atoms_to_cif.

    This function:
    1. Parses the CIF using Protenix's MMCIFParser
    2. Modifies auth_asym_id, label_asym_id, and chain_id annotations
    3. Saves the result using save_atoms_to_cif

    Chain ID conflicts (e.g., A<->B swap) are handled using two-stage mapping
    with temporary intermediate chain IDs.

    Args:
        input_cif_path: Path to input CIF file
        output_cif_path: Path to output CIF file
        chain_mapping: Dictionary mapping Protenix chain IDs to original chain IDs

    Returns:
        RemapResult with success status and details
    """
    from protenix.data.parser import MMCIFParser
    from protenix.data.utils import save_atoms_to_cif

    input_cif_path = Path(input_cif_path)
    output_cif_path = Path(output_cif_path)

    if not input_cif_path.exists():
        return RemapResult(
            entry_name=input_cif_path.stem,
            success=False,
            error_message=f"Input CIF file not found: {input_cif_path}",
        )

    try:
        # Parse the CIF file
        parser = MMCIFParser(input_cif_path)
        atom_array = parser.get_structure(
            altloc="first",
            model=1,
            bond_lenth_threshold=None,
        )

        # Get available chains before modification for tracking
        available_chains = set()
        if hasattr(atom_array, "auth_asym_id"):
            available_chains = set(np.unique(atom_array.auth_asym_id))

        # Make a deep copy to avoid modifying the original
        atom_array = copy.deepcopy(atom_array)

        # Build two-stage mapping to handle conflicts
        stage1_mapping, stage2_mapping = _build_two_stage_mapping(chain_mapping)

        # Stage 1: Apply source -> temp mapping
        if hasattr(atom_array, "auth_asym_id"):
            atom_array.auth_asym_id, _ = _apply_mapping_to_array(
                atom_array.auth_asym_id, stage1_mapping
            )

        if hasattr(atom_array, "label_asym_id"):
            atom_array.set_annotation(
                "label_asym_id",
                _apply_mapping_to_array(atom_array.label_asym_id, stage1_mapping)[0]
            )

        # Also update chain_id (used by CIFWriter)
        atom_array.chain_id, _ = _apply_mapping_to_array(
            atom_array.chain_id, stage1_mapping
        )

        # Stage 2: Apply temp -> target mapping
        if hasattr(atom_array, "auth_asym_id"):
            atom_array.auth_asym_id, _ = _apply_mapping_to_array(
                atom_array.auth_asym_id, stage2_mapping
            )

        if hasattr(atom_array, "label_asym_id"):
            atom_array.set_annotation(
                "label_asym_id",
                _apply_mapping_to_array(atom_array.label_asym_id, stage2_mapping)[0]
            )

        atom_array.chain_id, _ = _apply_mapping_to_array(
            atom_array.chain_id, stage2_mapping
        )

        # Build the chains_remapped dict from the original mapping
        # Only include chains that actually existed in the input
        all_chains_remapped = {}
        for old_id, new_id in chain_mapping.items():
            if old_id in available_chains:
                all_chains_remapped[old_id] = new_id

        # Get entity_poly_type for the structure
        entity_poly_type = parser.entity_poly_type

        # Clear bonds to avoid struct_conn inconsistency after chain ID remapping.
        # When chain IDs are remapped, the bond references in struct_conn become
        # inconsistent with the atom_site chain IDs, causing biotite to fail with
        # "cannot be unambiguously assigned to atoms" error when reading the CIF.
        # Since we only need coordinates for lDDT calculation, removing bonds is safe.
        # Use empty BondList instead of None for compatibility with CIFWriter.
        atom_array.bonds = struc.BondList(len(atom_array))

        # Save to CIF
        output_cif_path.parent.mkdir(parents=True, exist_ok=True)
        save_atoms_to_cif(
            output_cif_file=str(output_cif_path),
            atom_array=atom_array,
            entity_poly_type=entity_poly_type,
            pdb_id=input_cif_path.stem,
        )

        return RemapResult(
            entry_name=input_cif_path.stem,
            success=True,
            output_path=str(output_cif_path),
            chains_remapped=all_chains_remapped,
        )

    except Exception as e:
        logger.error(f"Failed to remap {input_cif_path}: {e}")
        return RemapResult(
            entry_name=input_cif_path.stem,
            success=False,
            error_message=str(e),
        )


def remap_single_entry(
    pred_cif_path: str | Path,
    cdr_info_path: str | Path,
    output_path: str | Path,
) -> RemapResult:
    """
    Remap chain IDs for a single entry.

    Args:
        pred_cif_path: Path to Protenix prediction CIF file
        cdr_info_path: Path to CDR info JSON file
        output_path: Path to save remapped CIF file

    Returns:
        RemapResult with success status
    """
    pred_cif_path = Path(pred_cif_path)
    cdr_info_path = Path(cdr_info_path)

    if not cdr_info_path.exists():
        return RemapResult(
            entry_name=pred_cif_path.stem,
            success=False,
            error_message=f"CDR info file not found: {cdr_info_path}",
        )

    chain_mapping = load_chain_mapping(cdr_info_path)

    if not chain_mapping:
        return RemapResult(
            entry_name=pred_cif_path.stem,
            success=False,
            error_message="No chain mapping found in CDR info file",
        )

    return remap_cif_chain_ids(pred_cif_path, output_path, chain_mapping)


def _process_single_cif(
    args: tuple[Path, Path, Path],
) -> RemapResult:
    """
    Process a single CIF file for parallel execution.

    This is a top-level function to allow pickling for multiprocessing.

    Args:
        args: Tuple of (cif_path, cdr_info_dir, output_dir)

    Returns:
        RemapResult for this entry
    """
    cif_path, cdr_info_dir, output_dir = args

    # Extract entry name from CIF filename
    entry_name = cif_path.stem

    # Try to find matching CDR info file
    cdr_info_candidates = [
        cdr_info_dir / f"{entry_name}_cdr_info.json",
        cdr_info_dir / f"{entry_name}.json",
    ]

    # Also try without sample suffix
    if "_sample_" in entry_name:
        base_name = entry_name.rsplit("_sample_", 1)[0]
        cdr_info_candidates.extend([
            cdr_info_dir / f"{base_name}_cdr_info.json",
            cdr_info_dir / f"{base_name}.json",
        ])

    cdr_info_path = None
    for candidate in cdr_info_candidates:
        if candidate.exists():
            cdr_info_path = candidate
            break

    if cdr_info_path is None:
        return RemapResult(
            entry_name=entry_name,
            success=False,
            error_message=f"CDR info file not found. Tried: {[str(c) for c in cdr_info_candidates]}",
        )

    output_path = output_dir / cif_path.name
    return remap_single_entry(cif_path, cdr_info_path, output_path)


def remap_batch(
    pred_cif_dir: str | Path,
    cdr_info_dir: str | Path,
    output_dir: str | Path,
    pred_cif_pattern: str = "**/*.cif",
    num_workers: int | None = None,
) -> list[RemapResult]:
    """
    Batch remap chain IDs for multiple entries.

    Uses multiprocessing to speed up batch processing. Each CIF file is processed
    independently, making this ideal for parallel execution.

    Args:
        pred_cif_dir: Directory containing Protenix prediction CIF files
        cdr_info_dir: Directory containing CDR info JSON files
        output_dir: Directory to save remapped CIF files
        pred_cif_pattern: Glob pattern for CIF files (default: "**/*.cif")
        num_workers: Number of worker processes. Defaults to number of CPU cores.
            Set to 1 for sequential processing (original behavior).

    Returns:
        List of RemapResult for each entry
    """
    pred_cif_dir = Path(pred_cif_dir)
    cdr_info_dir = Path(cdr_info_dir)
    output_dir = Path(output_dir)

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    cif_files = sorted(pred_cif_dir.glob(pred_cif_pattern))

    if not cif_files:
        return []

    # Determine number of workers
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, num_workers)

    # Prepare arguments for each task
    task_args = [
        (cif_path, cdr_info_dir, output_dir)
        for cif_path in cif_files
    ]

    # Sequential processing if only 1 worker (preserves original behavior)
    if num_workers == 1:
        results = []
        for args in tqdm(task_args, desc="Remapping chain IDs"):
            result = _process_single_cif(args)
            results.append(result)
        return results

    # Parallel processing with multiple workers
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_args = {
            executor.submit(_process_single_cif, args): args
            for args in task_args
        }

        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_args),
            total=len(task_args),
            desc=f"Remapping chain IDs ({num_workers} workers)",
        ):
            result = future.result()
            results.append(result)

    # Sort results by entry name to maintain consistent output order
    results.sort(key=lambda r: r.entry_name)

    return results


if __name__ == "__main__":
    import argparse
    import multiprocessing
    multiprocessing.freeze_support()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Remap chain IDs in Protenix output CIF files")

    # Single file mode
    parser.add_argument("--pred_cif", type=str, help="Path to Protenix prediction CIF file")
    parser.add_argument("--cdr_info", type=str, help="Path to CDR info JSON file")
    parser.add_argument("--output", type=str, help="Path to save remapped CIF file")

    # Batch mode
    parser.add_argument("--pred_cif_dir", type=str, help="Directory containing Protenix prediction CIF files")
    parser.add_argument("--cdr_info_dir", type=str, help="Directory containing CDR info JSON files")
    parser.add_argument("--output_dir", type=str, help="Directory to save remapped CIF files")
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=None,
        help="Number of worker processes for batch mode (default: number of CPU cores). Set to 1 for sequential processing.",
    )

    args = parser.parse_args()

    # Single file mode
    if args.pred_cif and args.cdr_info and args.output:
        result = remap_single_entry(args.pred_cif, args.cdr_info, args.output)
        if result.success:
            print(f"Successfully remapped: {result.output_path}")
            print(f"Chains remapped: {result.chains_remapped}")
        else:
            print(f"Failed: {result.error_message}")

    # Batch mode
    elif args.pred_cif_dir and args.cdr_info_dir and args.output_dir:
        results = remap_batch(
            args.pred_cif_dir,
            args.cdr_info_dir,
            args.output_dir,
            num_workers=args.workers,
        )

        success_count = sum(1 for r in results if r.success)
        fail_count = sum(1 for r in results if not r.success)

        print(f"\nBatch remapping complete: {success_count} succeeded, {fail_count} failed")

        if fail_count > 0:
            print("\nFailed entries:")
            for r in results:
                if not r.success:
                    print(f"  {r.entry_name}: {r.error_message}")

    else:
        parser.print_help()
        print("\nError: Please provide either single file arguments (--pred_cif, --cdr_info, --output)")
        print("       or batch arguments (--pred_cif_dir, --cdr_info_dir, --output_dir)")
