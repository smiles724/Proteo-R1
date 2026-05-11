#!/usr/bin/env python3
"""
CDR Evaluation Pipeline - Stage 2: Chain Extraction

This module extracts antibody-antigen complex chains from CIF files and saves them
as ground truth CIF files for LDDT evaluation.

Key Design Decisions:
    - Uses `auth_asym_id` for chain identification (aligned with cdr_masking.py)
    - Applies standard Protenix Filter preprocessing (removes water, hydrogen, etc.)
    - Preserves original coordinates of retained atoms
    - Output format compatible with downstream CDR scoring

Entry Format:
    "{pdb_id}_{heavy_chain}_{light_chain}_{antigen_chains}"

Examples:
    - "7ucf_D_E_G": Heavy=D, Light=E, Antigen=G
    - "7sr3_D__C": Heavy=D, No Light (Nanobody), Antigen=C
    - "8k6n_S_s_ABC": Heavy=S, Light=s, Antigen=A,B,C

Usage:
    from proteor1.cdr_eval.chain_extractor import (
        extract_chains_from_cif,
        process_entries_batch,
    )

    # Single entry extraction
    result = extract_chains_from_cif(
        cif_path="/path/to/7ucf.cif",
        heavy_chain="D",
        light_chain="E",
        antigen_chains=["G"],
        output_path="/output/7ucf_D_E_G.cif",
    )

    # Batch processing
    results = process_entries_batch(entries, output_dir="/output/gt_cifs/")
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from biotite.structure import AtomArray

logger = logging.getLogger(__name__)


# =============================================================================
# Result Data Classes
# =============================================================================


@dataclass
class ExtractionResult:
    """Result of a single chain extraction operation."""

    success: bool
    entry: str
    pdb_id: str
    output_path: str | None
    error_message: str | None = None
    # Statistics
    n_atoms_original: int = 0
    n_atoms_after_filter: int = 0
    n_atoms_extracted: int = 0
    chains_found: list[str] | None = None
    chains_missing: list[str] | None = None


@dataclass
class BatchResult:
    """Result of batch processing multiple entries."""

    total: int
    success_count: int
    failed_count: int
    results: list[ExtractionResult]
    failed_entries: list[str]


# =============================================================================
# Filter Functions (Adapted from Protenix)
# =============================================================================


def apply_standard_filters(
    atom_array: "AtomArray",
    entity_poly_type: dict[str, str],
    methods: list[str],
) -> "AtomArray":
    """
    Apply standard Protenix preprocessing filters to the atom array.

    This applies the same filters used in protenix.data.json_maker.cif_to_input_json:
        - remove_water: Remove water molecules (HOH, DOD)
        - remove_hydrogens: Remove hydrogen and deuterium atoms
        - mse_to_met: Convert MSE (selenomethionine) to MET
        - remove_element_X: Remove unknown elements and convert ASX/GLX
        - remove_crystallization_aids: Remove crystallization aids (for X-ray structures)

    Note: These filters remove certain atoms but do NOT modify coordinates
    of the retained atoms.

    Args:
        atom_array: Biotite AtomArray object
        entity_poly_type: Dict mapping entity_id to polymer type
        methods: List of experimental methods (e.g., ["X-RAY DIFFRACTION"])

    Returns:
        Filtered AtomArray with standard preprocessing applied
    """
    # Import Filter from protenix
    from protenix.data.filter import Filter
    from protenix.data.parser import MMCIFParser

    # Apply standard filters in order (same as json_maker.cif_to_input_json)
    atom_array = Filter.remove_water(atom_array)
    atom_array = Filter.remove_hydrogens(atom_array)
    atom_array = MMCIFParser.mse_to_met(atom_array)
    atom_array = Filter.remove_element_X(atom_array)

    # Remove crystallization aids for X-ray structures
    if any("DIFFRACTION" in m for m in methods):
        atom_array = Filter.remove_crystallization_aids(atom_array, entity_poly_type)

    return atom_array


# =============================================================================
# Chain Extraction Functions
# =============================================================================


def get_chains_to_extract(
    heavy_chain: str,
    light_chain: str | None,
    antigen_chains: list[str],
) -> list[str]:
    """
    Build list of chain IDs to extract.

    Args:
        heavy_chain: Heavy chain ID (e.g., "D")
        light_chain: Light chain ID or None for nanobody
        antigen_chains: List of antigen chain IDs (e.g., ["A", "B", "C"])

    Returns:
        List of unique chain IDs to extract
    """
    chains = [heavy_chain]
    if light_chain:
        chains.append(light_chain)
    chains.extend(antigen_chains)
    # Remove duplicates while preserving order
    seen = set()
    unique_chains = []
    for c in chains:
        if c not in seen:
            seen.add(c)
            unique_chains.append(c)
    return unique_chains


def extract_chains_by_auth_asym_id(
    atom_array: "AtomArray",
    chain_ids: list[str],
    case_insensitive: bool = True,
    protein_only: bool = True,
) -> tuple["AtomArray", list[str], list[str]]:
    """
    Extract specified chains using auth_asym_id field.

    Uses auth_asym_id for chain identification to align with cdr_masking.py
    which uses the same identifier for CDR region annotation.

    Supports case-insensitive matching for ScFv entries where heavy and light
    chains may use the same letter in different cases (e.g., 'E' and 'e').

    Args:
        atom_array: Biotite AtomArray with auth_asym_id annotation
        chain_ids: List of chain IDs to extract (auth_asym_id values)
        case_insensitive: Whether to match chain IDs case-insensitively (default: True)
        protein_only: If True, exclude HETATM records (ligands, sugars, ions, etc.)
                      Only keep ATOM records (proteins, nucleic acids). Default: True.
                      This is important because ligands often share auth_asym_id with
                      their attached protein chain (e.g., glycosylation sugars).

    Returns:
        Tuple of:
            - Extracted AtomArray containing only specified chains
            - List of chains that were found
            - List of chains that were missing
    """
    # Check if auth_asym_id annotation exists
    if not hasattr(atom_array, "auth_asym_id"):
        raise ValueError(
            "AtomArray does not have auth_asym_id annotation. "
            "Make sure to parse the CIF with extra_fields=['auth_asym_id']"
        )

    # Get available chains in the atom array
    available_chains = list(np.unique(atom_array.auth_asym_id))

    # Build mapping from requested chain_id to actual chain_id in the atom array
    # This handles case-insensitive matching for ScFv entries (e.g., 'E' vs 'e')
    chains_found = []
    chains_missing = []
    chain_id_mapping = {}  # requested -> actual

    for requested_id in chain_ids:
        matched = False
        for actual_id in available_chains:
            if case_insensitive:
                if requested_id.upper() == actual_id.upper():
                    chains_found.append(requested_id)
                    chain_id_mapping[requested_id] = actual_id
                    matched = True
                    break
            else:
                if requested_id == actual_id:
                    chains_found.append(requested_id)
                    chain_id_mapping[requested_id] = actual_id
                    matched = True
                    break
        if not matched:
            chains_missing.append(requested_id)

    if not chains_found:
        raise ValueError(
            f"None of the requested chains found. "
            f"Requested: {chain_ids}, Available: {sorted(available_chains)}"
        )

    # Get the actual chain IDs to use for filtering
    actual_chain_ids = list(chain_id_mapping.values())

    # Create mask for selected chains (using actual chain IDs from the atom array)
    chain_mask = np.isin(atom_array.auth_asym_id, actual_chain_ids)

    # Optionally exclude HETATM records (ligands, sugars, ions, etc.)
    # This is important because ligands often share auth_asym_id with their
    # attached protein chain (e.g., glycosylation sugars like NAG, MAN).
    if protein_only:
        # Prefer chain_mol_type (semantic molecule type) over hetero (HETATM flag)
        # chain_mol_type is set by Protenix parser and is consistent with ESMFeaturizer
        if hasattr(atom_array, "chain_mol_type"):
            atom_mask = atom_array.chain_mol_type == "protein"
        else:
            # Fallback to hetero for non-Protenix parsed structures
            atom_mask = ~atom_array.hetero
        mask = chain_mask & atom_mask
    else:
        mask = chain_mask

    # Extract atoms
    extracted = atom_array[mask]

    return extracted, chains_found, chains_missing


def build_entity_poly_type_for_extracted(
    extracted_array: "AtomArray",
    original_entity_poly_type: dict[str, str],
) -> dict[str, str]:
    """
    Build entity_poly_type dict for the extracted subset.

    Only includes entity IDs that are present in the extracted array.

    Args:
        extracted_array: The extracted AtomArray
        original_entity_poly_type: Original entity_poly_type from parser

    Returns:
        Filtered entity_poly_type containing only relevant entities
    """
    # Get unique entity IDs in the extracted array
    extracted_entities = set(np.unique(extracted_array.label_entity_id))

    # Filter to only include entities present in extracted array
    filtered_poly_type = {
        entity_id: poly_type
        for entity_id, poly_type in original_entity_poly_type.items()
        if entity_id in extracted_entities
    }

    return filtered_poly_type


# =============================================================================
# CIF Writing Functions
# =============================================================================


def save_extracted_to_cif(
    atom_array: "AtomArray",
    entity_poly_type: dict[str, str],
    output_path: str | Path,
    entry_id: str,
) -> None:
    """
    Save extracted atom array to CIF file.

    Uses Protenix's CIFWriter for consistent output format.
    Preserves original auth_asym_id values in the output CIF to ensure
    chain IDs match the original PDB structure.

    Args:
        atom_array: Extracted AtomArray to save
        entity_poly_type: Entity polymer type information
        output_path: Output file path
        entry_id: Entry ID for CIF file (e.g., "7ucf_D_E_G")
    """
    from protenix.data.utils import save_atoms_to_cif

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    atom_array_to_save = copy.deepcopy(atom_array)
    if hasattr(atom_array_to_save, "auth_asym_id"):
        atom_array_to_save.chain_id = atom_array_to_save.auth_asym_id.copy()
        # Also update label_asym_id annotation for entity_poly consistency
        if hasattr(atom_array_to_save, "label_asym_id"):
            atom_array_to_save.set_annotation(
                "label_asym_id",
                atom_array_to_save.auth_asym_id.copy()
            )

    save_atoms_to_cif(
        output_cif_file=str(output_path),
        atom_array=atom_array_to_save,
        entity_poly_type=entity_poly_type,
        pdb_id=entry_id,
    )


# =============================================================================
# Main Extraction Function
# =============================================================================


def extract_chains_from_cif(
    cif_path: str | Path,
    heavy_chain: str,
    light_chain: str | None,
    antigen_chains: list[str],
    output_path: str | Path,
    entry_name: str | None = None,
    apply_filters: bool = True,
) -> ExtractionResult:
    """
    Extract antibody-antigen chains from a CIF file.

    This function:
    1. Parses the CIF file using Protenix's MMCIFParser
    2. Optionally applies standard Filter preprocessing
    3. Extracts chains by auth_asym_id
    4. Saves the result to a new CIF file

    Args:
        cif_path: Path to input CIF file
        heavy_chain: Heavy chain auth_asym_id
        light_chain: Light chain auth_asym_id (None for nanobody)
        antigen_chains: List of antigen chain auth_asym_ids
        output_path: Path for output CIF file
        entry_name: Entry name for logging (default: derived from output_path)
        apply_filters: Whether to apply standard Protenix filters (default: True)

    Returns:
        ExtractionResult with success status and statistics
    """
    from protenix.data.parser import MMCIFParser

    cif_path = Path(cif_path)
    output_path = Path(output_path)
    entry_name = entry_name or output_path.stem

    try:
        # Parse CIF file
        logger.debug(f"Parsing CIF: {cif_path}")
        parser = MMCIFParser(cif_path)

        # Get atom array with auth_asym_id field
        # Note: Using assembly_id="1" and model=1 as defaults
        atom_array = parser.get_structure(
            altloc="first",
            model=1,
            bond_lenth_threshold=None,
        )
        n_atoms_original = len(atom_array)

        # Apply standard filters if requested
        if apply_filters:
            atom_array = apply_standard_filters(
                atom_array=atom_array,
                entity_poly_type=parser.entity_poly_type,
                methods=parser.methods,
            )
        n_atoms_after_filter = len(atom_array)

        # Build list of chains to extract
        chains_to_extract = get_chains_to_extract(
            heavy_chain=heavy_chain,
            light_chain=light_chain,
            antigen_chains=antigen_chains,
        )

        # Extract chains using auth_asym_id
        extracted_array, chains_found, chains_missing = extract_chains_by_auth_asym_id(
            atom_array=atom_array,
            chain_ids=chains_to_extract,
        )
        n_atoms_extracted = len(extracted_array)

        # Log warnings for missing chains
        if chains_missing:
            logger.warning(
                f"{entry_name}: Missing chains {chains_missing}. "
                f"Proceeding with found chains: {chains_found}"
            )

        # Build entity_poly_type for extracted subset
        extracted_entity_poly_type = build_entity_poly_type_for_extracted(
            extracted_array=extracted_array,
            original_entity_poly_type=parser.entity_poly_type,
        )

        # Save to CIF
        save_extracted_to_cif(
            atom_array=extracted_array,
            entity_poly_type=extracted_entity_poly_type,
            output_path=output_path,
            entry_id=entry_name,
        )

        logger.info(
            f"{entry_name}: Extracted {n_atoms_extracted} atoms "
            f"(chains: {chains_found}) -> {output_path}"
        )

        return ExtractionResult(
            success=True,
            entry=entry_name,
            pdb_id=parser.pdb_id,
            output_path=str(output_path),
            n_atoms_original=n_atoms_original,
            n_atoms_after_filter=n_atoms_after_filter,
            n_atoms_extracted=n_atoms_extracted,
            chains_found=chains_found,
            chains_missing=chains_missing,
        )

    except Exception as e:
        logger.error(f"{entry_name}: Extraction failed - {e}")
        return ExtractionResult(
            success=False,
            entry=entry_name,
            pdb_id=cif_path.stem,
            output_path=None,
            error_message=str(e),
        )


# =============================================================================
# Batch Processing
# =============================================================================


def process_entries_batch(
    entries: list[dict],
    output_dir: str | Path,
    apply_filters: bool = True,
    skip_existing: bool = False,
) -> BatchResult:
    """
    Process multiple entries in batch.

    Args:
        entries: List of EntryInfo dicts from data_preparation module.
                 Each dict should have: entry, pdb_id, heavy_chain,
                 light_chain, antigen_chains, cif_path
        output_dir: Directory for output CIF files
        apply_filters: Whether to apply standard Protenix filters
        skip_existing: Skip entries where output file already exists

    Returns:
        BatchResult with summary and individual results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ExtractionResult] = []
    failed_entries: list[str] = []

    total = len(entries)
    logger.info(f"Processing {total} entries...")

    for i, entry_info in enumerate(entries, 1):
        entry_name = entry_info["entry"]
        output_path = output_dir / f"{entry_name}.cif"

        # Skip if exists and skip_existing is True
        if skip_existing and output_path.exists():
            logger.debug(f"[{i}/{total}] Skipping {entry_name} (already exists)")
            results.append(
                ExtractionResult(
                    success=True,
                    entry=entry_name,
                    pdb_id=entry_info["pdb_id"],
                    output_path=str(output_path),
                    error_message="Skipped (already exists)",
                )
            )
            continue

        logger.debug(f"[{i}/{total}] Processing {entry_name}...")

        result = extract_chains_from_cif(
            cif_path=entry_info["cif_path"],
            heavy_chain=entry_info["heavy_chain"],
            light_chain=entry_info["light_chain"],
            antigen_chains=entry_info["antigen_chains"],
            output_path=output_path,
            entry_name=entry_name,
            apply_filters=apply_filters,
        )

        results.append(result)
        if not result.success:
            failed_entries.append(entry_name)

        # Progress logging every 100 entries
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{total} entries processed")

    success_count = sum(1 for r in results if r.success)
    failed_count = len(failed_entries)

    logger.info(
        f"Batch complete: {success_count}/{total} succeeded, "
        f"{failed_count} failed"
    )

    return BatchResult(
        total=total,
        success_count=success_count,
        failed_count=failed_count,
        results=results,
        failed_entries=failed_entries,
    )


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Command-line interface for chain extraction."""
    import argparse
    import json
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Extract antibody-antigen chains from CIF files for CDR evaluation"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Single extraction command
    single_parser = subparsers.add_parser(
        "single", help="Extract chains from a single CIF file"
    )
    single_parser.add_argument("cif_path", help="Path to input CIF file")
    single_parser.add_argument("output_path", help="Path for output CIF file")
    single_parser.add_argument("--heavy", "-H", required=True, help="Heavy chain ID")
    single_parser.add_argument("--light", "-L", default=None, help="Light chain ID (omit for nanobody)")
    single_parser.add_argument(
        "--antigen", "-A", default="", help="Antigen chain IDs (concatenated, e.g., 'ABC')"
    )
    single_parser.add_argument(
        "--no-filter", action="store_true", help="Skip standard Protenix filters"
    )

    # Batch extraction command
    batch_parser = subparsers.add_parser(
        "batch", help="Extract chains from multiple entries"
    )
    batch_parser.add_argument("split_json", help="Path to split JSON file")
    batch_parser.add_argument("cif_dir", help="Directory containing CIF files")
    batch_parser.add_argument("output_dir", help="Directory for output CIF files")
    batch_parser.add_argument(
        "--no-filter", action="store_true", help="Skip standard Protenix filters"
    )
    batch_parser.add_argument(
        "--skip-existing", action="store_true", help="Skip entries with existing output"
    )
    batch_parser.add_argument(
        "--save-report", help="Path to save JSON report of results"
    )

    args = parser.parse_args()

    if args.command == "single":
        antigen_chains = list(args.antigen) if args.antigen else []
        result = extract_chains_from_cif(
            cif_path=args.cif_path,
            heavy_chain=args.heavy,
            light_chain=args.light,
            antigen_chains=antigen_chains,
            output_path=args.output_path,
            apply_filters=not args.no_filter,
        )
        if result.success:
            print(f"Success: {result.output_path}")
            print(f"  Atoms: {result.n_atoms_original} -> {result.n_atoms_after_filter} -> {result.n_atoms_extracted}")
            print(f"  Chains found: {result.chains_found}")
            if result.chains_missing:
                print(f"  Chains missing: {result.chains_missing}")
            sys.exit(0)
        else:
            print(f"Failed: {result.error_message}")
            sys.exit(1)

    elif args.command == "batch":
        from proteor1.cdr_eval.data_preparation import load_entries_for_split

        # Load and validate entries
        entries = load_entries_for_split(args.split_json, args.cif_dir)
        print(f"Loaded {len(entries)} valid entries")

        # Process batch
        batch_result = process_entries_batch(
            entries=entries,
            output_dir=args.output_dir,
            apply_filters=not args.no_filter,
            skip_existing=args.skip_existing,
        )

        # Print summary
        print(f"\nBatch Summary:")
        print(f"  Total:   {batch_result.total}")
        print(f"  Success: {batch_result.success_count}")
        print(f"  Failed:  {batch_result.failed_count}")

        if batch_result.failed_entries:
            print(f"\nFailed entries:")
            for entry in batch_result.failed_entries[:10]:
                print(f"  - {entry}")
            if len(batch_result.failed_entries) > 10:
                print(f"  ... and {len(batch_result.failed_entries) - 10} more")

        # Save report if requested
        if args.save_report:
            report = {
                "total": batch_result.total,
                "success_count": batch_result.success_count,
                "failed_count": batch_result.failed_count,
                "failed_entries": batch_result.failed_entries,
                "results": [
                    {
                        "entry": r.entry,
                        "success": r.success,
                        "output_path": r.output_path,
                        "error_message": r.error_message,
                        "n_atoms_extracted": r.n_atoms_extracted,
                        "chains_found": r.chains_found,
                        "chains_missing": r.chains_missing,
                    }
                    for r in batch_result.results
                ],
            }
            with open(args.save_report, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.save_report}")

        sys.exit(0 if batch_result.failed_count == 0 else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
