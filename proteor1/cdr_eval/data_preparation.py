#!/usr/bin/env python3
"""
CDR Evaluation Pipeline - Stage 1: Data Preparation

This module handles data preprocessing for the antibody CDR reconstruction evaluation:
1. Parse split JSON files to extract entry information (pdb_id, chains)
2. Validate CIF file existence
3. Generate validated entry lists for downstream processing

Entry Format:
    "{pdb_id}_{heavy_chain}_{light_chain}_{antigen_chains}"

Examples:
    - "7ucf_D_E_G": Heavy=D, Light=E, Antigen=G
    - "7sr3_D__C": Heavy=D, Light=None (Nanobody), Antigen=C
    - "8k6n_S_s_ABC": Heavy=S, Light=s, Antigen=A,B,C

Usage:
    from proteor1.cdr_eval.data_preparation import (
        parse_split_json,
        validate_cif_existence,
    )

    entries = parse_split_json("path/to/split.json")
    valid_entries = validate_cif_existence(entries, "path/to/cif_dir")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


class EntryInfo(TypedDict):
    """Entry information extracted from split JSON."""
    entry: str
    pdb_id: str
    heavy_chain: str
    light_chain: str | None
    antigen_chains: list[str]
    cif_path: str | None


@dataclass
class ParseResult:
    """Result of parsing split JSON file."""
    entries: list[EntryInfo]
    total_count: int
    split_name: str


@dataclass
class ValidationResult:
    """Result of CIF existence validation."""
    valid_entries: list[EntryInfo]
    missing_entries: list[EntryInfo]
    valid_count: int
    missing_count: int
    missing_pdb_ids: list[str] = field(default_factory=list)


def parse_entry_string(entry: str) -> EntryInfo:
    """
    Parse a single entry string into structured information.

    Entry format: "{pdb_id}_{heavy_chain}_{light_chain}_{antigen_chains}"

    Args:
        entry: Entry string like "7ucf_D_E_G" or "7sr3_D__C"

    Returns:
        EntryInfo dict with parsed components

    Examples:
        >>> parse_entry_string("7ucf_D_E_G")
        {'entry': '7ucf_D_E_G', 'pdb_id': '7ucf', 'heavy_chain': 'D',
         'light_chain': 'E', 'antigen_chains': ['G'], 'cif_path': None}

        >>> parse_entry_string("7sr3_D__C")  # Nanobody (no light chain)
        {'entry': '7sr3_D__C', 'pdb_id': '7sr3', 'heavy_chain': 'D',
         'light_chain': None, 'antigen_chains': ['C'], 'cif_path': None}
    """
    parts = entry.split("_")

    if len(parts) < 2:
        raise ValueError(f"Invalid entry format: {entry}. Expected at least pdb_id and heavy_chain.")

    pdb_id = parts[0].lower()  # PDB IDs are case-insensitive, normalize to lowercase
    heavy_chain = parts[1] if len(parts) > 1 else ""
    light_chain_raw = parts[2] if len(parts) > 2 else ""
    antigen_raw = parts[3] if len(parts) > 3 else ""

    # Empty string means no light chain (Nanobody)
    light_chain = light_chain_raw if light_chain_raw else None

    # Antigen chains are concatenated as a single string (e.g., "ABC" -> ["A", "B", "C"])
    antigen_chains = list(antigen_raw) if antigen_raw else []

    return EntryInfo(
        entry=entry,
        pdb_id=pdb_id,
        heavy_chain=heavy_chain,
        light_chain=light_chain,
        antigen_chains=antigen_chains,
        cif_path=None,
    )


def parse_split_json(json_path: str | Path) -> ParseResult:
    """
    Parse split JSON file to extract entry information.

    Args:
        json_path: Path to split JSON file containing list of entry strings

    Returns:
        ParseResult with parsed entries and metadata

    Raises:
        FileNotFoundError: If JSON file does not exist
        json.JSONDecodeError: If JSON is malformed
        ValueError: If entry format is invalid
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"Split JSON file not found: {json_path}")

    logger.info(f"Parsing split JSON: {json_path}")

    with open(json_path, "r") as f:
        raw_entries = json.load(f)

    if not isinstance(raw_entries, list):
        raise ValueError(f"Expected JSON array, got {type(raw_entries).__name__}")

    entries: list[EntryInfo] = []
    failed_entries: list[tuple[str, str]] = []

    for entry_str in raw_entries:
        if not isinstance(entry_str, str):
            logger.warning(f"Skipping non-string entry: {entry_str}")
            continue

        try:
            entry_info = parse_entry_string(entry_str)
            entries.append(entry_info)
        except ValueError as e:
            failed_entries.append((entry_str, str(e)))
            logger.warning(f"Failed to parse entry: {entry_str} - {e}")

    if failed_entries:
        logger.warning(f"Failed to parse {len(failed_entries)} entries")

    split_name = json_path.stem
    result = ParseResult(
        entries=entries,
        total_count=len(entries),
        split_name=split_name,
    )

    logger.info(f"Parsed {len(entries)} entries from {split_name}")

    return result


def validate_cif_existence(
    entries: list[EntryInfo],
    cif_dir: str | Path,
) -> ValidationResult:
    """
    Validate that CIF files exist for all entries.

    Args:
        entries: List of parsed entry information
        cif_dir: Directory containing CIF files

    Returns:
        ValidationResult with valid entries and missing information
    """
    cif_dir = Path(cif_dir)

    if not cif_dir.exists():
        raise FileNotFoundError(f"CIF directory not found: {cif_dir}")

    if not cif_dir.is_dir():
        raise ValueError(f"CIF path is not a directory: {cif_dir}")

    logger.info(f"Validating CIF existence in: {cif_dir}")

    valid_entries: list[EntryInfo] = []
    missing_entries: list[EntryInfo] = []
    missing_pdb_ids: list[str] = []
    seen_pdb_ids: set[str] = set()

    for entry in entries:
        pdb_id = entry["pdb_id"]
        cif_path = cif_dir / f"{pdb_id}.cif"

        if cif_path.exists():
            # Create a new dict with cif_path set
            valid_entry = EntryInfo(
                entry=entry["entry"],
                pdb_id=entry["pdb_id"],
                heavy_chain=entry["heavy_chain"],
                light_chain=entry["light_chain"],
                antigen_chains=entry["antigen_chains"],
                cif_path=str(cif_path),
            )
            valid_entries.append(valid_entry)
        else:
            missing_entries.append(entry)
            if pdb_id not in seen_pdb_ids:
                missing_pdb_ids.append(pdb_id)
                seen_pdb_ids.add(pdb_id)

    result = ValidationResult(
        valid_entries=valid_entries,
        missing_entries=missing_entries,
        valid_count=len(valid_entries),
        missing_count=len(missing_entries),
        missing_pdb_ids=missing_pdb_ids,
    )

    if missing_entries:
        preview = missing_pdb_ids[:5]
        suffix = f"... and {len(missing_pdb_ids) - 5} more" if len(missing_pdb_ids) > 5 else ""
        logger.warning(
            f"Missing {len(missing_entries)} CIF files ({len(missing_pdb_ids)} unique PDB IDs): "
            f"{preview}{suffix}"
        )

    logger.info(
        f"Validation complete: {len(valid_entries)} valid, {len(missing_entries)} missing"
    )

    return result


def load_entries_for_split(
    json_path: str | Path,
    cif_dir: str | Path,
) -> list[EntryInfo]:
    """
    Convenience function to parse and validate entries in one call.

    Args:
        json_path: Path to split JSON file
        cif_dir: Directory containing CIF files

    Returns:
        List of valid entries with cif_path set
    """
    parse_result = parse_split_json(json_path)
    validation_result = validate_cif_existence(parse_result.entries, cif_dir)
    return validation_result.valid_entries


def get_entry_stats(entries: list[EntryInfo]) -> dict:
    """
    Calculate statistics about the entries.

    Args:
        entries: List of parsed entries

    Returns:
        Dictionary with statistics
    """
    total = len(entries)
    with_light_chain = sum(1 for e in entries if e["light_chain"] is not None)
    nanobodies = total - with_light_chain

    antigen_counts: dict[int, int] = {}
    for e in entries:
        n_antigens = len(e["antigen_chains"])
        antigen_counts[n_antigens] = antigen_counts.get(n_antigens, 0) + 1

    unique_pdb_ids = len(set(e["pdb_id"] for e in entries))

    return {
        "total_entries": total,
        "unique_pdb_ids": unique_pdb_ids,
        "with_light_chain": with_light_chain,
        "nanobodies": nanobodies,
        "antigen_count_distribution": antigen_counts,
    }


@dataclass
class DuplicatePdbInfo:
    """Information about a PDB ID with multiple entries."""
    pdb_id: str
    entries: list[EntryInfo]
    entry_count: int
    # Different chain combinations
    chain_combinations: list[dict]  # [{heavy, light, antigen}, ...]


@dataclass
class DuplicateAnalysisResult:
    """Result of duplicate PDB ID analysis."""
    total_entries: int
    unique_pdb_ids: int
    pdb_ids_with_duplicates: int
    total_duplicate_entries: int  # Entries in duplicate PDB IDs
    duplicates: list[DuplicatePdbInfo]
    # Distribution of entry counts per PDB
    entry_count_distribution: dict[int, int]  # {count: num_pdbs}


def analyze_duplicate_pdb_ids(entries: list[EntryInfo]) -> DuplicateAnalysisResult:
    """
    Analyze entries to find PDB IDs with multiple records.

    Same PDB ID can have multiple entries when the structure contains
    multiple antibody-antigen complexes with different chain combinations.

    Args:
        entries: List of parsed entries

    Returns:
        DuplicateAnalysisResult with detailed analysis
    """
    from collections import defaultdict

    # Group entries by pdb_id
    pdb_to_entries: dict[str, list[EntryInfo]] = defaultdict(list)
    for entry in entries:
        pdb_to_entries[entry["pdb_id"]].append(entry)

    # Analyze duplicates
    duplicates: list[DuplicatePdbInfo] = []
    entry_count_distribution: dict[int, int] = defaultdict(int)

    for pdb_id, pdb_entries in pdb_to_entries.items():
        count = len(pdb_entries)
        entry_count_distribution[count] += 1

        if count > 1:
            chain_combinations = [
                {
                    "heavy": e["heavy_chain"],
                    "light": e["light_chain"],
                    "antigen": "".join(e["antigen_chains"]),
                    "entry": e["entry"],
                }
                for e in pdb_entries
            ]
            duplicates.append(
                DuplicatePdbInfo(
                    pdb_id=pdb_id,
                    entries=pdb_entries,
                    entry_count=count,
                    chain_combinations=chain_combinations,
                )
            )

    # Sort by entry count descending
    duplicates.sort(key=lambda x: x.entry_count, reverse=True)

    total_duplicate_entries = sum(d.entry_count for d in duplicates)

    return DuplicateAnalysisResult(
        total_entries=len(entries),
        unique_pdb_ids=len(pdb_to_entries),
        pdb_ids_with_duplicates=len(duplicates),
        total_duplicate_entries=total_duplicate_entries,
        duplicates=duplicates,
        entry_count_distribution=dict(entry_count_distribution),
    )


def print_duplicate_analysis(result: DuplicateAnalysisResult, top_n: int = 10) -> None:
    """
    Print formatted duplicate analysis results.

    Args:
        result: DuplicateAnalysisResult from analyze_duplicate_pdb_ids
        top_n: Number of top duplicates to show in detail
    """
    print("\n" + "=" * 60)
    print("DUPLICATE PDB ID ANALYSIS")
    print("=" * 60)

    print(f"\nOverview:")
    print(f"  Total entries:              {result.total_entries}")
    print(f"  Unique PDB IDs:             {result.unique_pdb_ids}")
    print(f"  PDB IDs with duplicates:    {result.pdb_ids_with_duplicates}")
    print(f"  Entries in duplicate PDBs:  {result.total_duplicate_entries}")

    print(f"\nEntry count distribution (entries per PDB ID):")
    for count in sorted(result.entry_count_distribution.keys()):
        num_pdbs = result.entry_count_distribution[count]
        label = "entry " if count == 1 else "entries"
        print(f"  {count} {label}: {num_pdbs} PDB IDs")

    if result.duplicates:
        print(f"\nTop {min(top_n, len(result.duplicates))} PDB IDs with most entries:")
        print("-" * 60)

        for dup in result.duplicates[:top_n]:
            print(f"\n  {dup.pdb_id.upper()}: {dup.entry_count} entries")
            print(f"  {'─' * 50}")

            # Table header
            print(f"  {'Entry':<25} {'Heavy':>6} {'Light':>6} {'Antigen':>10}")
            print(f"  {'-' * 25} {'-' * 6} {'-' * 6} {'-' * 10}")

            for combo in dup.chain_combinations:
                light_str = combo["light"] if combo["light"] else "-"
                antigen_str = combo["antigen"] if combo["antigen"] else "-"
                print(
                    f"  {combo['entry']:<25} {combo['heavy']:>6} "
                    f"{light_str:>6} {antigen_str:>10}"
                )

        # Summary of all duplicates
        if len(result.duplicates) > top_n:
            remaining = len(result.duplicates) - top_n
            print(f"\n  ... and {remaining} more PDB IDs with duplicates")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Simple test when run directly
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="CDR Evaluation Data Preparation - Parse and analyze split JSON files"
    )
    parser.add_argument(
        "split_json",
        nargs="?",
        help="Path to split JSON file"
    )
    parser.add_argument(
        "cif_dir",
        nargs="?",
        help="Directory containing CIF files"
    )
    parser.add_argument(
        "--analyze-duplicates",
        action="store_true",
        help="Analyze duplicate PDB IDs in the split file (no CIF validation needed)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top duplicates to show in detail (default: 10)"
    )
    args = parser.parse_args()

    # Mode 1: Analyze duplicates only (no CIF validation)
    if args.analyze_duplicates:
        if not args.split_json:
            print("Error: split_json is required for --analyze-duplicates")
            sys.exit(1)

        parse_result = parse_split_json(args.split_json)
        entries = parse_result.entries

        # Basic stats
        stats = get_entry_stats(entries)
        print(f"\nEntry Statistics for {parse_result.split_name}:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Unique PDB IDs: {stats['unique_pdb_ids']}")
        print(f"  With light chain: {stats['with_light_chain']}")
        print(f"  Nanobodies: {stats['nanobodies']}")
        print(f"  Antigen count distribution: {stats['antigen_count_distribution']}")

        # Duplicate analysis
        dup_result = analyze_duplicate_pdb_ids(entries)
        print_duplicate_analysis(dup_result, top_n=args.top_n)

        sys.exit(0)

    # Mode 2: Full validation with CIF directory
    if not args.split_json or not args.cif_dir:
        print("Usage: python data_preparation.py <split_json> <cif_dir>")
        print("       python data_preparation.py <split_json> --analyze-duplicates [--top-n N]")
        print("\nExamples:")
        print("  python data_preparation.py test_entry.json ./cif/")
        print("  python data_preparation.py test_entry.json --analyze-duplicates")
        print("  python data_preparation.py test_entry.json --analyze-duplicates --top-n 20")
        sys.exit(1)

    json_path = args.split_json
    cif_dir = args.cif_dir

    # Parse and validate
    entries = load_entries_for_split(json_path, cif_dir)

    # Print statistics
    stats = get_entry_stats(entries)
    print(f"\nEntry Statistics:")
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Unique PDB IDs: {stats['unique_pdb_ids']}")
    print(f"  With light chain: {stats['with_light_chain']}")
    print(f"  Nanobodies: {stats['nanobodies']}")
    print(f"  Antigen count distribution: {stats['antigen_count_distribution']}")

    # Duplicate analysis
    dup_result = analyze_duplicate_pdb_ids(entries)
    print_duplicate_analysis(dup_result, top_n=args.top_n)

    # Print first 5 entries as sample
    print(f"\nFirst 5 entries:")
    for entry in entries[:5]:
        print(f"  {entry['entry']}: H={entry['heavy_chain']}, L={entry['light_chain']}, "
              f"Ag={entry['antigen_chains']}")
