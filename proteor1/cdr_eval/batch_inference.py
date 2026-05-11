#!/usr/bin/env python3
"""
CDR Evaluation Pipeline - Stage 3: Batch Structure Prediction

This module provides utilities for running batch structure predictions using
Protenix CLI on CDR-masked antibody sequences.

Key features:
- Wraps Protenix's `protenix predict` CLI with subprocess
- Uses --use_default_params to let Protenix handle model-specific settings
- Provides verification utilities for prediction outputs
- Supports skip_existing to avoid redundant predictions

Usage:
    from proteor1.cdr_eval.batch_inference import (
        run_batch_inference_cli,
        verify_prediction_outputs_batch,
        get_best_sample_path,
        load_confidence_scores,
    )

    # Run inference using Protenix CLI
    result = run_batch_inference_cli(
        input_json_dir="data/masked_jsons/",
        output_dir="predictions/",
        model_name="protenix_mini_ism_v0.5.0",
        seeds=[101],
        n_sample=5,
    )

    # Verify outputs
    verification = verify_prediction_outputs_batch(
        prediction_dir="predictions/",
        json_files=list(Path("data/masked_jsons/").glob("*.json")),
        seeds=[101],
    )

Notes:
    - Requires protenix to be installed and available in PATH
    - Uses --use_default_params for model-appropriate settings
    - Outputs are organized by entry_name/seed_XXX/
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Default Configuration Constants
# =============================================================================

DEFAULT_MODEL_NAME = "protenix_mini_ism_v0.5.0"
DEFAULT_N_SAMPLE = 5
DEFAULT_SEEDS = [101]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class InferenceResult:
    """Result of batch inference operation."""
    total_entries: int
    successful_entries: int
    failed_entries: int
    output_dir: str
    failed_list: list[str] = field(default_factory=list)
    error_messages: dict[str, str] = field(default_factory=dict)


@dataclass
class PredictionVerification:
    """Result of prediction output verification."""
    entry: str
    success: bool
    prediction_dir: str | None = None
    best_sample_cif: str | None = None
    confidence_json: str | None = None
    ranking_score: float | None = None
    plddt: float | None = None
    n_samples_found: int = 0
    error_message: str | None = None


@dataclass
class BatchVerificationResult:
    """Result of batch verification."""
    total_entries: int
    verified_count: int
    missing_count: int
    verifications: list[PredictionVerification]
    missing_entries: list[str]


# =============================================================================
# Main Inference Functions
# =============================================================================


def run_batch_inference_cli(
    input_json_dir: str,
    output_dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    seeds: list[int] | None = None,
    n_sample: int = DEFAULT_N_SAMPLE,
    skip_existing: bool = False,
) -> subprocess.CompletedProcess | InferenceResult:
    """
    Run batch structure prediction using Protenix CLI.

    This function calls `protenix predict` via subprocess with
    --use_default_params to let Protenix handle model-specific settings.

    Args:
        input_json_dir: Directory containing CDR-masked JSON files
        output_dir: Directory for prediction outputs
        model_name: Protenix model checkpoint name
        seeds: List of random seeds for reproducibility (default: [101])
        n_sample: Number of diffusion samples per seed
        skip_existing: Skip entries with existing predictions

    Returns:
        subprocess.CompletedProcess on success, or InferenceResult if
        skip_existing is True and all entries already have predictions

    Raises:
        FileNotFoundError: If input directory doesn't exist
        RuntimeError: If no JSON files found in input directory
        subprocess.CalledProcessError: If protenix predict fails
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS.copy()

    input_dir = Path(input_json_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON files found in {input_dir}")

    logger.info(f"Found {len(json_files)} JSON files for inference")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter out existing entries if skip_existing is True
    if skip_existing:
        json_files = _filter_existing_entries(json_files, output_path, seeds)
        logger.info(f"After filtering existing: {len(json_files)} entries to process")

        if not json_files:
            logger.info("All entries already have predictions, nothing to do")
            return InferenceResult(
                total_entries=0,
                successful_entries=0,
                failed_entries=0,
                output_dir=str(output_path),
            )

    # Determine effective input directory
    # If skip_existing filtered some files, create a temp directory with symlinks
    # to only the files that need processing
    effective_input_dir = input_dir
    temp_dir = None

    if skip_existing and len(json_files) < len(list(input_dir.glob("*.json"))):
        # Create temporary directory with symlinks to filtered files only
        temp_dir = tempfile.mkdtemp(prefix="protenix_filtered_")
        effective_input_dir = Path(temp_dir)
        for json_file in json_files:
            link_path = effective_input_dir / json_file.name
            link_path.symlink_to(json_file.resolve())
        logger.info(f"Created temp directory with {len(json_files)} symlinked files: {temp_dir}")

    logger.info(
        f"Starting batch inference:\n"
        f"  Model: {model_name}\n"
        f"  Seeds: {seeds}\n"
        f"  Samples per seed: {n_sample}\n"
        f"  Input: {effective_input_dir}\n"
        f"  Output: {output_path}"
    )

    # Build protenix predict command
    cmd = [
        "protenix", "predict",
        "-i", str(effective_input_dir),
        "-o", str(output_path),
        "-n", model_name,
        "-s", ",".join(map(str, seeds)),
        "-e", str(n_sample),
        "--use_default_params", "True",
    ]

    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        return subprocess.run(cmd, check=True)
    finally:
        # Clean up temporary directory if created
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.info(f"Cleaned up temp directory: {temp_dir}")


def _filter_existing_entries(
    json_files: list[Path],
    output_dir: Path,
    seeds: list[int],
) -> list[Path]:
    """
    Filter out entries that already have prediction outputs.

    Args:
        json_files: List of input JSON file paths
        output_dir: Directory where predictions are stored
        seeds: List of seeds to check

    Returns:
        List of JSON files that need processing
    """
    filtered = []

    for json_file in json_files:
        entry_name = json_file.stem

        # Check if any seed has predictions
        has_prediction = False
        for seed in seeds:
            seed_dir = output_dir / entry_name / f"seed_{seed}"
            if seed_dir.exists():
                # Check for at least one CIF file
                cif_files = list(seed_dir.glob("*.cif"))
                if cif_files:
                    has_prediction = True
                    break

        if not has_prediction:
            filtered.append(json_file)

    return filtered


# =============================================================================
# Verification Functions
# =============================================================================


def verify_prediction_outputs_batch(
    prediction_dir: Path | str,
    json_files: list[Path],
    seeds: list[int],
) -> BatchVerificationResult:
    """
    Verify prediction outputs for a batch of entries.

    Args:
        prediction_dir: Directory containing predictions
        json_files: List of input JSON files
        seeds: List of seeds used for inference

    Returns:
        BatchVerificationResult with verification details
    """
    prediction_dir = Path(prediction_dir)
    verifications = []
    missing_entries = []

    for json_file in json_files:
        entry_name = json_file.stem
        verification = verify_single_entry_prediction(
            prediction_dir=prediction_dir,
            entry_name=entry_name,
            seeds=seeds,
        )
        verifications.append(verification)

        if not verification.success:
            missing_entries.append(entry_name)

    verified_count = sum(1 for v in verifications if v.success)
    missing_count = len(verifications) - verified_count

    return BatchVerificationResult(
        total_entries=len(json_files),
        verified_count=verified_count,
        missing_count=missing_count,
        verifications=verifications,
        missing_entries=missing_entries,
    )


def verify_single_entry_prediction(
    prediction_dir: Path | str,
    entry_name: str,
    seeds: list[int] | None = None,
) -> PredictionVerification:
    """
    Verify prediction output for a single entry.

    Checks:
    1. Entry directory exists
    2. At least one seed directory exists
    3. CIF prediction files exist
    4. Confidence JSON files exist

    Args:
        prediction_dir: Base directory for predictions
        entry_name: Entry identifier (e.g., "7ucf_D_E_G")
        seeds: Expected seeds (default: [101])

    Returns:
        PredictionVerification with details
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    prediction_dir = Path(prediction_dir)
    entry_dir = prediction_dir / entry_name

    if not entry_dir.exists():
        return PredictionVerification(
            entry=entry_name,
            success=False,
            error_message=f"Entry directory not found: {entry_dir}",
        )

    best_sample_info = get_best_sample_info(entry_dir, seeds)

    if best_sample_info is None:
        return PredictionVerification(
            entry=entry_name,
            success=False,
            prediction_dir=str(entry_dir),
            error_message="No valid prediction samples found",
        )

    return PredictionVerification(
        entry=entry_name,
        success=True,
        prediction_dir=str(entry_dir),
        best_sample_cif=best_sample_info.get("cif_path"),
        confidence_json=best_sample_info.get("confidence_json"),
        ranking_score=best_sample_info.get("ranking_score"),
        plddt=best_sample_info.get("plddt"),
        n_samples_found=best_sample_info.get("n_samples", 0),
    )


# =============================================================================
# Best Sample Functions
# =============================================================================


def get_best_sample_info(
    entry_dir: Path,
    seeds: list[int] | None = None,
) -> dict | None:
    """
    Get information about the best sample (highest ranking_score).

    Protenix outputs samples with naming convention:
    - {entry_name}_sample_{N}.cif
    - {entry_name}_summary_confidence_sample_{N}.json

    The best sample (sample_0) should have the highest ranking_score.

    Args:
        entry_dir: Entry's prediction directory
        seeds: List of seeds to search

    Returns:
        Dict with best sample info, or None if no samples found
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    best_sample = None
    best_ranking_score = float("-inf")
    total_samples = 0

    for seed in seeds:
        seed_dir = entry_dir / f"seed_{seed}"
        if not seed_dir.exists():
            continue

        cif_files = list(seed_dir.glob("*_sample_*.cif"))
        total_samples += len(cif_files)

        for cif_file in cif_files:
            stem = cif_file.stem
            if "_sample_" not in stem:
                continue

            sample_num = stem.split("_sample_")[-1]

            confidence_pattern = f"*_summary_confidence_sample_{sample_num}.json"
            confidence_files = list(seed_dir.glob(confidence_pattern))

            if not confidence_files:
                continue

            confidence_json = confidence_files[0]

            try:
                with open(confidence_json) as f:
                    conf_data = json.load(f)
                ranking_score = conf_data.get("ranking_score", 0)
                plddt = conf_data.get("plddt")

                if ranking_score > best_ranking_score:
                    best_ranking_score = ranking_score
                    best_sample = {
                        "cif_path": str(cif_file),
                        "confidence_json": str(confidence_json),
                        "ranking_score": ranking_score,
                        "plddt": plddt,
                        "seed": seed,
                        "sample_num": sample_num,
                        "n_samples": total_samples,
                    }
            except Exception as e:
                logger.warning(f"Failed to load confidence from {confidence_json}: {e}")
                continue

    if best_sample:
        best_sample["n_samples"] = total_samples

    return best_sample


def get_best_sample_path(
    prediction_dir: Path | str,
    entry_name: str,
    seeds: list[int] | None = None,
) -> str | None:
    """
    Get path to the best sample CIF file for an entry.

    Convenience function that returns just the path to the best prediction.

    Args:
        prediction_dir: Base directory for predictions
        entry_name: Entry identifier
        seeds: List of seeds to search

    Returns:
        Path to best sample CIF, or None if not found
    """
    prediction_dir = Path(prediction_dir)
    entry_dir = prediction_dir / entry_name

    if not entry_dir.exists():
        return None

    best_info = get_best_sample_info(entry_dir, seeds)
    return best_info.get("cif_path") if best_info else None


def load_confidence_scores(
    prediction_dir: Path | str,
    entry_name: str,
    seed: int = 101,
    sample_rank: int = 0,
) -> dict | None:
    """
    Load confidence scores from Protenix prediction output.

    Args:
        prediction_dir: Base directory for predictions
        entry_name: Entry identifier
        seed: Random seed used
        sample_rank: Sample rank (0 = best)

    Returns:
        Dict with confidence scores or None if not found
    """
    prediction_dir = Path(prediction_dir)

    confidence_path = (
        prediction_dir / entry_name / f"seed_{seed}" /
        f"{entry_name}_summary_confidence_sample_{sample_rank}.json"
    )

    if not confidence_path.exists():
        return None

    try:
        with open(confidence_path) as f:
            data = json.load(f)

        return {
            "ranking_score": data.get("ranking_score"),
            "plddt": data.get("plddt"),
            "ptm": data.get("ptm"),
            "iptm": data.get("iptm"),
            "has_clash": data.get("has_clash"),
            "seed": seed,
            "sample_rank": sample_rank,
        }
    except Exception as e:
        logger.warning(f"Failed to load confidence from {confidence_path}: {e}")
        return None
