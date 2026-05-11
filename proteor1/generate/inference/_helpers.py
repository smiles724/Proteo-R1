#!/usr/bin/env python3
"""
upstream Boltz1 Inference Script

This script runs inference using the pure PyTorch Boltz model implementation.
It loads the upstream Boltz1 checkpoint and generates antibody CDR sequences.

Data flow (matching upstream Boltz1 predict.py):
1. YAML files are parsed and Structure NPZ files are auto-generated to out_dir/processed/structures/
2. CDR positions in YAML (marked as 'X') become UNK token (res_type=22)
3. Model predicts sequences for CDR positions

Supports two execution modes:
1. Single-GPU mode with parallel data loading (DataLoader num_workers)
2. Multi-GPU distributed mode with torchrun (DDP + DistributedSampler)

Usage (Single-GPU, basic):
    python -m proteor1.generate.inference._helpers \
        --data datasets/upstream/test_yaml_dir \
        --checkpoint ckpts/upstream/stage_4.ckpt \
        --out_dir outputs/upstream_test

Usage (with processed MSA):
    python -m proteor1.generate.inference._helpers \
        --data datasets/upstream/test_yaml_dir \
        --checkpoint ckpts/upstream/stage_4.ckpt \
        --out_dir outputs/upstream_test \
        --processed_msa_dir datasets/upstream/msa

Usage (Multi-GPU with torchrun):
    torchrun --nproc_per_node=4 -m proteor1.generate.inference._helpers \
        --data datasets/upstream/test_yaml_dir \
        --checkpoint ckpts/upstream/stage_4.ckpt \
        --out_dir outputs/upstream_test \
        --num_workers 4

With PDB output:
    python -m proteor1.generate.inference._helpers \
        --data datasets/upstream/test_yaml_dir \
        --checkpoint ckpts/upstream/stage_4.ckpt \
        --out_dir outputs/upstream_test \
        --output_format pdb

With AbX_eval evaluation:
    python -m proteor1.generate.inference._helpers \
        --data datasets/upstream/test_yaml_dir \
        --checkpoint ckpts/upstream/stage_4.ckpt \
        --out_dir outputs/upstream_test \
        --output_format pdb \
        --run_eval \
        --ref_dir datasets/upstream/reference_pdb \
        --test_yaml_dir datasets/upstream/test_yaml_dir
"""

from __future__ import annotations

import argparse
import copy as cp
import datetime
import json
import logging
import os
import pickle
import random
import sys
from dataclasses import dataclass, replace, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, from_numpy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import yaml
from tqdm import tqdm


def seed_everything(seed: int):
    """
    Set random seed for reproducibility across all libraries.

    This matches upstream Boltz1's seed_everything() from pytorch_lightning.

    Parameters
    ----------
    seed : int
        Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full reproducibility (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Debug logging flag - controlled by environment variable MFDESIGN_DEBUG_TENSORS
DEBUG_TENSORS = os.environ.get("MFDESIGN_DEBUG_TENSORS", "0") == "1"

# Ensure logger level is INFO when DEBUG_TENSORS is enabled
if DEBUG_TENSORS:
    logger.setLevel(logging.INFO)


def send_to_device(value: Any, device: torch.device | str):
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: send_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [send_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(send_to_device(item, device) for item in value)
    return value


def _ensure_runtime_imports() -> None:
    if "Manifest" in globals():
        return

    global parse_fasta, BoltzWriter, Boltz1, constants
    global Structure, Input, BoltzTokenizer, BoltzFeaturizer
    global Manifest, Record, AntibodyInfo, pad_dim, pad_to_max, load_input, ab_region_type, ag_region_type
    global parse_yaml, Target, const
    global Atom, Residue, Chain, Bond, Connection, Interface, ChainInfo, InterfaceInfo, max_msa_seqs

    from proteor1.generate import Boltz1 as _Boltz1
    from proteor1.generate import constants as _constants
    from proteor1.generate.data_load import (
        AntibodyInfo as _AntibodyInfo,
        BoltzFeaturizer as _BoltzFeaturizer,
        BoltzTokenizer as _BoltzTokenizer,
        Input as _Input,
        Manifest as _Manifest,
        Record as _Record,
        Structure as _Structure,
        Target as _Target,
        ab_region_type as _ab_region_type,
        ag_region_type as _ag_region_type,
        load_input as _load_input,
        pad_dim as _pad_dim,
        pad_to_max as _pad_to_max,
        parse_yaml as _parse_yaml,
    )
    from proteor1.generate.data_load import const as _const
    from proteor1.generate.data_load.parse.fasta import parse_fasta as _parse_fasta
    from proteor1.generate.data_load.types import (
        Atom as _Atom,
        Bond as _Bond,
        Chain as _Chain,
        ChainInfo as _ChainInfo,
        Connection as _Connection,
        Interface as _Interface,
        InterfaceInfo as _InterfaceInfo,
        Residue as _Residue,
    )
    from proteor1.generate.data_load.write.writer import BoltzWriter as _BoltzWriter

    parse_fasta = _parse_fasta
    BoltzWriter = _BoltzWriter
    Boltz1 = _Boltz1
    constants = _constants
    Structure = _Structure
    Input = _Input
    BoltzTokenizer = _BoltzTokenizer
    BoltzFeaturizer = _BoltzFeaturizer
    Manifest = _Manifest
    Record = _Record
    AntibodyInfo = _AntibodyInfo
    pad_dim = _pad_dim
    pad_to_max = _pad_to_max
    load_input = _load_input
    ab_region_type = _ab_region_type
    ag_region_type = _ag_region_type
    parse_yaml = _parse_yaml
    Target = _Target
    const = _const
    Atom = _Atom
    Residue = _Residue
    Chain = _Chain
    Bond = _Bond
    Connection = _Connection
    Interface = _Interface
    ChainInfo = _ChainInfo
    InterfaceInfo = _InterfaceInfo
    max_msa_seqs = getattr(_constants, "max_msa_seqs", 4096)


@dataclass
class BoltzProcessedInput:
    """Processed input data.

    This dataclass matches upstream Boltz1's BoltzProcessedInput from predict.py.
    It contains all the preprocessed data needed for inference:
    - manifest: The Manifest object containing all Record objects
    - targets_dir: Path to the directory containing structure NPZ files
    - msa_dir: Path to the directory containing processed MSA NPZ files
    """

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    temperature: float = 1.0
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True
    noise_type: str = "discrete_absorb"


@dataclass
class AF3DiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.0
    rho: float = 7
    step_scale: float = 1.0
    temperature: float = 1.0
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True
    noise_type: str = "discrete_absorb"


def check_inputs(
        data: Path,
        outdir: Path,
        override: bool = False,
):
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.

    """
    # Check if data is a directory
    if data.is_dir():  # if the input data is a directory, expand the directory to all files in this directory
        data: list[Path] = list(
            data.glob("*"))  # data.glob("*") returns a list of all files in the directory, each file is a Path object

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        filtered_data = []  # filtered_data is a list of data with allowed file types
        for d in data:
            if d.suffix in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                filtered_data.append(d)
            elif d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

        data = filtered_data
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if
                e.is_dir()}  # existing is a set of the names of directories in the predictions directory

    # Remove them from the input data
    msg = None
    if existing and not override:
        # if there are existing predictions and the override flag is not set to True, we will skip the input data that has already been predicted
        data = [d for d in data if d.stem not in existing]
        # remove the existing predictions from the input data
        num_skipped = len(existing) - len(data)
        # num_skipped is the number of existing predictions that will be skipped
        msg = (
            f"Found some existing predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
    elif existing and override:
        # override is set to True, we will override the existing predictions
        msg = "Found existing predictions, will override."

    return data, msg


def log_tensor_stats(tensor_name: str, tensor, sample_name: str = "", is_mask: bool = False, unk_value: int = None):
    """
    Log tensor statistics for debugging tensor comparisons between codebases.

    Parameters
    ----------
    tensor_name : str
        Name of the tensor being logged.
    tensor : np.ndarray or torch.Tensor
        The tensor to analyze.
    sample_name : str
        Sample identifier for context.
    is_mask : bool
        If True, log mask-specific stats (True count and first N True indices).
    unk_value : int
        If provided, log count and indices of positions equal to this value.
    """
    if not DEBUG_TENSORS:
        return

    # Convert to numpy if torch tensor
    if hasattr(tensor, 'cpu'):
        arr = tensor.cpu().numpy()
    else:
        arr = np.asarray(tensor)

    prefix = f"[DEBUG] Sample: {sample_name}" if sample_name else "[DEBUG]"

    if is_mask:
        # For boolean masks
        arr_bool = arr.astype(bool)
        true_count = arr_bool.sum()
        true_indices = np.where(arr_bool.flatten())[0]
        first_10_indices = true_indices[:10].tolist()
        logger.info(f"{prefix} {tensor_name}: True count={true_count}, first 10 True indices={first_10_indices}")
    elif unk_value is not None:
        # For sequences, find UNK positions
        unk_mask = (arr == unk_value)
        unk_count = unk_mask.sum()
        unk_indices = np.where(unk_mask.flatten())[0]
        first_10_indices = unk_indices[:10].tolist()
        logger.info(f"{prefix} {tensor_name}: UNK({unk_value}) count={unk_count}, first 10 UNK indices={first_10_indices}")
    else:
        # For categorical tensors (region_type, chain_type)
        unique_vals, counts = np.unique(arr.flatten(), return_counts=True)
        logger.info(f"{prefix} {tensor_name}: unique={unique_vals.tolist()}, counts={counts.tolist()}")


def log_dataset_features(features: Dict, sample_name: str):
    """
    Log all key tensor features from dataset __getitem__ for comparison.

    Parameters
    ----------
    features : Dict
        Features dictionary from dataset.
    sample_name : str
        Sample name for identification.
    """
    if not DEBUG_TENSORS:
        return

    logger.info(f"[DEBUG] ========== Dataset Features: {sample_name} ==========")

    # region_type: distribution
    if "region_type" in features:
        log_tensor_stats("region_type", features["region_type"], sample_name)

    # chain_type: distribution
    if "chain_type" in features:
        log_tensor_stats("chain_type", features["chain_type"], sample_name)

    # cdr_mask: True count and positions
    if "cdr_mask" in features:
        log_tensor_stats("cdr_mask", features["cdr_mask"], sample_name, is_mask=True)

    # seq_mask: True count and positions
    if "seq_mask" in features:
        log_tensor_stats("seq_mask", features["seq_mask"], sample_name, is_mask=True)

    # masked_seq: UNK (22) positions
    if "masked_seq" in features:
        log_tensor_stats("masked_seq", features["masked_seq"], sample_name, unk_value=22)

    logger.info(f"[DEBUG] ========== End Dataset Features: {sample_name} ==========")


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="upstream Boltz1 Inference Script")

    # Input/output (matching upstream Boltz1 predict.py)
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to input YAML file or directory"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./predictions",
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to upstream Boltz1 checkpoint"
    )

    # MSA options (matching upstream Boltz1 predict.py)
    parser.add_argument(
        "--msa_dir", type=str, default=None,
        help="Path to raw MSA CSV files directory. If not provided, defaults to out_dir/msa"
    )
    parser.add_argument(
        "--processed_msa_dir", type=str, default=None,
        help="Path to processed MSA NPZ files directory. If not provided, defaults to out_dir/processed/msa"
    )

    # Ground truth structure for inpainting (matching upstream Boltz1 predict.py)
    parser.add_argument(
        "--ground_truth_structure_dir",
        "--ground_truth_dir",
        dest="ground_truth_structure_dir",
        type=str,
        default="./datasets/upstream/antibody_data/structures",
        help="Path to ground truth structure directory for inpainting/oracle evaluation",
    )
    parser.add_argument(
        "--structure_inpainting", action="store_true",
        help="Whether to perform structure inpainting"
    )
    parser.add_argument(
        "--gt_oracle_cdr",
        action="store_true",
        help="Use ground-truth CDR residue types and coordinates as input-side oracle features",
    )

    # Model settings
    parser.add_argument(
        "--accelerator", type=str, default="gpu",
        choices=["gpu", "cpu"],
        help="Accelerator type"
    )

    # Inference settings
    parser.add_argument(
        "--recycling_steps", type=int, default=3,
        help="Number of recycling steps"
    )
    parser.add_argument(
        "--sampling_steps", type=int, default=200,
        help="Number of diffusion sampling steps"
    )
    parser.add_argument(
        "--diffusion_samples", type=int, default=1,
        help="Number of diffusion samples per input (upstream Boltz1 default: 1)"
    )
    # Diffusion sampling parameters (BoltzDiffusionParams from upstream Boltz1 predict.py)
    # These are optimized for inference and differ from training defaults
    parser.add_argument(
        "--step_scale", type=float, default=1.638,
        help="Step size scale for diffusion (upstream Boltz1 inference: 1.638, training: 1.0)"
    )
    parser.add_argument(
        "--gamma_0", type=float, default=0.605,
        help="Gamma_0 for SDE stochasticity (upstream Boltz1 inference: 0.605, training: 0.8)"
    )
    parser.add_argument(
        "--gamma_min", type=float, default=1.107,
        help="Gamma_min threshold (upstream Boltz1 inference: 1.107, training: 1.0)"
    )
    parser.add_argument(
        "--noise_scale", type=float, default=0.901,
        help="Noise injection scale (upstream Boltz1 inference: 0.901, training: 1.0)"
    )
    parser.add_argument(
        "--rho", type=float, default=8.0,
        help="Rho for noise schedule (upstream Boltz1 inference: 8.0, training: 7.0)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Temperature for sequence sampling"
    )
    parser.add_argument(
        "--noise_type", type=str, default="discrete_absorb",
        choices=["discrete_absorb", "discrete_uniform", "continuous"],
        help="Noise type for diffusion process"
    )
    parser.add_argument(
        "--only_structure_prediction", action="store_true",
        help="Only perform structure prediction (no sequence generation)"
    )
    parser.add_argument(
        "--write_full_pae", action="store_true",
        help="Write full PAE matrix to output"
    )
    parser.add_argument(
        "--write_full_pde", action="store_true",
        help="Write full PDE matrix to output"
    )
    parser.add_argument(
        "--max_samples", type=int, default=-1,
        help="Maximum number of samples to process (-1 for all)"
    )

    # Output format options
    parser.add_argument(
        "--output_format", type=str, default="pdb",
        choices=["npz", "pdb", "both"],
        help="Output format for structures (npz, pdb, or both)"
    )

    # AbX_eval integration options
    parser.add_argument(
        "--run_eval", action="store_true",
        help="Run AbX_eval evaluation after inference"
    )
    parser.add_argument(
        "--ref_dir", type=str, default=None,
        help="Reference PDB directory for evaluation"
    )
    parser.add_argument(
        "--test_yaml_dir", type=str, default=None,
        help="Test YAML directory for CDR mask extraction (for eval)"
    )
    parser.add_argument(
        "--test_json_fpath", type=str, default=None,
        help="Path to test list JSON file (regular_list.json or nano_list.json)"
    )
    parser.add_argument(
        "--eval_cpus", type=int, default=4,
        help="Number of CPUs for parallel evaluation"
    )

    # Distributed inference options
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of DataLoader workers for parallel data loading"
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=2,
        help="Number of batches to prefetch per worker"
    )

    # Random seed for reproducibility (matching upstream Boltz1 default)
    parser.add_argument(
        "--seed", type=int, default=2025,
        help="Random seed for reproducibility (default: 2025 to match upstream Boltz1)"
    )

    # Cache directory for CCD and model files (matching upstream Boltz1 predict.py)
    parser.add_argument(
        "--cache", type=str, default="ckpts/upstream",
        help="Path to cache directory for CCD and model files"
    )

    # Override existing predictions (matching upstream Boltz1 predict.py)
    parser.add_argument(
        "--override", action="store_true",
        help="Override existing predictions"
    )

    # Use epitope region information (matching upstream Boltz1 predict.py)
    parser.add_argument(
        "--use_epitope", action="store_true", default=True,
        help="Use epitope region information for antigen"
    )
    parser.add_argument(
        "--no_epitope", action="store_false", dest="use_epitope",
        help="Disable epitope region support"
    )

    # Preprocessed data path (optional legacy dev hook; safe to omit).
    parser.add_argument(
        "--preprocessed_data_path", type=str, default=None,
        help="Optional JSON file with per-entry preprocessed metadata (legacy; safe to omit)"
    )

    # CDR JSON output options (enabled by default)
    parser.add_argument(
        "--no_cdr_json", action="store_true", default=False,
        help="Disable CDR JSON output (enabled by default)"
    )

    parsed_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(parsed_argv)
    gt_dir_provided = any(
        arg == "--ground_truth_dir"
        or arg.startswith("--ground_truth_dir=")
        or arg == "--ground_truth_structure_dir"
        or arg.startswith("--ground_truth_structure_dir=")
        for arg in parsed_argv
    )
    if args.gt_oracle_cdr and not gt_dir_provided:
        parser.error("--gt_oracle_cdr requires --ground_truth_dir")

    return args


def load_yaml_input(yaml_path: Path) -> Dict:
    """Load input from YAML file."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


class PredictionDataset(torch.utils.data.Dataset):
    """Base iterable dataset."""

    def __init__(
            self,
            manifest: Manifest,
            target_dir: Path,
            msa_dir: Path,
            inpaint: bool = False,
            ground_truth_dir: Optional[Path] = None,
            gt_oracle_cdr: bool = False,
            use_epitope: bool = True
    ) -> None:
        """Initialize the training dataset.

        Parameters
        ----------
        manifest : Manifest
            The manifest to load data from.
        target_dir : Path
            The path to the target directory.
        msa_dir : Path
            The path to the msa directory.

        """
        _ensure_runtime_imports()
        super().__init__()
        self.manifest = manifest
        self.target_dir = target_dir
        self.msa_dir = msa_dir
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()
        self.inpaint = inpaint
        self.ground_truth_dir = Path(ground_truth_dir) if ground_truth_dir is not None else None
        self.gt_oracle_cdr = gt_oracle_cdr
        if self.gt_oracle_cdr and self.ground_truth_dir is None:
            raise ValueError("gt_oracle_cdr requires ground_truth_dir")
        self.use_epitope = use_epitope

    def _load_ground_truth(
            self,
            record: Record,
            token_count: Optional[int] = None,
    ) -> tuple[Structure, np.ndarray]:
        ground_truth = np.load(self.ground_truth_dir / f"{record.id}.npz")
        ground_truth = Structure(
            atoms=ground_truth["atoms"],
            bonds=ground_truth["bonds"],
            residues=ground_truth["residues"],
            chains=ground_truth["chains"],
            connections=ground_truth["connections"],
            interfaces=ground_truth["interfaces"],
            mask=ground_truth["mask"],
        )
        ground_truth_tokens = self.tokenizer.tokenize(Input(ground_truth, {}))[0].tokens
        if token_count is not None:
            ground_truth_tokens = ground_truth_tokens[:token_count]
        return ground_truth, ground_truth_tokens

    @staticmethod
    def _validate_ground_truth_token_alignment(
            record_id: str,
            input_tokens: np.ndarray,
            ground_truth_tokens: np.ndarray,
            allow_res_type_mismatch_mask: Optional[np.ndarray] = None,
    ) -> None:
        if len(input_tokens) != len(ground_truth_tokens):
            raise ValueError(
                f"{record_id}: ground truth token count mismatch "
                f"(input={len(input_tokens)}, gt={len(ground_truth_tokens)})"
            )

        if allow_res_type_mismatch_mask is None:
            allow_res_type_mismatch_mask = np.zeros(len(input_tokens), dtype=bool)
        elif len(allow_res_type_mismatch_mask) != len(input_tokens):
            raise ValueError(
                f"{record_id}: oracle mismatch mask length "
                f"{len(allow_res_type_mismatch_mask)} does not match input token count {len(input_tokens)}"
            )

        for i, (token, ground_truth_token) in enumerate(zip(input_tokens, ground_truth_tokens)):
            if token["atom_num"] != ground_truth_token["atom_num"]:
                raise ValueError(
                    f"{record_id}: ground truth token {i} atom_num mismatch "
                    f"(input={token['atom_num']}, gt={ground_truth_token['atom_num']})"
                )
            if token["res_idx"] != ground_truth_token["res_idx"]:
                raise ValueError(
                    f"{record_id}: ground truth token {i} res_idx mismatch "
                    f"(input={token['res_idx']}, gt={ground_truth_token['res_idx']})"
                )
            if token["asym_id"] != ground_truth_token["asym_id"]:
                raise ValueError(
                    f"{record_id}: ground truth token {i} asym_id mismatch "
                    f"(input={token['asym_id']}, gt={ground_truth_token['asym_id']})"
                )
            if (
                    not allow_res_type_mismatch_mask[i]
                    and token["res_type"] != ground_truth_token["res_type"]
            ):
                raise ValueError(
                    f"{record_id}: ground truth token {i} res_type mismatch "
                    f"(input={token['res_type']}, gt={ground_truth_token['res_type']})"
                )

    def __getitem__(self, idx: int) -> dict:
        """Get an item from the dataset.

        Returns
        -------
        Dict[str, Tensor]
            The sampled data features.

        """
        # Get a sample from the dataset
        record = self.manifest.records[idx]

        # Get the structure
        try:
            input_data = load_input(record, self.target_dir, self.msa_dir)
        except Exception as e:  # noqa: BLE001
            print(f"Failed to load input for {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        # Tokenize structure
        try:
            tokenized, spec_token_mask = self.tokenizer.tokenize(input_data)
        except Exception as e:  # noqa: BLE001
            print(f"Tokenizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)
        seq_mask = np.zeros_like(tokenized.tokens["res_type"], dtype=bool)
        if isinstance(record.structure, AntibodyInfo):
            seq_mask[(tokenized.tokens["res_type"] == 22) &
                     (tokenized.tokens["asym_id"] == record.structure.H_chain_id)] = True
            seq_mask[(tokenized.tokens["res_type"] == 22) &
                     (tokenized.tokens["asym_id"] == record.structure.L_chain_id)] = True

        ground_truth = None
        ground_truth_tokens = None
        if self.gt_oracle_cdr:
            try:
                ground_truth, ground_truth_tokens = self._load_ground_truth(record)
                self._validate_ground_truth_token_alignment(
                    record.id,
                    tokenized.tokens,
                    ground_truth_tokens,
                    allow_res_type_mismatch_mask=seq_mask,
                )
                tokenized.tokens["res_type"][seq_mask] = ground_truth_tokens["res_type"][seq_mask]
            except Exception as e:
                raise ValueError(
                    f"Failed to load aligned ground truth oracle for {record.id}: {e}"
                ) from e

        if self.inpaint:
            # Load the ground truth
            try:
                if ground_truth is None or ground_truth_tokens is None:
                    ground_truth, ground_truth_tokens = self._load_ground_truth(record, len(tokenized.tokens))

                for i, (token, ground_truth_token) in enumerate(zip(tokenized.tokens, ground_truth_tokens)):
                    if spec_token_mask[i]:
                        continue

                    assert token["atom_num"] == ground_truth_token["atom_num"]
                    assert token["res_idx"] == ground_truth_token["res_idx"]
                    if not (self.gt_oracle_cdr and seq_mask[i]):
                        assert token["res_type"] == ground_truth_token["res_type"]
                    assert token["asym_id"] == ground_truth_token["asym_id"]

                coord_data = []
                resolved_mask = []
                coord_mask = []
                for i, token in enumerate(ground_truth_tokens):
                    start = token["atom_idx"]
                    end = token["atom_idx"] + token["atom_num"]
                    token_atoms = ground_truth.atoms[start:end]
                    if len(token_atoms) < tokenized.tokens[i]["atom_num"]:
                        token_atoms = np.concatenate([token_atoms,
                                                      np.zeros(tokenized.tokens[i]["atom_num"] - len(token_atoms),
                                                               dtype=token_atoms.dtype)])
                    coord_data.append(np.array([token_atoms["coords"]]))
                    resolved_mask.append(token_atoms["is_present"])
                    if seq_mask[i]:
                        coord_mask.append(np.ones_like(token_atoms["is_present"], dtype=bool))
                    else:
                        coord_mask.append(1 - token_atoms["is_present"])

                resolved_mask = from_numpy(np.concatenate(resolved_mask))
                coord_mask = from_numpy(np.concatenate(coord_mask))
                coords = from_numpy(np.concatenate(coord_data, axis=1))

                assert (len(coord_mask) == len(resolved_mask))
                assert (len(coord_mask) == coords.shape[1])

                center = (coords * resolved_mask[None, :, None]).sum(dim=1)
                center = center / resolved_mask.sum().clamp(min=1)
                coords = coords - center[:, None]

                atoms_per_window_queries = 32
                pad_len = (
                                  (len(resolved_mask) - 1) // atoms_per_window_queries + 1
                          ) * atoms_per_window_queries - len(resolved_mask)
                coords = pad_dim(coords, 1, pad_len)
                coord_mask = pad_dim(coord_mask, 0, pad_len)
                resolved_mask = pad_dim(resolved_mask, 0, pad_len)
            except Exception as e:
                print(f"Failed to load ground truth for {record.id} with error {e}. Skipping.")
                return self.__getitem__(0)
        else:
            coords = coord_mask = resolved_mask = None

        if isinstance(record.structure, AntibodyInfo):
            h_region_type = ab_region_type(tokenized.tokens, spec_token_mask, record.structure.H_chain_id)
            l_region_type = ab_region_type(tokenized.tokens, spec_token_mask, record.structure.L_chain_id)
            ag_region_types = ag_region_type(tokenized.tokens, spec_token_mask,
                                             [record.structure.H_chain_id, record.structure.L_chain_id],
                                             self.use_epitope)
            region_type = h_region_type + l_region_type + ag_region_types

        assert len(region_type) == len(spec_token_mask)

        # Inference specific options
        options = record.inference_options
        if options is None:
            binders, pocket = None, None
        else:
            binders, pocket = options.binders, options.pocket

        if isinstance(record.structure, AntibodyInfo):
            indices = [i for i, x in enumerate(tokenized.tokens) if
                       x["asym_id"] in [record.structure.H_chain_id, record.structure.L_chain_id]]
            cdr_token_mask = np.zeros_like(spec_token_mask, dtype=bool)
            cdr_token_mask[indices] = spec_token_mask[indices]
            chain_type = torch.ones_like(from_numpy(tokenized.tokens["asym_id"])).long() * 3
            chain_type[tokenized.tokens["asym_id"] == record.structure.H_chain_id] = 1
            chain_type[tokenized.tokens["asym_id"] == record.structure.L_chain_id] = 2

        # Compute features
        try:
            features = self.featurizer.process(
                tokenized,
                training=False,
                max_atoms=None,
                max_tokens=None,
                max_seqs=max_msa_seqs,
                pad_to_max_seqs=False,
                symmetries={},
                compute_symmetries=False,
                inference_binder=binders,
                inference_pocket=pocket,
                gt_oracle_cdr_mask=seq_mask if self.gt_oracle_cdr else None,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Featurizer failed on {record.id} with error {e}. Skipping.")  # noqa: T201
            return self.__getitem__(0)

        if coords is not None:
            features["coords_gt"] = coords
            features["coord_mask"] = coord_mask
            assert features["atom_resolved_mask"].shape == resolved_mask.shape
            features["atom_resolved_mask"] = resolved_mask
            assert features["coords"].shape == features["coords_gt"].shape

        features["record"] = record
        features["masked_seq"] = from_numpy(cp.deepcopy(tokenized.tokens["res_type"])).long()
        features["pdb_id"] = torch.tensor([ord(c) for c in record.id])
        features["seq_mask"] = from_numpy(seq_mask).bool()
        features["cdr_mask"] = from_numpy(cdr_token_mask).bool()
        features["attn_mask"] = torch.ones_like(features["cdr_mask"]).bool()
        features["region_type"] = from_numpy(region_type).long()
        features["chain_type"] = chain_type

        # Debug logging: log tensor features before returning
        log_dataset_features(features, record.id)

        return features

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.

        """
        return len(self.manifest.records)


def collate(data: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    """Collate the data.

    Parameters
    ----------
    data : List[Dict[str, Tensor]]
        The data to collate.

    Returns
    -------
    Dict[str, Tensor]
        The collated data.

    """
    _ensure_runtime_imports()
    # Get the keys
    keys = data[0].keys()

    # Collate the data
    collated = {}
    for key in keys:
        values = [d[key] for d in data]

        if key not in [
            "all_coords",
            "all_resolved_mask",
            "crop_to_all_atom_map",
            "chain_symmetries",
            "amino_acids_symmetries",
            "ligand_symmetries",
            "record",
        ]:
            # Check if all have the same shape
            shape = values[0].shape
            if not all(v.shape == shape for v in values):
                values, _ = pad_to_max(values, 0)
            else:
                values = torch.stack(values, dim=0)

        # Stack the values
        collated[key] = values

    return collated


def process_inputs(  # noqa: C901, PLR0912, PLR0915
        data: list[Path],
        out_dir: Path,
        ccd_path: Path,
        msa_dir: Path,
        preprocessed_data_path: str,
        msa_filtering_threshold: float = None,
        max_msa_seqs: int = 4096,
        processed_msa_dir: Optional[Path] = None,
        use_msa_server: bool = False,
        msa_server_url: str = None,
        msa_pairing_strategy: str = None,
        only_process_msa: bool = False,
        generate_msa: bool = False
) -> dict:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    _ensure_runtime_imports()
    # Create output directories
    structure_dir = out_dir / "processed" / "structures"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    # Optional preprocessed data file (legacy dev hook; current parse path
    # does not consume it — all downstream uses are commented out below).
    preprocessed_data = {}
    if preprocessed_data_path and Path(preprocessed_data_path).is_file():
        with open(preprocessed_data_path, "r") as file:
            preprocessed_data = json.load(file)

    # Parse input data
    records: list[Record] = []
    chain_infos = {}
    for path in tqdm(data):
        # Parse data
        if path.suffix in (".fa", ".fas", ".fasta"):
            target = parse_fasta(path, ccd)
        elif path.suffix in (".yml", ".yaml"):
            target, now_chain_infos = parse_yaml(path, ccd)
            chain_infos[target.record.id] = now_chain_infos
        elif path.is_dir():
            msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
            raise RuntimeError(msg)
        else:
            msg = (
                f"Unable to parse filetype {path.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)

        # Get target id
        target_id = target.record.id
        # target_id the file name of the yaml file in our case

        # Get all MSA ids and decide whether to generate MSA
        to_generate = {}
        prot_id = const.chain_type_ids["PROTEIN"]
        # prot_id is the id of the protein chain type, i.e. 0, in our ab design, chains are always protein chains, prot_id is always 0

        for chain in target.record.chains:
            # Add to generate list, assigning entity id
            if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                # msa_id==0 means we have not generated the msa for this chain yet
                # else, msa_id is the precomputed msa file path
                entity_id = chain.entity_id
                # obtain the entity id of the chain. Note that may be several chains with the same entity id if their sequences are the same

                msa_id = f"{target_id}_{entity_id}"
                # target_id is the file name of the yaml file in our case
                # entity_id is the entity id of the chain
                gt_seq = chain_infos[target.record.id]["entity_to_gt"][entity_id]
                # Use ground truth sequence to compute MSA for better quality, since it has more coevolutionary coupling when pairing
                to_generate[msa_id] = gt_seq if gt_seq else target.sequences[entity_id]
                chain.msa_id = msa_dir / f"{msa_id}.csv"

            # We do not support msa generation for non-protein chains
            elif chain.msa_id == 0:
                # this branch means the chain is not a protein chain, we will not encounter this branch in our ab design
                chain.msa_id = -1

        # to_generate is not empty means we have not generated the msa for some chains; we need to check set the use_msa_server flag to True
        # Generate MSA
        if to_generate and not use_msa_server and generate_msa:
            msg = "Missing MSA's in input and --use_msa_server flag not set."
            raise RuntimeError(msg)

        # # to_generate is not empty means we have not generated the msa for some chains; then we need to generate the msa for these chains
        if to_generate and generate_msa:
            # msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
            # print(msg)
            # compute_msa(
            #     data=to_generate,
            #     target_id=target_id,
            #     msa_dir=msa_dir,
            #     msa_server_url=msa_server_url,
            #     msa_pairing_strategy=msa_pairing_strategy,
            # )
            raise NotImplementedError

        # Parse MSA data
        msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
        # msas is a list of the msa paths and sorted by the entity id

        msa_id_map = {}
        for msa_idx, msa_id in enumerate(msas):
            # Check that raw MSA exists
            msa_path = Path(msa_id)
            # Dump processed MSA
            processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
            msa_id_map[msa_id] = f"{target_id}_{msa_idx}"

            # check if the msa file exists, if not, raise an error
            if (not processed.exists()) and (not msa_path.exists()):
                msg = f"Processed MSA file {processed} not found."
                raise FileNotFoundError(msg)
            # check if the processed msa file exists, if not, we need to parse the msa file
            if not processed.exists():
                # # in our ab design, the msa file is always a csv file
                # if msa_path.suffix == ".csv":
                #     if msa_filtering_threshold is not None:
                #         msa: MSA = parse_csv_for_ab_design(
                #             msa_path, max_seqs=max_msa_seqs,
                #             entry_info=preprocessed_data[target_id],
                #             msa_filtering_threshold=msa_filtering_threshold
                #         )[0]
                #     else:
                #         msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                # else:
                #     msg = f"MSA file {msa_path} not supported, only csv."
                #     raise RuntimeError(msg)
                #
                # msa.dump(processed)
                raise NotImplementedError

        if only_process_msa:
            continue

        for c in target.record.chains:
            if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                c.msa_id = msa_id_map[c.msa_id]

        # Keep record
        records.append(target.record)

        # Dump structure
        struct_path = structure_dir / f"{target.record.id}.npz"
        target.structure.dump(struct_path)

    if only_process_msa:
        return

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")

    if len(chain_infos) > 0:
        with open(out_dir / "processed" / "chain_infos.json", "w") as f:
            json.dump(chain_infos, f)


def move_features_to_device(
    features: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Move feature tensors to device and add batch dimension.

    Parameters
    ----------
    features : Dict[str, torch.Tensor]
        Dictionary of feature tensors on CPU.
    device : torch.device
        Target device.

    Returns
    -------
    Dict[str, torch.Tensor]
        Features with batch dimension on target device.
    """
    result = {}
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.unsqueeze(0).to(device)
        else:
            result[key] = value
    return result


####################################################################################################
# DISTRIBUTED UTILITIES
####################################################################################################


def setup_distributed() -> Tuple[int, int, bool]:
    """
    Setup distributed training environment.

    Returns
    -------
    Tuple[int, int, bool]
        - rank: Process rank (0 if not distributed)
        - world_size: Total number of processes (1 if not distributed)
        - is_distributed: Whether running in distributed mode
    """
    # Check if running with torchrun/distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Initialize process group
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

        return rank, world_size, True
    else:
        return 0, 1, False


def cleanup_distributed():
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_local_rank() -> int:
    """Get local rank for GPU selection."""
    return int(os.environ.get("LOCAL_RANK", 0))


def main():
    """
    Main entry point for upstream Boltz1 inference.

    Supports two modes:
    1. Single-GPU mode: python proteor1-design ...
    2. Multi-GPU mode: torchrun --nproc_per_node=N proteor1-design ...

    In multi-GPU mode, samples are distributed across GPUs using DistributedSampler.
    Data preprocessing is parallelized using DataLoader with num_workers.
    """
    args = parse_args()

    if args.structure_inpainting and args.ground_truth_structure_dir is None:
        logger.error("Please provide the ground truth structure directory if inpainting.")
        return

    if args.ground_truth_structure_dir is not None:
        args.ground_truth_structure_dir = Path(args.ground_truth_structure_dir).expanduser()

    # Set no grad
    torch.set_grad_enabled(False)
    # disable gradient computation for inference

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set random seed for reproducibility (CRITICAL for consistent predictions)
    # This matches upstream Boltz1's seed_everything(seed) call in predict.py
    if args.seed is not None:
        seed_everything(args.seed)
        logger.info(f"Set random seed to {args.seed} for reproducibility")

    # Setup distributed environment
    rank, world_size, is_distributed = setup_distributed()

    # Setup device based on mode
    if args.accelerator == "gpu" and torch.cuda.is_available():
        if is_distributed:
            local_rank = get_local_rank()
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.info("Running on CPU, this will be slow. Consider using a GPU.")

    # Only main process logs basic info
    if is_main_process():
        logger.info(f"Running in {'distributed' if is_distributed else 'single-GPU'} mode")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info(f"Using device: {device}")

    # Set cache path
    args.cache = Path(args.cache).expanduser()
    # expand the cache path like ~/.boltz to the absolute path like /home/yangnianzu/.boltz
    args.cache.mkdir(parents=True, exist_ok=True)
    # create the cache directory if it does not exist and all parent directories will be created too

    # Create output directories
    args.data = Path(args.data).expanduser()

    # Create output directory (all processes)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle MSA directories (matching upstream Boltz1 predict.py logic)
    if args.msa_dir is not None:
        args.msa_dir = Path(args.msa_dir).expanduser()
    else:
        args.msa_dir = output_dir / "msa"

    if args.processed_msa_dir is not None:
        args.processed_msa_dir = Path(args.processed_msa_dir).expanduser()
    else:
        args.processed_msa_dir = output_dir / "processed" / "msa"

    # Validate inputs
    data, msg = check_inputs(args.data, output_dir, args.override)
    if msg is not None:
        logger.info(msg)
    # return the data files in the input data directory (may skip some files if we have already predicted them, depends on the override flag)
    if not data:
        # if the input data is empty, we will exit
        logger.error("No predictions to run, exiting.")
        return

    # Sort data by stem name to ensure consistent ordering across runs
    # This is critical for --max_samples to select the same samples as refactored code
    data = sorted(data, key=lambda x: x.stem)

    # Apply max_samples limit if specified
    if args.max_samples > 0:
        data = data[:args.max_samples]
        selected_names = [f.stem for f in data]
        logger.info(f"Selected {len(data)} samples (max_samples={args.max_samples}): {selected_names}")

    ccd_path = args.cache / "ccd.pkl"
    if is_main_process():
        process_inputs(
            data=data,
            out_dir=output_dir,
            msa_dir=args.msa_dir,
            processed_msa_dir=args.processed_msa_dir,
            preprocessed_data_path=args.preprocessed_data_path,
            ccd_path=ccd_path,
        )
    if is_distributed:
        torch.distributed.barrier()

    _ensure_runtime_imports()
    processed_dir = output_dir / "processed"
    # Update processed with final directories
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=args.processed_msa_dir,
    )

    # Create dataset using PredictionDataset (matching upstream Boltz1's BoltzInferenceDataModule)
    # This is the same as BoltzInferenceDataModule.predict_dataloader() creates internally
    dataset = PredictionDataset(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        inpaint=args.structure_inpainting,
        ground_truth_dir=args.ground_truth_structure_dir,
        gt_oracle_cdr=args.gt_oracle_cdr,
        use_epitope=args.use_epitope
    )

    # Create sampler for distributed mode
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # Keep order for reproducibility
        )

    # Create DataLoader with parallel workers
    # Note: num_workers > 0 enables parallel data preprocessing
    # Using collate function from PredictionDataset for proper batching
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=collate,  # Use collate function for PredictionDataset
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # Create model
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": args.write_full_pae,
        "write_full_pde": args.write_full_pde,
    }

    # diffusion_params = AF3DiffusionParams()
    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = args.step_scale
    diffusion_params.temperature = args.temperature
    diffusion_params.noise_type = args.noise_type

    model = Boltz1.from_pretrained(
        ckpt_path=args.checkpoint,
        predict_args=predict_args,
        structure_prediction_training=False,
        sequence_prediction_training=not args.only_structure_prediction,
        confidence_prediction=True,
        confidence_imitate_trunk=True,
        structure_inpainting=args.structure_inpainting,
        alpha_pae=1.0,
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
    )

    # Move to device
    model = model.to(device)
    model.eval()

    # Process samples with progress bar (all ranks show progress)
    success_count = 0
    error_count = 0

    chain_infos_path = processed_dir / "chain_infos.json"
    pred_writer = BoltzWriter(
        data_dir=str(processed.targets_dir),
        output_dir=str(output_dir / "predictions"),
        output_format=args.output_format,
        seq_info_path=str(chain_infos_path) if chain_infos_path.exists() else None,
        save_cdr_json=not args.no_cdr_json,
    )

    # Each rank shows its own progress bar with rank info
    pbar = tqdm(
        dataloader,
        desc=f"Rank {rank}/{world_size}" if is_distributed else "Inference",
        unit="sample",
        position=rank if is_distributed else 0,  # Stack progress bars vertically
        leave=True,
    )

    for batch in pbar:
        batch = send_to_device(batch, device=device)
        with torch.no_grad():
            prediction = model.predict(batch)

        if prediction["exception"]:
            logger.error(f"[Rank {rank}] No record in batch")
            error_count += 1
            continue

        pred_writer.write_on_batch_end(prediction=prediction, batch=batch)
        success_count += 1

    # Log local results (no synchronization to avoid NCCL timeout issues)
    logger.info(f"[Rank {rank}] Inference complete! Success={success_count}, Errors={error_count}")


if __name__ == "__main__":
    main()
