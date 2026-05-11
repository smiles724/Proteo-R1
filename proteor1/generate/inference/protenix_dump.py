#!/usr/bin/env python3
"""
Protenix Full Embedding Dump Script

Dumps Protenix features from protein structure files (CIF or PDB format) with GT coordinates:
- s: Pairformer output [N_token, 384]
- esm_embedding: ESM token embedding [N_token, 2560]
- a_token: Diffusion token embedding [N_token, 768]
- token_is_resolved: Token-level coordinate mask [N_token], indicating resolved coordinates

Supported formats and auto-detected database:
- .pdb, .pdb.gz: Convert to CIF first, then process -> saved to "pdb_full" database
- .cif, .cif.gz: Direct CIF processing -> saved to "cif_full" database

Key features:
1. Deterministic output: ref_pos_augment=False + sigma=0 + centre_only=True
2. Direct structure file loading for GT coordinates
3. Ensures atom ordering consistency
4. Multi-GPU parallel processing
5. Automatic database detection based on file extension

IMPORTANT: --no-bioassembly --protein-only are REQUIRED for training data alignment.
  --no-bioassembly: Use asymmetric unit instead of bioassembly, aligned with data-side CIF parser.
  --protein-only:   Keep protein tokens only, remove ligands/DNA/RNA/ions.
  These two flags produce the "_asym_prot" database suffix (e.g. cif_full_asym_prot/).

Usage:

    1) Standard ESM+A token feature dump (for training):

       python -m proteor1.generate.inference.protenix_dump \
           --input_dir <source-cif-dir> \
           --rank $RANK --world_size 8 \
           --no-bioassembly --protein-only \
           --num_workers 1

       Output: data/feat_dump/{model_name}/cif_full_asym_prot/{protein_id}.safetensors
       Contains: s [N,384] + esm_embedding [N,2560] + a_token [N,768] + position info

    2) No-noise + intermediate layer features (ablation):

       python -m proteor1.generate.inference.protenix_dump \
           --input_dir <source-cif-dir> \
           --rank $RANK --world_size 8 \
           --no-bioassembly --protein-only \
           --num_workers 1 \
           --sigma 0 --dump-intermediate-layers 0 3

       Output: data/feat_dump_layers0_3/{model_name}/cif_full_asym_prot/
       Extra keys: a_token_layer_0 [N,768], a_token_layer_3 [N,768]
       NOTE: Intermediate layer features are NOT LayerNorm'd. Apply LN manually for probing.

    3) Minimum-noise + intermediate layer features (ablation comparison):

       python -m proteor1.generate.inference.protenix_dump \
           --input_dir <source-cif-dir> \
           --rank $RANK --world_size 8 \
           --no-bioassembly --protein-only \
           --num_workers 1 \
           --sigma 4e-4 --dump-intermediate-layers 0 3

       Output: data/feat_dump_sigma0.0004_layers0_3/{model_name}/cif_full_asym_prot/
       sigma=4e-4 is the minimum noise level used during Protenix inference (sigma_min).

Output directory naming convention:
    {base_output}[_sigma{value}][_layers{indices}]/{model_name}/{database}{_asym}{_prot}/
    Examples:
      sigma=0,   no layers  -> data/feat_dump/
      sigma=0,   layers 0,3 -> data/feat_dump_layers0_3/
      sigma=4e-4, layers 0,3 -> data/feat_dump_sigma0.0004_layers0_3/

Safetensors file contents:
    Key                  Shape        Always saved   Description
    s                    [N, 384]     Yes            Pairformer embedding
    esm_embedding        [N, 2560]    Yes            ESM embedding
    a_token              [N, 768]     Yes            Diffusion final output (LayerNorm'd)
    residue_index        [N]          Yes            Within-chain residue index
    asym_id              [N]          Yes            Chain identifier (0-indexed)
    entity_id            [N]          Yes            Molecular entity ID
    sym_id               [N]          Yes            Symmetry ID
    token_is_resolved    [N]          Yes            Coordinate resolution mask
    a_token_layer_{i}    [N, 768]     Only w/ flag   Intermediate block output (NOT LayerNorm'd)

Notes:
    - mini_v0.5.0 has 8 diffusion blocks (0-7). Layer 0 = first, layer 3 = middle.
    - sigma=0 is fully deterministic. sigma=4e-4 introduces minimal noise for ablation.
    - Multi-GPU: files are interleaved across ranks (rank 0 gets [0::world_size], etc.).
    - Existing files are auto-skipped unless --force is set.
"""
from typing import Optional, Any

import argparse
import gzip
import json
import os
import sys
import tempfile
import traceback
from datetime import datetime
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from loguru import logger
from safetensors.torch import save_file, safe_open
from tqdm import tqdm

from proteor1.understand.data_collator import move_protenix_features_to_device
from proteor1.generate.inference._dump_helpers import (
    get_model_name,
    get_embedding_path,
    EmbeddingDumper,
)
from proteor1.generate.inference._dump_helpers import (
    get_precomputed_path,
    get_precomputed_suffix,
    load_processor_output,
)

# Optional psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    psutil = None
    HAS_PSUTIL = False

import gc
import signal


def get_database_suffix(use_bioassembly: bool, protein_only: bool) -> str:
    """
    Generate database suffix based on processing options.

    The suffix encodes the combination of --no-bioassembly and --protein-only flags:
    - "" (empty): use_bioassembly=True, protein_only=False (default)
    - "_asym": use_bioassembly=False, protein_only=False
    - "_prot": use_bioassembly=True, protein_only=True
    - "_asym_prot": use_bioassembly=False, protein_only=True

    Args:
        use_bioassembly: Whether bioassembly is used (False means --no-bioassembly was set)
        protein_only: Whether to filter to protein-only tokens (--protein-only flag)

    Returns:
        Suffix string to append to database name (e.g., "cif_full" -> "cif_full_asym_prot")
    """
    parts = []
    if not use_bioassembly:
        parts.append("asym")
    if protein_only:
        parts.append("prot")

    if parts:
        return "_" + "_".join(parts)
    return ""


def log_memory_usage(prefix: str = "") -> None:
    """
    Log current memory usage (RAM and GPU).

    Args:
        prefix: Optional prefix for the log message
    """
    msg_parts = []

    # RAM usage
    if HAS_PSUTIL:
        process = psutil.Process()
        mem_info = process.memory_info()
        ram_gb = mem_info.rss / (1024 ** 3)
        msg_parts.append(f"RAM: {ram_gb:.2f} GB")

    # GPU memory usage
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        msg_parts.append(f"GPU allocated: {gpu_allocated:.2f} GB, reserved: {gpu_reserved:.2f} GB")

    if msg_parts:
        prefix_str = f"{prefix} | " if prefix else ""
        logger.info(f"{prefix_str}Memory usage: {', '.join(msg_parts)}")


# ============================================================================
# DataLoader Parallelization Components
# ============================================================================

def collate_fn_first(batch: list) -> Any:
    """
    Collate function that returns the first element of the batch.
    Used with batch_size=1 to avoid default collation behavior.
    Reference: protenix/utils/torch_utils.py:247-248
    """
    return batch[0]


def worker_init_fn(worker_id: int) -> None:
    """
    Worker initialization function for DataLoader workers.
    Sets up logging and any worker-specific initialization.
    """
    # Suppress verbose logging in workers to avoid log spam
    logger.remove()
    logger.add(
        sys.stderr,
        format=f"<dim>[W{worker_id}]</dim> <level>{{level: <8}}</level> | {{message}}",
        level="WARNING",
    )


def is_dataloader_worker_error(e: Exception) -> bool:
    """
    Check if an exception is a DataLoader worker-related error.

    These errors typically indicate worker process crashes, timeouts, or
    resource exhaustion that require DataLoader rebuild to recover.

    Args:
        e: The exception to check

    Returns:
        True if this is a DataLoader worker error that can be recovered
        by rebuilding the DataLoader
    """
    if not isinstance(e, RuntimeError):
        return False

    error_msg = str(e).lower()

    # Patterns indicating DataLoader worker issues
    patterns = [
        "dataloader worker",           # Worker crash or exit
        "dataloader timed out",        # Worker timeout
        "pin memory thread",           # Pin memory thread crash
        "too many open files",         # File descriptor exhaustion
        "cannot allocate memory",      # Memory allocation failure
        "broken pipe",                 # Communication pipe broken
        "connection reset",            # IPC connection reset
        "killed",                      # Worker killed by OOM killer
        "_multiprocessingdataloadriter", # Internal DataLoader error
        "worker exited",               # Worker process exited
        "unexpectedly",                # Unexpected termination
    ]

    return any(pattern in error_msg for pattern in patterns)


def is_cuda_device_error(e: Exception) -> bool:
    """
    Check if an exception is a CUDA device-side error (e.g., device-side assert).

    These errors corrupt the CUDA context and require special handling.
    After a device-side assert, most CUDA operations will fail until
    the context is reset (typically requires process restart for full recovery).

    Args:
        e: The exception to check

    Returns:
        True if this is a CUDA device error
    """
    if not isinstance(e, RuntimeError):
        return False

    error_msg = str(e).lower()

    # Patterns indicating CUDA device errors
    patterns = [
        "device-side assert",          # Index out of bounds or assertion failure
        "cuda error",                  # Generic CUDA error
        "cublas",                      # CUBLAS errors
        "cudnn",                       # CUDNN errors
        "illegal memory access",       # Memory access violation
        "an illegal instruction",      # Illegal instruction on device
        "misaligned address",          # Memory alignment issue
        "cuda_error_",                 # CUDA error codes
    ]

    return any(pattern in error_msg for pattern in patterns)


def safe_cuda_empty_cache() -> bool:
    """
    Safely call torch.cuda.empty_cache(), handling CUDA errors gracefully.

    After a device-side assert or other CUDA error, the CUDA context may be
    corrupted. This function attempts to clear the cache safely and returns
    whether it succeeded.

    Returns:
        True if cache was cleared successfully, False if CUDA is in a bad state
    """
    if not torch.cuda.is_available():
        return True

    try:
        # First try to synchronize to surface any pending errors
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        if is_cuda_device_error(e):
            logger.warning(f"CUDA in bad state, cannot clear cache: {e}")
            # Try to at least reset memory stats
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            return False
        raise  # Re-raise non-CUDA errors


def check_esm_index_bounds(features_dict: dict, protein_id: str = "") -> tuple[bool, str]:
    """
    Check if residue indices are within ESM embedding bounds.

    ESM models have maximum sequence length limits. When a protein sequence exceeds
    this limit, the ESM tokenizer truncates the sequence, but the residue_index
    tensor may still reference positions beyond the ESM embedding length.

    This causes CUDA index out of bounds errors during GPU inference when
    ESM embeddings are gathered using residue indices.

    Based on protenix/data/esm_featurizer.py:116:
        res_index = res_id - 1  (convert 1-based to 0-based)
        x_esm[res_index]  # requires res_index < esm_len

    So valid residue_index range is [1, esm_len].

    Args:
        features_dict: Input feature dictionary containing 'residue_index' and 'esm_input_ids'
        protein_id: Optional protein ID for logging

    Returns:
        tuple: (is_valid, error_message)
            - is_valid: True if indices are within bounds
            - error_message: Description of the issue if invalid, empty string if valid
    """
    # Check if ESM features exist
    if 'esm_input_ids' not in features_dict or 'residue_index' not in features_dict:
        return True, ""  # No ESM features, skip check

    esm_input_ids = features_dict['esm_input_ids']
    residue_index = features_dict['residue_index']

    # Get ESM sequence length (shape is [1, seq_len] or [seq_len])
    if hasattr(esm_input_ids, 'shape'):
        esm_len = esm_input_ids.shape[-1]
    else:
        return True, ""  # Unknown format, skip check

    # Get max residue index
    if hasattr(residue_index, 'numpy'):
        max_res_idx = int(residue_index.numpy().max())
    elif hasattr(residue_index, 'max'):
        max_res_idx = int(residue_index.max())
    else:
        return True, ""  # Unknown format, skip check

    # residue_index is 1-based, ESM indexing uses (res_id - 1)
    # So max valid residue_index is esm_len (which becomes esm_len-1 after -1)
    if max_res_idx > esm_len:
        error_msg = (
            f"ESM index out of bounds: max_residue_index={max_res_idx} > esm_len={esm_len}. "
            f"Protein sequence ({max_res_idx} residues) exceeds ESM model's maximum length limit ({esm_len})."
        )
        return False, error_msg

    return True, ""


class CIFDataset(Dataset):
    """
    Dataset for parallel CIF preprocessing using DataLoader.

    This dataset handles:
    1. CIF file loading and preprocessing
    2. PDB to CIF conversion (for .pdb/.pdb.gz files)
    3. ProtenixProcessor initialization (lazy, per-worker)

    The preprocessing is CPU-bound and can be parallelized across workers.
    GPU inference is done separately in the main process.

    Blacklist mechanism:
    - Class-level failure counts and blacklist (shared across DataLoader rebuilds)
    - Files that fail MAX_FAILURES_PER_FILE times are added to blacklist
    - Blacklisted files are skipped in __getitem__ to avoid repeated crashes
    """

    # Class-level variables (shared across instances for crash recovery)
    _failure_counts: dict[str, int] = {}
    _blacklist: set[str] = set()
    MAX_FAILURES_PER_FILE: int = 3

    @classmethod
    def record_failure(cls, file_path: str, protein_id: str = "") -> bool:
        """
        Record a file failure. Returns True if file was added to blacklist.

        Args:
            file_path: Path to the failed file
            protein_id: Optional protein ID for logging

        Returns:
            True if the file was added to blacklist (reached MAX_FAILURES_PER_FILE)
        """
        cls._failure_counts[file_path] = cls._failure_counts.get(file_path, 0) + 1
        count = cls._failure_counts[file_path]

        if count >= cls.MAX_FAILURES_PER_FILE and file_path not in cls._blacklist:
            cls._blacklist.add(file_path)
            logger.warning(
                f"File blacklisted after {count} failures: {file_path} "
                f"(protein_id={protein_id})"
            )
            return True
        return False

    @classmethod
    def is_blacklisted(cls, file_path: str) -> bool:
        """Check if a file is in the blacklist."""
        return file_path in cls._blacklist

    @classmethod
    def get_blacklist_stats(cls) -> dict:
        """
        Get blacklist statistics.

        Returns:
            dict with:
            - blacklisted_files: list of blacklisted file paths
            - blacklist_count: number of blacklisted files
            - failure_counts: dict of file -> failure count for files with failures
        """
        return {
            "blacklisted_files": list(cls._blacklist),
            "blacklist_count": len(cls._blacklist),
            "failure_counts": {k: v for k, v in cls._failure_counts.items() if v > 0},
        }

    @classmethod
    def reset_blacklist(cls) -> None:
        """Reset blacklist and failure counts. Mainly for testing."""
        cls._failure_counts.clear()
        cls._blacklist.clear()

    def __init__(
        self,
        protein_files: list[str],
        protenix_path: str,
        ref_pos_augment: bool = False,
        output_dir: Optional[str] = None,
        model_name: Optional[str] = None,
        force: bool = False,
        use_bioassembly: bool = True,
        protein_only: bool = False,
        precomputed_dir: Optional[str] = None,
        max_esm_seq_length: Optional[int] = None,
        max_chain_count: Optional[int] = None,
        min_chain_count: Optional[int] = None,
    ):
        """
        Args:
            protein_files: List of protein structure file paths (.cif, .cif.gz, .pdb, .pdb.gz)
            protenix_path: Path to pretrained Protenix model (for ProtenixProcessor)
            ref_pos_augment: Apply random augmentation to ref_pos
            output_dir: Output directory for embeddings (unused, kept for compatibility)
            model_name: Model name (unused, kept for compatibility)
            force: Force overwrite existing files (unused, kept for compatibility)
            use_bioassembly: Use bioassembly structure (default True); if False, use asymmetric unit
            protein_only: If True, only include protein chains in mapping; if False (default), include all chains
            precomputed_dir: Directory containing precomputed processor outputs. If specified,
                will attempt to load precomputed features instead of processing CIF files.
                Falls back to process_from_cif if precomputed file not found.
            max_esm_seq_length: Maximum allowed ESM sequence length. If specified, samples
                with esm_input_ids length exceeding this threshold will be skipped.
                If None (default), no length filtering is applied.
            max_chain_count: Maximum allowed number of chains (based on asym_id_to_auth_asym_id length).
                If specified, samples with chain count exceeding this threshold will be skipped.
                If None (default), no upper limit filtering is applied.
            min_chain_count: Minimum required number of chains (based on asym_id_to_auth_asym_id length).
                If specified, samples with chain count below this threshold will be skipped.
                If None (default), no lower limit filtering is applied.

        Note:
            The skip check (already_exists_with_a_token) is now performed in
            dump_protein_files_parallel() before DataLoader creation, so output_dir,
            model_name, and force parameters are no longer used for skip checking
            in __getitem__. They are kept for backward compatibility.
        """
        self.protein_files = protein_files
        self.protenix_path = protenix_path
        self.ref_pos_augment = ref_pos_augment
        self.output_dir = output_dir
        self.model_name = model_name
        self.force = force
        self.use_bioassembly = use_bioassembly
        self.protein_only = protein_only
        self.precomputed_dir = precomputed_dir
        self.max_esm_seq_length = max_esm_seq_length
        self.max_chain_count = max_chain_count
        self.min_chain_count = min_chain_count

        # Lazy-initialized processor (per-worker to avoid pickle issues)
        self._processor = None

    @property
    def processor(self):
        """Lazy load ProtenixProcessor in worker process"""
        if self._processor is None:
            from proteor1.understand.protenix_encoder import ProtenixProcessor
            self._processor = ProtenixProcessor.from_pretrained(
                self.protenix_path,
                init_esm_tokenizer=True,
            )
        return self._processor

    def __len__(self) -> int:
        return len(self.protein_files)

    def _extract_protein_id(self, file_path: str) -> str:
        """Extract protein_id from filename"""
        basename = os.path.basename(file_path)
        # Handle patterns like "1abc-assembly1.cif.gz", "1abc.pdb", etc.
        # return basename.split(".")[0].upper()
        return basename.split(".")[0].upper()

    def _get_database(self, file_path: str) -> str:
        """Determine database based on file extension and processing options"""
        lower_path = file_path.lower()
        if lower_path.endswith(".pdb") or lower_path.endswith(".pdb.gz"):
            base = "pdb_full"
        else:
            base = "cif_full"

        # Add suffix based on processing options
        suffix = get_database_suffix(self.use_bioassembly, self.protein_only)
        return base + suffix

    def _convert_pdb_to_cif(self, pdb_path: str) -> str:
        """
        Convert PDB file to CIF format.
        Returns path to temporary CIF file.
        Caller is responsible for cleanup.
        """
        from protenix.data.utils import pdb_to_cif

        lower_path = pdb_path.lower()

        # Handle .pdb.gz: decompress to temp .pdb file first
        if lower_path.endswith(".pdb.gz"):
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_pdb:
                tmp_pdb_path = tmp_pdb.name
                with gzip.open(pdb_path, "rb") as gz_in:
                    tmp_pdb.write(gz_in.read())
            pdb_path_to_convert = tmp_pdb_path
        else:
            pdb_path_to_convert = pdb_path
            tmp_pdb_path = None

        # Convert PDB to CIF
        with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp_cif:
            tmp_cif_path = tmp_cif.name

        pdb_to_cif(pdb_path_to_convert, tmp_cif_path)

        # Clean up temp PDB file if created
        if tmp_pdb_path is not None:
            os.unlink(tmp_pdb_path)

        return tmp_cif_path

    def __getitem__(self, idx: int) -> dict:
        """
        Preprocess a single protein file.

        Returns:
            dict with either:
            - Preprocessed features (on success):
                {
                    "status": "success",
                    "protein_id": str,
                    "database": str,
                    "file_path": str,
                    "features_dict": dict,  # CPU tensors
                    "atom_array": AtomArray,
                    "token_array": TokenArray,
                }
            - Skip information (for blacklisted or already processed):
                {
                    "status": "skip",
                    "protein_id": str,
                    "database": str,
                    "file_path": str,
                    "reason": str,
                }
            - Error information (on failure):
                {
                    "status": "error",
                    "protein_id": str,
                    "database": str,
                    "file_path": str,
                    "error": str,
                    "error_type": str,
                    "traceback": str,
                }
        """
        file_path = self.protein_files[idx]
        protein_id = self._extract_protein_id(file_path)
        database = self._get_database(file_path)
        lower_path = file_path.lower()

        # Blacklist check: skip files that have failed too many times
        # Note: already_exists check is now done in dump_protein_files_parallel
        # before DataLoader creation, so we don't need to check it here anymore.
        if self.is_blacklisted(file_path):
            return {
                "status": "skip",
                "protein_id": protein_id,
                "database": database,
                "file_path": file_path,
                "reason": "blacklisted",
            }

        tmp_cif_path = None

        try:
            # Try to load precomputed features if precomputed_dir is specified
            if self.precomputed_dir is not None:
                precomputed_path = os.path.join(self.precomputed_dir, f"{protein_id}.pkl.gz")

                if os.path.exists(precomputed_path):
                    # Load precomputed processor output
                    try:
                        processor_output = load_processor_output(precomputed_path)

                        # Build asym_id -> auth_asym_id mapping for chain ID recovery
                        atom_array = processor_output.atom_array
                        token_array = processor_output.token_array
                        asym_id_to_auth_asym_id = build_asym_id_to_auth_asym_id_mapping(
                            atom_array, token_array, protein_only=self.protein_only
                        )

                        # Check ESM index bounds before returning
                        esm_valid, esm_error = check_esm_index_bounds(
                            processor_output.input_feature_dict, protein_id
                        )
                        if not esm_valid:
                            return {
                                "status": "error",
                                "protein_id": protein_id,
                                "database": database,
                                "file_path": file_path,
                                "error": esm_error,
                                "error_type": "ESMIndexOutOfBounds",
                                "traceback": "",
                            }

                        # Check ESM sequence length threshold
                        if self.max_esm_seq_length is not None:
                            esm_input_ids = processor_output.input_feature_dict.get('esm_input_ids')
                            if esm_input_ids is not None and hasattr(esm_input_ids, 'shape'):
                                esm_seq_len = esm_input_ids.shape[-1]
                                if esm_seq_len > self.max_esm_seq_length:
                                    return {
                                        "status": "skip",
                                        "protein_id": protein_id,
                                        "database": database,
                                        "file_path": file_path,
                                        "reason": f"esm_seq_length_exceeded (len={esm_seq_len} > max={self.max_esm_seq_length})",
                                    }

                        # Check chain count threshold
                        chain_count = len(asym_id_to_auth_asym_id)
                        if self.max_chain_count is not None and chain_count > self.max_chain_count:
                            return {
                                "status": "skip",
                                "protein_id": protein_id,
                                "database": database,
                                "file_path": file_path,
                                "reason": f"chain_count_exceeded (count={chain_count} > max={self.max_chain_count})",
                            }
                        if self.min_chain_count is not None and chain_count < self.min_chain_count:
                            return {
                                "status": "skip",
                                "protein_id": protein_id,
                                "database": database,
                                "file_path": file_path,
                                "reason": f"chain_count_below_min (count={chain_count} < min={self.min_chain_count})",
                            }

                        return {
                            "status": "success",
                            "protein_id": protein_id,
                            "database": database,
                            "file_path": file_path,
                            "features_dict": processor_output.input_feature_dict,
                            "atom_array": atom_array,
                            "token_array": token_array,
                            "asym_id_to_auth_asym_id": asym_id_to_auth_asym_id,
                            "source": "precomputed",
                        }
                    except Exception as e:
                        # Precomputed file is corrupted, fall back to process_from_cif
                        print(
                            f"[W] Failed to load precomputed for {protein_id}: {e}, "
                            f"falling back to process_from_cif",
                            file=sys.stderr
                        )
                else:
                    # Precomputed file not found, log and fall back
                    print(
                        f"[W] Precomputed not found for {protein_id}, falling back to process_from_cif",
                        file=sys.stderr
                    )

            # Original logic: process from CIF file
            # Determine CIF path (direct or converted from PDB)
            if lower_path.endswith(".cif") or lower_path.endswith(".cif.gz"):
                cif_path = file_path
            elif lower_path.endswith(".pdb") or lower_path.endswith(".pdb.gz"):
                tmp_cif_path = self._convert_pdb_to_cif(file_path)
                cif_path = tmp_cif_path
            else:
                return {
                    "status": "error",
                    "protein_id": protein_id,
                    "database": database,
                    "file_path": file_path,
                    "error": f"Unsupported file format: {file_path}",
                    "error_type": "UnsupportedFormat",
                    "traceback": "",
                }

            # Run preprocessing
            processor_output = self.processor.process_from_cif(
                cif_path=cif_path,
                assembly_id="1",
                ref_pos_augment=self.ref_pos_augment,
                return_atom_array=True,
                return_token_array=True,
                use_bioassembly=self.use_bioassembly,
                protein_only=self.protein_only,
            )

            # Build asym_id -> auth_asym_id mapping for chain ID recovery
            atom_array = processor_output.atom_array
            token_array = processor_output.token_array
            asym_id_to_auth_asym_id = build_asym_id_to_auth_asym_id_mapping(atom_array, token_array, protein_only=self.protein_only)

            # Check ESM index bounds before returning
            esm_valid, esm_error = check_esm_index_bounds(
                processor_output.input_feature_dict, protein_id
            )
            if not esm_valid:
                return {
                    "status": "error",
                    "protein_id": protein_id,
                    "database": database,
                    "file_path": file_path,
                    "error": esm_error,
                    "error_type": "ESMIndexOutOfBounds",
                    "traceback": "",
                }

            # Check ESM sequence length threshold
            if self.max_esm_seq_length is not None:
                esm_input_ids = processor_output.input_feature_dict.get('esm_input_ids')
                if esm_input_ids is not None and hasattr(esm_input_ids, 'shape'):
                    esm_seq_len = esm_input_ids.shape[-1]
                    if esm_seq_len > self.max_esm_seq_length:
                        return {
                            "status": "skip",
                            "protein_id": protein_id,
                            "database": database,
                            "file_path": file_path,
                            "reason": f"esm_seq_length_exceeded (len={esm_seq_len} > max={self.max_esm_seq_length})",
                        }

            # Check chain count threshold
            chain_count = len(asym_id_to_auth_asym_id)
            if self.max_chain_count is not None and chain_count > self.max_chain_count:
                return {
                    "status": "skip",
                    "protein_id": protein_id,
                    "database": database,
                    "file_path": file_path,
                    "reason": f"chain_count_exceeded (count={chain_count} > max={self.max_chain_count})",
                }
            if self.min_chain_count is not None and chain_count < self.min_chain_count:
                return {
                    "status": "skip",
                    "protein_id": protein_id,
                    "database": database,
                    "file_path": file_path,
                    "reason": f"chain_count_below_min (count={chain_count} < min={self.min_chain_count})",
                }

            result = {
                "status": "success",
                "protein_id": protein_id,
                "database": database,
                "file_path": file_path,
                "features_dict": processor_output.input_feature_dict,
                "atom_array": atom_array,
                "token_array": token_array,
                "asym_id_to_auth_asym_id": asym_id_to_auth_asym_id,
                "source": "process_from_cif",
            }

            return result

        except ValueError as e:
            # Check for specific "No protein atoms found" error
            error_msg = str(e)
            if "No protein atoms found" in error_msg:
                error_type = "NoProteinAtoms"
            else:
                error_type = "ValueError"
            return {
                "status": "error",
                "protein_id": protein_id,
                "database": database,
                "file_path": file_path,
                "error": error_msg,
                "error_type": error_type,
                "traceback": traceback.format_exc(),
            }

        except Exception as e:
            return {
                "status": "error",
                "protein_id": protein_id,
                "database": database,
                "file_path": file_path,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }

        finally:
            # Clean up temp CIF file if created
            if tmp_cif_path is not None and os.path.exists(tmp_cif_path):
                try:
                    os.unlink(tmp_cif_path)
                except Exception:
                    pass


# ============================================================================
# Diffusion Intermediate Layer Utilities
# ============================================================================

def build_output_dir(base_output: str, sigma: float, intermediate_layers=None) -> str:
    """
    Build output directory by appending suffix to base_output.
    When sigma=0.0 and no intermediate layers, returns base_output unchanged
    for full backward compatibility.
    """
    suffix = ""
    if sigma > 0:
        suffix += f"_sigma{sigma:g}"
    if intermediate_layers:
        layers_str = "_".join(str(l) for l in sorted(intermediate_layers))
        suffix += f"_layers{layers_str}"
    return base_output + suffix


def register_diffusion_intermediate_hooks(diffusion_module, layer_indices):
    """
    Register forward hooks on specified DiffusionTransformerBlock layers.
    DiffusionTransformerBlock.forward() returns (out_a, s, z) tuple; we capture output[0].

    Args:
        diffusion_module: DiffusionModule instance (full_encoder.diffusion_module)
        layer_indices: list of int, block indices to capture (0-23)

    Returns:
        captured: dict[int -> Tensor], filled during forward pass
        hooks: list of hook handles (call remove_diffusion_hooks after forward)
    """
    captured = {}
    hooks = []
    blocks = diffusion_module.diffusion_transformer.blocks

    for idx in layer_indices:
        if idx < 0 or idx >= len(blocks):
            raise ValueError(
                f"Intermediate layer index {idx} out of range [0, {len(blocks) - 1}]"
            )

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is (out_a, s, z) tuple from DiffusionTransformerBlock.forward()
                captured[layer_idx] = output[0].detach()
            return hook_fn

        hook = blocks[idx].register_forward_hook(make_hook(idx))
        hooks.append(hook)

    return captured, hooks


def remove_diffusion_hooks(hooks):
    """Remove all registered forward hooks."""
    for hook in hooks:
        hook.remove()


# ============================================================================
# Save / Load Functions (Extended for a_token)
# ============================================================================

def save_full_embedding(
    path: str,
    s: torch.Tensor,
    esm_embedding: torch.Tensor,
    a_token: torch.Tensor,
    n_token: int,
    metadata: dict,
    dtype: torch.dtype = torch.bfloat16,
    residue_index: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
    entity_id: Optional[torch.Tensor] = None,
    sym_id: Optional[torch.Tensor] = None,
    coordinate_mask: Optional[torch.Tensor] = None,
    asym_id_to_auth_asym_id: Optional[dict[int, str]] = None,
    auth_residue_index: Optional[torch.Tensor] = None,
    token_is_resolved: Optional[torch.Tensor] = None,
    intermediate_features: Optional[dict] = None,
):
    """
    Save full embedding including a_token

    File contents:
    - tensors: s, esm_embedding, a_token, n_token, residue_index, asym_id, entity_id, sym_id, coordinate_mask, auth_residue_index, token_is_resolved
    - metadata: protein_id, database, n_token, sigma, ref_pos_augment, has_a_token, asym_id_to_auth_asym_id, ...

    Position info tensors (all [N_token]):
    - residue_index: Residue index within each chain (label_seq_id, 1-indexed, continuous)
    - auth_residue_index: Original PDB residue numbering (auth_seq_id, may have gaps/insertion codes).
      NOTE: This value is UNRELIABLE for tokens where token_is_resolved=False (missing residues).
      For missing residues, auth_seq_id is empty in the CIF and we fallback to res_id, which may
      conflict with other residues' auth_seq_id. Always filter by token_is_resolved=True first.
    - asym_id: Asymmetric unit ID (chain identifier, integer starting from 0)
    - entity_id: Entity ID (unique molecular entity)
    - sym_id: Symmetry ID (for biological assemblies with multiple copies)

    Chain ID mapping (in metadata):
    - asym_id_to_auth_asym_id: Maps asym_id (int) -> auth_asym_id (str, e.g., "H", "L", "A")
      This allows recovering embeddings by original chain ID from the CIF file.

    Token-level coordinate mask (all [N_token]):
    - token_is_resolved: Boolean indicating if each token's centre atom has resolved coordinates.
      Based on atom_array.is_resolved for centre atoms. Tokens with is_resolved=False have
      coordinates set to 0.0 (missing atoms added by get_bioassembly()).
      IMPORTANT: auth_residue_index is only reliable when token_is_resolved=True.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tensors = {
        "s": s.detach().cpu().to(dtype),
        "esm_embedding": esm_embedding.detach().cpu().to(dtype),
        "a_token": a_token.detach().cpu().to(dtype),
        "n_token": torch.tensor([n_token], dtype=torch.int64),
    }

    # Save position info if provided (all are [N_token] tensors)
    if residue_index is not None:
        tensors["residue_index"] = residue_index.detach().cpu().long()
    if asym_id is not None:
        tensors["asym_id"] = asym_id.detach().cpu().long()
    if entity_id is not None:
        tensors["entity_id"] = entity_id.detach().cpu().long()
    if sym_id is not None:
        tensors["sym_id"] = sym_id.detach().cpu().long()
    # Save coordinate_mask if provided (from atom_array.is_resolved, indicates resolved atoms)
    if coordinate_mask is not None:
        tensors["coordinate_mask"] = coordinate_mask.detach().cpu().long()
    # Save auth_residue_index if provided (original PDB residue numbering, auth_seq_id)
    if auth_residue_index is not None:
        tensors["auth_residue_index"] = auth_residue_index.detach().cpu().long()
    # Save token_is_resolved if provided (token-level coordinate mask based on centre atom is_resolved)
    if token_is_resolved is not None:
        tensors["token_is_resolved"] = token_is_resolved.detach().cpu().long()

    # Save intermediate diffusion layer features if provided
    if intermediate_features:
        for layer_idx, feat in intermediate_features.items():
            tensors[f"a_token_layer_{layer_idx}"] = feat.detach().cpu().to(dtype)

    metadata_str = {k: str(v) for k, v in metadata.items()}
    metadata_str["dump_timestamp"] = datetime.now().isoformat()
    metadata_str["dtype"] = str(dtype).split(".")[-1]
    metadata_str["has_position_info"] = str(
        residue_index is not None and asym_id is not None and entity_id is not None and sym_id is not None
    )
    metadata_str["has_a_token"] = "True"

    # Save asym_id -> auth_asym_id mapping as JSON string
    if asym_id_to_auth_asym_id is not None:
        metadata_str["asym_id_to_auth_asym_id"] = json.dumps(asym_id_to_auth_asym_id)

    metadata_str["intermediate_layers"] = json.dumps(
        sorted(intermediate_features.keys()) if intermediate_features else []
    )

    save_file(tensors, path, metadata=metadata_str)


def load_full_embedding(path: str) -> dict:
    """
    Load full embedding including a_token and optional intermediate layer features.

    Returns:
        {
            "s": Tensor [N_token, 384],
            "esm_embedding": Tensor [N_token, 2560],
            "a_token": Tensor [N_token, 768] (may be None for old format),
            "n_token": int,
            "metadata": dict,
            "residue_index": Tensor [N_token] (optional, label_seq_id - continuous 1-indexed),
            "auth_residue_index": Tensor [N_token] (optional, auth_seq_id - original PDB numbering,
                NOTE: unreliable for tokens where token_is_resolved=False),
            "asym_id": Tensor [N_token] (optional),
            "entity_id": Tensor [N_token] (optional),
            "sym_id": Tensor [N_token] (optional),
            "coordinate_mask": Tensor [N_atom] (optional, from atom_array.is_resolved),
            "token_is_resolved": Tensor [N_token] (optional, centre atom is_resolved status,
                use this to filter reliable auth_residue_index values),
            "asym_id_to_auth_asym_id": dict[int, str] (optional, maps asym_id -> auth_asym_id),
            "a_token_layer_{i}": Tensor [N_token, 768] (optional, intermediate DiffusionTransformerBlock
                output at block index i, without LayerNorm; present when dumped with
                --dump-intermediate-layers),
            "intermediate_layer_indices": list[int] (optional, sorted list of available
                intermediate layer indices; present when any a_token_layer_* keys exist),
        }
    """
    with safe_open(path, framework="pt") as f:
        s = f.get_tensor("s")
        esm_embedding = f.get_tensor("esm_embedding")
        n_token = f.get_tensor("n_token").item()
        metadata = f.metadata()

        keys = f.keys()
        a_token = f.get_tensor("a_token") if "a_token" in keys else None
        residue_index = f.get_tensor("residue_index") if "residue_index" in keys else None
        auth_residue_index = f.get_tensor("auth_residue_index") if "auth_residue_index" in keys else None
        asym_id = f.get_tensor("asym_id") if "asym_id" in keys else None
        entity_id = f.get_tensor("entity_id") if "entity_id" in keys else None
        sym_id = f.get_tensor("sym_id") if "sym_id" in keys else None
        coordinate_mask = f.get_tensor("coordinate_mask") if "coordinate_mask" in keys else None
        token_is_resolved = f.get_tensor("token_is_resolved") if "token_is_resolved" in keys else None

        # Load intermediate layer features (a_token_layer_0, a_token_layer_5, ...)
        intermediate_layer_keys = sorted(
            [k for k in keys if k.startswith("a_token_layer_")],
            key=lambda k: int(k.split("_")[-1]),
        )
        intermediate_tensors = {k: f.get_tensor(k) for k in intermediate_layer_keys}

    result = {
        "s": s,
        "esm_embedding": esm_embedding,
        "a_token": a_token,
        "n_token": n_token,
        "metadata": metadata,
    }

    if residue_index is not None:
        result["residue_index"] = residue_index
    if auth_residue_index is not None:
        result["auth_residue_index"] = auth_residue_index
    if asym_id is not None:
        result["asym_id"] = asym_id
    if entity_id is not None:
        result["entity_id"] = entity_id
    if sym_id is not None:
        result["sym_id"] = sym_id
    if coordinate_mask is not None:
        result["coordinate_mask"] = coordinate_mask
    if token_is_resolved is not None:
        result["token_is_resolved"] = token_is_resolved

    # Add intermediate layer tensors and a convenience index list
    if intermediate_tensors:
        result.update(intermediate_tensors)
        result["intermediate_layer_indices"] = [
            int(k.split("_")[-1]) for k in intermediate_layer_keys
        ]

    # Parse asym_id_to_auth_asym_id mapping from metadata
    if metadata and "asym_id_to_auth_asym_id" in metadata:
        mapping_str = metadata["asym_id_to_auth_asym_id"]
        # JSON keys are strings, convert back to int
        mapping = json.loads(mapping_str)
        result["asym_id_to_auth_asym_id"] = {int(k): v for k, v in mapping.items()}

    return result


def get_protein_mask(atom_array, token_array) -> np.ndarray:
    """
    Get boolean mask indicating which tokens are proteins.

    This follows the same logic as Protenix's ESMFeaturizer:
        centre_atom_array.chain_mol_type == "protein"

    With fallback to hetero attribute for non-Protenix parsed structures.

    Args:
        atom_array: Biotite AtomArray with chain_mol_type or hetero attribute
        token_array: TokenArray with centre_atom_index annotation

    Returns:
        np.ndarray[bool]: Boolean mask of shape [N_token], True for protein tokens
    """
    centre_atom_indices = token_array.get_annotation("centre_atom_index")
    centre_atoms = atom_array[centre_atom_indices]
    # Prefer chain_mol_type (semantic molecule type) over hetero (HETATM flag)
    # chain_mol_type is set by Protenix parser and is consistent with ESMFeaturizer
    if hasattr(centre_atoms, "chain_mol_type"):
        is_protein = centre_atoms.chain_mol_type == "protein"
    else:
        # Fallback to hetero for non-Protenix parsed structures
        is_protein = ~centre_atoms.hetero
    return is_protein


def build_asym_id_to_auth_asym_id_mapping(
    atom_array, token_array, protein_only: bool = True
) -> dict[int, str]:
    """
    Build mapping from asym_id (int) to auth_asym_id (str) using atom_array and token_array.

    Args:
        atom_array: Biotite AtomArray with asym_id_int and auth_asym_id attributes
        token_array: TokenArray with centre_atom_index annotation
        protein_only: If True, only include protein chains in the mapping (default: True)

    Returns:
        dict[int, str]: Mapping from asym_id (int) -> auth_asym_id (str, e.g., "H", "L", "A")
    """
    # Get centre atom indices from token_array
    centre_atom_indices = token_array.get_annotation("centre_atom_index")
    centre_atoms = atom_array[centre_atom_indices]

    # Get protein mask if filtering
    if protein_only:
        # Prefer chain_mol_type (semantic molecule type) over hetero (HETATM flag)
        # chain_mol_type is set by Protenix parser and is consistent with ESMFeaturizer
        if hasattr(centre_atoms, "chain_mol_type"):
            is_protein = centre_atoms.chain_mol_type == "protein"
        else:
            # Fallback to hetero for non-Protenix parsed structures
            is_protein = ~centre_atoms.hetero
    else:
        is_protein = np.ones(len(centre_atoms), dtype=bool)

    # Build mapping: {asym_id_int: auth_asym_id} for protein chains only
    asym_id_to_auth_asym_id = {}
    for i, (asym_int, auth_id) in enumerate(zip(centre_atoms.asym_id_int, centre_atoms.auth_asym_id)):
        if not is_protein[i]:
            continue
        asym_int = int(asym_int)
        auth_id = str(auth_id)
        if asym_int not in asym_id_to_auth_asym_id:
            asym_id_to_auth_asym_id[asym_int] = auth_id

    return asym_id_to_auth_asym_id


def get_chain_embedding_by_auth_asym_id(
    embedding_data: dict,
    auth_asym_id: str,
    embedding_key: str = "s",
    case_insensitive: bool = False,
) -> torch.Tensor:
    """
    Extract embedding for a specific chain by its auth_asym_id.

    Args:
        embedding_data: dict returned by load_full_embedding()
        auth_asym_id: Chain ID from original CIF file (e.g., "H", "L", "A")
        embedding_key: Which embedding to extract ("s", "esm_embedding", "a_token")
        case_insensitive: If True, match chain ID case-insensitively

    Returns:
        Tensor [N_chain_token, D]: Embedding for the specified chain

    Raises:
        ValueError: If chain ID not found or mapping not available

    Example:
        >>> data = load_full_embedding("protein.safetensors")
        >>> h_embedding = get_chain_embedding_by_auth_asym_id(data, "H")
        >>> l_embedding = get_chain_embedding_by_auth_asym_id(data, "L", embedding_key="esm_embedding")
    """
    mapping = embedding_data.get("asym_id_to_auth_asym_id")
    if mapping is None:
        raise ValueError(
            "No asym_id_to_auth_asym_id mapping found in embedding data. "
            "Re-dump with updated code to include chain ID mapping."
        )

    asym_id = embedding_data.get("asym_id")
    if asym_id is None:
        raise ValueError("No asym_id found in embedding data.")

    # Find the asym_id_int for the requested auth_asym_id
    target_asym_id_int = None
    if case_insensitive:
        auth_asym_id_lower = auth_asym_id.lower()
        for asym_int, auth_id in mapping.items():
            if auth_id.lower() == auth_asym_id_lower:
                target_asym_id_int = asym_int
                break
    else:
        for asym_int, auth_id in mapping.items():
            if auth_id == auth_asym_id:
                target_asym_id_int = asym_int
                break

    if target_asym_id_int is None:
        available = list(mapping.values())
        raise ValueError(
            f"auth_asym_id '{auth_asym_id}' not found. Available chains: {available}"
        )

    # Extract embedding for this chain
    chain_mask = (asym_id == target_asym_id_int)
    embedding = embedding_data.get(embedding_key)
    if embedding is None:
        raise ValueError(f"Embedding key '{embedding_key}' not found in embedding data.")

    return embedding[chain_mask]


def get_all_chain_embeddings(
    embedding_data: dict,
    embedding_key: str = "s",
) -> dict[str, torch.Tensor]:
    """
    Extract embeddings for all chains, keyed by auth_asym_id.

    Args:
        embedding_data: dict returned by load_full_embedding()
        embedding_key: Which embedding to extract ("s", "esm_embedding", "a_token")

    Returns:
        dict[str, Tensor]: {auth_asym_id: embedding_tensor} for each chain

    Example:
        >>> data = load_full_embedding("protein.safetensors")
        >>> all_chains = get_all_chain_embeddings(data)
        >>> h_embedding = all_chains["H"]
        >>> l_embedding = all_chains["L"]
    """
    mapping = embedding_data.get("asym_id_to_auth_asym_id")
    if mapping is None:
        raise ValueError(
            "No asym_id_to_auth_asym_id mapping found in embedding data. "
            "Re-dump with updated code to include chain ID mapping."
        )

    asym_id = embedding_data.get("asym_id")
    if asym_id is None:
        raise ValueError("No asym_id found in embedding data.")

    embedding = embedding_data.get(embedding_key)
    if embedding is None:
        raise ValueError(f"Embedding key '{embedding_key}' not found in embedding data.")

    result = {}
    for asym_int, auth_id in mapping.items():
        chain_mask = (asym_id == asym_int)
        result[auth_id] = embedding[chain_mask]

    return result


def has_a_token(path: str) -> bool:
    """Check if safetensors file contains a_token"""
    try:
        with safe_open(path, framework="pt") as f:
            return "a_token" in f.keys()
    except Exception:
        return False


# ============================================================================
# FullEmbeddingDumper Class
# ============================================================================

class FullEmbeddingDumper(EmbeddingDumper):
    """
    Full Protenix Embedding Dumper (CIF-only mode)

    Processes CIF files directly to extract:
    1. s_trunk: Pairformer output [N_token, 384]
    2. esm_embedding: ESM token embedding [N_token, 2560]
    3. a_token: Diffusion token embedding [N_token, 768]

    Uses GT coordinates directly from CIF files for accurate feature extraction.
    """

    def __init__(
        self,
        output_dir: str,
        protenix_path: str,
        device: str = "cuda",
        dtype: str = "bf16",
        rank: int = 0,
        world_size: int = 1,
        triangle_by_torch: Optional[bool] = None,
        ref_pos_augment: bool = False,
        sigma: float = 0.0,
        use_bioassembly: bool = True,
        protein_only: bool = False,
        precomputed_dir: Optional[str] = None,
        max_esm_seq_length: Optional[int] = None,
        max_chain_count: Optional[int] = None,
        min_chain_count: Optional[int] = None,
        intermediate_layers: Optional[list[int]] = None,
    ):
        """
        Args:
            output_dir: Output directory
            protenix_path: Path to pretrained Protenix encoder
            device: Device ("cuda" or "cpu")
            dtype: Data type ("bf16" or "fp32")
            rank: Current process rank (for multi-GPU)
            world_size: Total number of processes (for multi-GPU)
            triangle_by_torch: Use PyTorch for triangle operations
            ref_pos_augment: Apply random augmentation to ref_pos (default False for determinism)
            sigma: Noise level for diffusion (default 0.0 for no noise)
            use_bioassembly: Use bioassembly structure (default True); if False, use asymmetric unit
            protein_only: If True, filter to protein-only tokens after inference.
                         If False (default), keep all tokens (including ligands, DNA/RNA, ions, etc.)
            precomputed_dir: Directory containing precomputed processor outputs. If specified,
                will attempt to load precomputed features instead of processing CIF files.
                Falls back to process_from_cif if precomputed file not found.
            max_esm_seq_length: Maximum allowed ESM sequence length. If specified, samples
                with esm_input_ids length exceeding this threshold will be skipped.
                If None (default), no length filtering is applied.
            max_chain_count: Maximum allowed number of chains (based on asym_id_to_auth_asym_id length).
                If specified, samples with chain count exceeding this threshold will be skipped.
                If None (default), no upper limit filtering is applied.
            min_chain_count: Minimum required number of chains (based on asym_id_to_auth_asym_id length).
                If specified, samples with chain count below this threshold will be skipped.
                If None (default), no lower limit filtering is applied.
        """
        super().__init__(
            output_dir=output_dir,
            protenix_path=protenix_path,
            device=device,
            dtype=dtype,
            rank=rank,
            world_size=world_size,
            triangle_by_torch=triangle_by_torch,
        )
        self.ref_pos_augment = ref_pos_augment
        self.sigma = sigma
        self.use_bioassembly = use_bioassembly
        self.protein_only = protein_only
        self.precomputed_dir = precomputed_dir
        self.max_esm_seq_length = max_esm_seq_length
        self.max_chain_count = max_chain_count
        self.min_chain_count = min_chain_count
        self.intermediate_layers = intermediate_layers  # list[int] or None

        # Override output_dir with suffix for sigma/intermediate_layers
        self.output_dir = build_output_dir(output_dir, sigma, intermediate_layers)

        self._full_encoder = None
        self._processor = None

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown on SIGTERM/SIGINT."""
        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.warning(f"Received {sig_name} signal. Initiating graceful shutdown...")
            log_memory_usage("Final memory state")
            logger.info("Exiting gracefully.")
            sys.exit(0)

        # Register handlers for SIGTERM (kill) and SIGINT (Ctrl+C)
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        logger.info("Signal handlers registered for SIGTERM and SIGINT")

    @property
    def full_encoder(self):
        """Lazy load full encoder with Diffusion"""
        if self._full_encoder is None:
            from proteor1.understand.protenix_encoder import ProtenixEncoder

            logger.info(f"Loading ProtenixEncoder with Diffusion from {self.protenix_path}...")
            self._full_encoder = ProtenixEncoder.from_pretrained(
                self.protenix_path,
                load_esm=True,
                load_diffusion=True,
                device=self.device,
                dtype=self.dtype_str,
                triangle_by_torch=self.triangle_by_torch,
            )
            # Convert only the non-Diffusion modules to bf16; keep Diffusion in fp32.
            # This gives us:
            # 1. Non-Diffusion modules pair with autocast with no dynamic-cast overhead (faster)
            # 2. Diffusion modules stay in fp32, preserving weight precision when disable_amp=True
            # Note: fp32 -> bf16 conversion is deterministic; pre-conversion and autocast dynamic conversion yield identical results.
            for name, module in self._full_encoder.named_children():
                if name != "diffusion_module" and module is not None:
                    module.to(dtype=self.dtype)
            self._full_encoder.eval()
        return self._full_encoder

    @property
    def processor(self):
        """Lazy load ProtenixProcessor"""
        if self._processor is None:
            from proteor1.understand.protenix_encoder import ProtenixProcessor

            logger.info(f"Loading ProtenixProcessor from {self.protenix_path}...")
            self._processor = ProtenixProcessor.from_pretrained(
                self.protenix_path,
                init_esm_tokenizer=True,
            )
        return self._processor

    def process_protein_from_cif(
        self,
        cif_path: str,
        protein_id: str,
        database: str = "pdb",
        source_file: str = "",
        force: bool = False,
    ) -> dict:
        """
        Process single protein directly from CIF file (GT coordinates).

        This is the recommended approach for feature dumping as it:
        1. Uses actual GT coordinates from the crystal structure
        2. Ensures atom ordering consistency
        3. Avoids coordinate alignment issues between JSON and CIF pipelines

        This method handles preprocessing (CIF parsing, tokenization, featurization)
        and then delegates to _run_gpu_inference_and_save for GPU inference and saving.

        Args:
            cif_path: Path to CIF file
            protein_id: Protein identifier
            database: Database name (default "pdb")
            source_file: Source file for metadata
            force: Force overwrite existing files

        Returns:
            Result dict with processing status
        """
        try:
            # Preprocessing: Use ProtenixProcessor.process_from_cif()
            # This handles CIF parsing, tokenization, featurization, and ESM tokenization
            # If protein_only=True, the processor filters to protein atoms before featurization
            processor_output = self.processor.process_from_cif(
                cif_path=cif_path,
                assembly_id="1",
                ref_pos_augment=self.ref_pos_augment,
                return_atom_array=True,
                return_token_array=True,
                use_bioassembly=self.use_bioassembly,
                protein_only=self.protein_only,
            )

            features_dict = processor_output.input_feature_dict
            atom_array = processor_output.atom_array
            token_array = processor_output.token_array

            # Build asym_id -> auth_asym_id mapping for chain ID recovery
            asym_id_to_auth_asym_id = build_asym_id_to_auth_asym_id_mapping(
                atom_array, token_array, protein_only=self.protein_only
            )

            # Delegate GPU inference and saving to _run_gpu_inference_and_save
            # Use source_file if provided, otherwise use cif_path
            return self._run_gpu_inference_and_save(
                features_dict=features_dict,
                atom_array=atom_array,
                token_array=token_array,
                protein_id=protein_id,
                database=database,
                source_file=source_file or cif_path,
                force=force,
                asym_id_to_auth_asym_id=asym_id_to_auth_asym_id,
            )

        except ValueError as e:
            # Handle preprocessing errors - check for specific "No protein atoms found" error
            error_msg = str(e)
            if "No protein atoms found" in error_msg:
                error_type = "NoProteinAtoms"
            else:
                error_type = "ValueError"
            error_detail = {
                "action": "error",
                "protein_id": protein_id,
                "database": database,
                "cif_path": cif_path,
                "error": error_msg,
                "error_type": error_type,
                "traceback": traceback.format_exc(),
            }
            self._log_action(error_detail)
            logger.error(f"Error on {protein_id} from {cif_path}: {error_type}: {e}")
            return error_detail

        except Exception as e:
            # Handle other preprocessing errors (before GPU inference)
            error_detail = {
                "action": "error",
                "protein_id": protein_id,
                "database": database,
                "cif_path": cif_path,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }
            self._log_action(error_detail)
            logger.error(f"Error on {protein_id} from {cif_path}: {type(e).__name__}: {e}")
            return error_detail

    def process_protein_from_file(
        self,
        file_path: str,
        protein_id: str,
        source_file: str = "",
        force: bool = False,
    ) -> dict:
        """
        Process single protein from file, auto-detecting format and database.

        Supports:
        - .cif, .cif.gz: Direct CIF processing -> database: cif_full
        - .pdb, .pdb.gz: Convert to CIF first, then process -> database: pdb_full

        The database is automatically determined based on file extension:
        - .pdb or .pdb.gz files -> "pdb_full"
        - .cif or .cif.gz files -> "cif_full"

        Args:
            file_path: Path to protein structure file (.cif, .cif.gz, .pdb, .pdb.gz)
            protein_id: Protein identifier
            source_file: Source file for metadata
            force: Force overwrite existing files

        Returns:
            Result dict with processing status
        """
        lower_path = file_path.lower()

        # Determine database based on file extension and processing options
        suffix = get_database_suffix(self.use_bioassembly, self.protein_only)
        if lower_path.endswith(".pdb") or lower_path.endswith(".pdb.gz"):
            database = "pdb_full" + suffix
        else:  # .cif or .cif.gz
            database = "cif_full" + suffix

        # CIF files: direct processing
        if lower_path.endswith(".cif") or lower_path.endswith(".cif.gz"):
            return self.process_protein_from_cif(
                cif_path=file_path,
                protein_id=protein_id,
                database=database,
                source_file=source_file or file_path,
                force=force,
            )

        # PDB files: convert to CIF first
        if lower_path.endswith(".pdb") or lower_path.endswith(".pdb.gz"):
            from protenix.data.utils import pdb_to_cif

            try:
                # Handle .pdb.gz: decompress to temp .pdb file first
                if lower_path.endswith(".pdb.gz"):
                    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp_pdb:
                        tmp_pdb_path = tmp_pdb.name
                        with gzip.open(file_path, "rb") as gz_in:
                            tmp_pdb.write(gz_in.read())
                    pdb_path_to_convert = tmp_pdb_path
                else:
                    pdb_path_to_convert = file_path
                    tmp_pdb_path = None

                # Convert PDB to CIF
                with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as tmp_cif:
                    tmp_cif_path = tmp_cif.name

                pdb_to_cif(pdb_path_to_convert, tmp_cif_path)

                # Clean up temp PDB file if created
                if tmp_pdb_path is not None:
                    os.unlink(tmp_pdb_path)

                # Process the converted CIF
                result = self.process_protein_from_cif(
                    cif_path=tmp_cif_path,
                    protein_id=protein_id,
                    database=database,
                    source_file=source_file or file_path,
                    force=force,
                )

                # Clean up temp CIF file
                os.unlink(tmp_cif_path)

                return result

            except Exception as e:
                # Clean up temp files on error
                if "tmp_pdb_path" in locals() and tmp_pdb_path is not None and os.path.exists(tmp_pdb_path):
                    os.unlink(tmp_pdb_path)
                if "tmp_cif_path" in locals() and os.path.exists(tmp_cif_path):
                    os.unlink(tmp_cif_path)

                error_detail = {
                    "action": "error",
                    "protein_id": protein_id,
                    "database": database,
                    "file_path": file_path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                }
                self._log_action(error_detail)
                logger.error(f"Error converting PDB to CIF for {protein_id}: {type(e).__name__}: {e}")
                return error_detail

        # Unsupported format
        error_detail = {
            "action": "error",
            "protein_id": protein_id,
            "database": database,
            "file_path": file_path,
            "error": f"Unsupported file format: {file_path}",
            "error_type": "UnsupportedFormat",
        }
        self._log_action(error_detail)
        logger.error(f"Unsupported file format for {protein_id}: {file_path}")
        return error_detail

    def _run_gpu_inference_and_save(
        self,
        features_dict: dict,
        atom_array,
        token_array,
        protein_id: str,
        database: str,
        source_file: str,
        force: bool = False,
        asym_id_to_auth_asym_id: Optional[dict[int, str]] = None,
    ) -> dict:
        """
        Run GPU inference on preprocessed features and save the result.

        This method is extracted from process_protein_from_cif to support
        parallel preprocessing with DataLoader.

        Args:
            features_dict: Preprocessed feature dictionary (CPU tensors)
            atom_array: AtomArray from processor
            token_array: TokenArray from processor
            protein_id: Protein identifier
            database: Database name
            source_file: Source file path for metadata
            force: Force overwrite existing files
            asym_id_to_auth_asym_id: Mapping from asym_id (int) to auth_asym_id (str)

        Returns:
            Result dict with processing status
        """
        path = get_embedding_path(self.model_name, database, protein_id, self.output_dir)

        # Skip if already exists with a_token
        if not force and os.path.exists(path):
            try:
                if has_a_token(path):
                    result = {
                        "action": "skip",
                        "protein_id": protein_id,
                        "database": database,
                        "reason": "already_exists_with_a_token",
                    }
                    self._log_action(result)
                    return result
            except Exception:
                pass  # Corrupted file, reprocess

        try:
            # Move features to device
            input_feature_dict_on_device = move_protenix_features_to_device(
                features_dict,
                self.full_encoder.device,
                dtype=self.full_encoder.dtype,
            )

            # NaN detection with retry logic
            max_retries = 10
            s_trunk, esm_embedding, a_token = None, None, None

            for attempt in range(max_retries):
                with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=self.dtype, enabled=torch.cuda.is_available()):
                    # Use forward() with return_full_output=True to get PairformerOutput
                    pairformer_output = self.full_encoder.forward(
                        input_feature_dict=input_feature_dict_on_device,
                        atom_array=atom_array,
                        token_array=token_array,
                        return_full_output=True,
                    )

                    # Extract features from PairformerOutput
                    s_inputs = pairformer_output.s_inputs
                    s_trunk = pairformer_output.s_trunk
                    z_trunk = pairformer_output.z_trunk
                    esm_embedding = pairformer_output.esm_embedding
                    input_feature_dict = pairformer_output.input_feature_dict

                    # Get GT coordinates directly from atom_array
                    gt_coords = torch.from_numpy(atom_array.coord).to(
                        self.full_encoder.device, dtype=self.full_encoder.dtype
                    )

                    # Get coordinate_mask from atom_array.is_resolved
                    coordinate_mask = torch.from_numpy(atom_array.is_resolved.astype(int)).long().to(
                        self.full_encoder.device
                    )

                    # Register intermediate layer hooks if requested
                    _captured_intermediates = {}
                    _hooks = []
                    try:
                        if self.intermediate_layers:
                            _captured_intermediates, _hooks = register_diffusion_intermediate_hooks(
                                self.full_encoder.diffusion_module, self.intermediate_layers
                            )

                        # Extract a_token using forward_diffusion
                        a_token = self.full_encoder.forward_diffusion(
                            input_feature_dict=input_feature_dict,
                            s_inputs=s_inputs,
                            s_trunk=s_trunk,
                            z_trunk=z_trunk,
                            gt_coords=gt_coords,
                            coordinate_mask=coordinate_mask,
                            sigma=self.sigma,
                            centre_only=True,
                        )
                    finally:
                        # Remove hooks unconditionally to prevent hook leaks on retry
                        remove_diffusion_hooks(_hooks)

                    a_token = a_token.to(self.dtype)

                    # Process captured intermediate features: match shape and dtype of a_token
                    intermediate_features = {}
                    for layer_idx, feat in _captured_intermediates.items():
                        # feat shape: [n_sample, N_token, 768] or [N_token, 768]
                        if feat.dim() == 3 and feat.shape[0] == 1:
                            feat = feat.squeeze(0)  # [N_token, 768]
                        intermediate_features[layer_idx] = feat.to(self.dtype)

                # Synchronize CUDA to catch any device-side errors early
                # This ensures errors are detected here rather than during later cleanup
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                # Check for NaN
                has_nan = (
                    torch.isnan(s_trunk).any().item() or
                    torch.isnan(esm_embedding).any().item() or
                    torch.isnan(a_token).any().item() or
                    any(torch.isnan(feat).any().item() for feat in intermediate_features.values())
                )

                if not has_nan:
                    break

                logger.warning(f"NaN detected in {protein_id} (attempt {attempt + 1}/{max_retries}), retrying...")
                gc.collect()
                safe_cuda_empty_cache()

            # After all retries, check if NaN still persists
            if (torch.isnan(s_trunk).any().item() or
                torch.isnan(esm_embedding).any().item() or
                torch.isnan(a_token).any().item() or
                any(torch.isnan(feat).any().item() for feat in intermediate_features.values())):
                result = {
                    "action": "error",
                    "protein_id": protein_id,
                    "database": database,
                    "error": f"NaN persists after {max_retries} retries",
                    "error_type": "PersistentNaN",
                }
                self._log_action(result)
                logger.error(f"NaN persists in {protein_id} after {max_retries} retries, skipping...")
                return result

            # When protein_only=True, filtering already happened in process_from_cif()
            # So atom_array and token_array already contain only protein atoms/tokens,
            # and the embeddings are already for protein-only tokens.
            n_token = s_trunk.shape[0]

            # Get position info directly from features_dict (already filtered if protein_only=True)
            residue_index = features_dict.get("residue_index")
            asym_id = features_dict.get("asym_id")
            entity_id = features_dict.get("entity_id")
            sym_id = features_dict.get("sym_id")

            # Build asym_id -> auth_asym_id mapping if not provided
            # Use protein_only setting to control whether mapping includes only protein chains
            if asym_id_to_auth_asym_id is None:
                asym_id_to_auth_asym_id = build_asym_id_to_auth_asym_id_mapping(
                    atom_array, token_array, protein_only=self.protein_only
                )

            # Extract auth_residue_index (original PDB residue numbering, auth_seq_id)
            # This is needed for downstream tasks that use PDB author numbering (e.g., Chothia/Kabat)
            # WARNING: auth_residue_index is UNRELIABLE for missing residues (token_is_resolved=False).
            # For missing residues, auth_seq_id is empty in CIF, so we fallback to res_id which may
            # conflict with other residues' auth_seq_id. Always filter by token_is_resolved=True first.
            centre_atom_indices = token_array.get_annotation("centre_atom_index")
            centre_atoms = atom_array[centre_atom_indices]
            # auth_seq_id is stored as string, convert to int
            # Use res_id as fallback for missing residues (where auth_seq_id is empty string)
            auth_seq_id_all = np.array([
                int(x) if x != '' else int(centre_atoms.res_id[i])
                for i, x in enumerate(centre_atoms.auth_seq_id)
            ])
            auth_residue_index = torch.from_numpy(auth_seq_id_all).long()

            # Extract token_is_resolved: token-level coordinate mask based on centre atom is_resolved
            # This indicates which tokens have resolved coordinates in the crystal structure.
            # Tokens with is_resolved=False have coordinates set to 0.0 (missing atoms added by get_bioassembly()).
            # IMPORTANT: Use this to filter auth_residue_index - only trust values where token_is_resolved=True.
            centre_atom_is_resolved = centre_atoms.is_resolved  # [N_token]
            token_is_resolved = torch.from_numpy(centre_atom_is_resolved.astype(int)).long()

            # Note: atom-level coordinate_mask is not saved because it's at atom-level while embeddings
            # are at token-level. We now save token_is_resolved instead, which is the token-level version.

            # Save
            save_full_embedding(
                path=path,
                s=s_trunk,
                esm_embedding=esm_embedding,
                a_token=a_token,
                n_token=n_token,
                metadata={
                    "protein_id": protein_id,
                    "database": database,
                    "n_token": n_token,
                    "s_dim": s_trunk.shape[-1],
                    "esm_dim": esm_embedding.shape[-1],
                    "a_token_dim": a_token.shape[-1],
                    "sigma": self.sigma,
                    "ref_pos_augment": self.ref_pos_augment,
                    "use_bioassembly": self.use_bioassembly,
                    "source_file": source_file,
                    "model_name": self.model_name,
                    "cif_path": source_file,  # Use source_file as cif_path for consistency
                    "coord_source": "cif_gt",
                    "protein_only": self.protein_only,  # Whether only protein tokens are saved
                },
                dtype=self.dtype,
                residue_index=residue_index,
                asym_id=asym_id,
                entity_id=entity_id,
                sym_id=sym_id,
                coordinate_mask=None,  # Atom-level mask not saved; use token_is_resolved instead
                asym_id_to_auth_asym_id=asym_id_to_auth_asym_id,
                auth_residue_index=auth_residue_index,
                token_is_resolved=token_is_resolved,
                intermediate_features=intermediate_features,
            )

            result = {
                "action": "dump",
                "protein_id": protein_id,
                "database": database,
                "n_token": n_token,
                "path": path,
                "coord_source": "cif_gt",
            }
            self._log_action(result)
            return result

        except torch.cuda.OutOfMemoryError as e:
            gc.collect()
            safe_cuda_empty_cache()
            result = {
                "action": "error",
                "protein_id": protein_id,
                "database": database,
                "error": f"CUDA OOM: {e}",
                "error_type": "OOM",
                "traceback": traceback.format_exc(),
            }
            self._log_action(result)
            logger.warning(f"OOM on {protein_id}, cleared CUDA cache and continuing...")
            return result

        except RuntimeError as e:
            # Check for CUDA device-side errors (e.g., device-side assert)
            if is_cuda_device_error(e):
                gc.collect()
                cuda_ok = safe_cuda_empty_cache()
                result = {
                    "action": "error",
                    "protein_id": protein_id,
                    "database": database,
                    "error": f"CUDA device error: {e}",
                    "error_type": "CUDADeviceError",
                    "traceback": traceback.format_exc(),
                    "cuda_recoverable": cuda_ok,
                }
                self._log_action(result)
                logger.error(f"CUDA device error on {protein_id}: {e}")
                if not cuda_ok:
                    logger.error("CUDA context corrupted. Process may need restart for full recovery.")
                return result
            # Re-raise non-CUDA RuntimeErrors to be caught by the generic handler
            raise

        except Exception as e:
            error_detail = {
                "action": "error",
                "protein_id": protein_id,
                "database": database,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }
            self._log_action(error_detail)
            logger.error(f"Error on {protein_id}: {type(e).__name__}: {e}")
            return error_detail

    def _create_dataloader(
        self,
        protein_files: list[str],
        num_workers: int,
        force: bool,
    ) -> DataLoader:
        """
        Create a DataLoader for protein file processing.

        This is extracted as a helper method to support DataLoader rebuild
        after worker crashes.

        Args:
            protein_files: List of protein structure file paths
            num_workers: Number of DataLoader workers (0 or 1 only for precise crash tracking)
            force: Force overwrite existing files

        Returns:
            Configured DataLoader instance
        """
        dataset = CIFDataset(
            protein_files=protein_files,
            protenix_path=self.protenix_path,
            ref_pos_augment=self.ref_pos_augment,
            output_dir=self.output_dir,
            model_name=self.model_name,
            force=force,
            use_bioassembly=self.use_bioassembly,
            protein_only=self.protein_only,
            precomputed_dir=self.precomputed_dir,
            max_esm_seq_length=self.max_esm_seq_length,
            max_chain_count=self.max_chain_count,
            min_chain_count=self.min_chain_count,
        )

        # When using precomputed files, we can enable prefetch for better performance
        # since pkl.gz loading is safe and won't cause worker crashes
        use_precomputed_mode = self.precomputed_dir is not None
        if use_precomputed_mode and num_workers > 0:
            prefetch_factor = 2  # Enable prefetch for parallel pkl.gz loading
            persistent_workers = True  # Keep workers alive for better performance
        else:
            prefetch_factor = None  # Disabled for precise crash file tracking
            persistent_workers = False

        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn_first,
            worker_init_fn=worker_init_fn,
            persistent_workers=persistent_workers,
            timeout=300 if num_workers > 0 else 0,  # 5 minute timeout to detect stuck workers
        )

    def dump_protein_files_parallel(
        self,
        protein_files: list[str],
        num_workers: int = 1,
        force: bool = False,
        enable_crash_recovery: bool = True,
    ) -> dict:
        """
        Dump features from protein structure files with parallel CPU preprocessing.

        Uses DataLoader to process protein structure files.

        Two modes of operation:
        1. **Precomputed mode** (when precomputed_dir is set):
           - Loads pre-processed features from pkl.gz files
           - Allows higher num_workers (2-8) for parallel loading
           - Enables prefetch_factor=2 and persistent_workers=True
           - Safer since pkl.gz loading rarely crashes

        2. **Online mode** (without precomputed_dir):
           - Processes CIF files on-the-fly
           - Restricts num_workers to 0 or 1 for precise crash tracking
           - Disables prefetch for crash file identification
           - Supports worker crash recovery

        The database is automatically determined for each file based on extension:
        - .pdb or .pdb.gz files -> "pdb_full"
        - .cif or .cif.gz files -> "cif_full"

        Args:
            protein_files: List of protein structure file paths (.cif, .cif.gz, .pdb, .pdb.gz)
            num_workers: Number of DataLoader workers.
                         With precomputed_dir: 0-8 allowed (higher = faster pkl.gz loading)
                         Without precomputed_dir: 0 or 1 only (for crash tracking)
            force: Force overwrite existing files
            enable_crash_recovery: Enable automatic recovery from worker crashes.
                                   Auto-disabled in precomputed mode (pkl.gz loading is safe
                                   and multi-worker mode cannot precisely track crash files).

        Returns:
            Statistics dict including worker_crashes and blacklist info
        """
        # Determine if we're in precomputed mode (safer, allows more workers)
        use_precomputed_mode = self.precomputed_dir is not None

        # Validate num_workers based on mode
        if use_precomputed_mode:
            # Auto-disable crash recovery in precomputed mode
            # (crash recovery requires num_workers<=1 and prefetch disabled to work correctly)
            if enable_crash_recovery:
                logger.info("Precomputed mode: auto-disabling crash recovery (not needed for pkl.gz loading)")
                enable_crash_recovery = False
            logger.info(f"Precomputed mode: num_workers={num_workers}, prefetch=2, persistent_workers=True")
        else:
            # Online mode: restrict to 0 or 1 for precise crash tracking
            if num_workers not in (0, 1):
                logger.warning(f"num_workers={num_workers} not supported without precomputed_dir. Only 0 or 1 allowed for precise crash tracking. Using 1.")
                num_workers = 1
            if num_workers > 0:
                logger.info(f"Online mode: num_workers={num_workers}, prefetch=disabled (for crash tracking)")
            else:
                logger.info("Online mode: sequential processing (num_workers=0)")
            if enable_crash_recovery:
                logger.info("Worker crash recovery enabled")

        # Multi-GPU sharding
        if self.world_size > 1:
            protein_files = protein_files[self.rank::self.world_size]
            logger.info(f"Rank {self.rank}/{self.world_size}: assigned {len(protein_files)} protein files")

        # Initialize stats
        stats = {
            "dumped": 0,
            "skipped": 0,
            "errors": 0,
            "preprocess_errors": 0,
            "worker_crashes": 0,
            "blacklisted_skipped": 0,
        }

        # Pre-filter: check which embeddings already exist before processing
        # This avoids loading data just to skip it in __getitem__
        if not force:
            files_to_process = []
            suffix = get_database_suffix(self.use_bioassembly, self.protein_only)

            for file_path in tqdm(protein_files, desc=f"Checking existing (rank {self.rank})"):
                # Extract protein_id and database from file path
                basename = os.path.basename(file_path)
                protein_id = basename.split(".")[0].upper()
                lower_path = file_path.lower()
                if lower_path.endswith(".pdb") or lower_path.endswith(".pdb.gz"):
                    database = "pdb_full" + suffix
                else:
                    database = "cif_full" + suffix

                output_path = get_embedding_path(self.model_name, database, protein_id, self.output_dir)
                if os.path.exists(output_path):
                    try:
                        if has_a_token(output_path):
                            stats["skipped"] += 1
                            self._log_action({
                                "action": "skip",
                                "protein_id": protein_id,
                                "database": database,
                                "reason": "already_exists_with_a_token",
                            })
                            continue
                    except Exception:
                        pass  # Corrupted file, continue to process
                files_to_process.append(file_path)

            logger.info(
                f"Rank {self.rank}: {len(files_to_process)} to process, "
                f"{stats['skipped']} already exist (skipped)"
            )
            protein_files = files_to_process
        else:
            logger.info(f"Rank {self.rank}: {len(protein_files)} to process (force mode)")
        details = []

        # Constants for periodic operations
        CLEANUP_INTERVAL = 100  # Run gc.collect() and torch.cuda.empty_cache() every N proteins
        MEMORY_LOG_INTERVAL = 500  # Log memory usage every N proteins
        MAX_ERROR_DETAILS = 1000  # Limit error details list size to prevent memory bloat
        MAX_DATALOADER_REBUILDS = 50  # Maximum number of DataLoader rebuilds to prevent infinite loops

        processed_count = 0
        error_details_truncated = False
        dataloader_rebuild_count = 0
        current_file_idx = 0  # Track the current file index being processed

        # Create initial DataLoader
        remaining_files = protein_files.copy()
        dataloader = self._create_dataloader(remaining_files, num_workers, force)

        while True:
            try:
                dataloader_iter = iter(dataloader)
                local_idx = 0  # Index within current DataLoader iteration

                for item in tqdm(
                    dataloader_iter,
                    desc=f"Dumping protein files (rank {self.rank})",
                    initial=processed_count,
                    total=len(protein_files),
                ):
                    protein_id = item["protein_id"]
                    database = item["database"]
                    file_path = item["file_path"]

                    current_file_idx = processed_count
                    processed_count += 1
                    local_idx += 1

                    # Periodic memory cleanup
                    if processed_count % CLEANUP_INTERVAL == 0:
                        gc.collect()
                        if not safe_cuda_empty_cache():
                            # CUDA context corrupted, need to abort
                            logger.error("CUDA context corrupted during periodic cleanup. Aborting.")
                            raise RuntimeError("CUDA context corrupted, cannot continue processing")
                        logger.debug(f"Periodic cleanup at {processed_count} proteins")

                    # Periodic memory monitoring
                    if processed_count % MEMORY_LOG_INTERVAL == 0:
                        log_memory_usage(f"Progress: {processed_count} proteins")

                    # Handle skip from dataset (blacklisted files)
                    # Note: already_exists check is done before DataLoader creation
                    if item["status"] == "skip":
                        reason = item.get("reason", "blacklisted")
                        stats["blacklisted_skipped"] += 1
                        stats["skipped"] += 1
                        result = {
                            "action": "skip",
                            "protein_id": protein_id,
                            "database": database,
                            "reason": reason,
                        }
                        self._log_action(result)
                        continue

                    # Handle preprocessing errors from workers
                    if item["status"] == "error":
                        stats["preprocess_errors"] += 1
                        stats["errors"] += 1

                        # Record failure for blacklist tracking
                        CIFDataset.record_failure(file_path, protein_id)

                        error_detail = {
                            "action": "error",
                            "protein_id": protein_id,
                            "database": database,
                            "file_path": file_path,
                            "error": item["error"],
                            "error_type": item["error_type"],
                            "traceback": item.get("traceback", ""),
                            "stage": "preprocessing",
                        }
                        self._log_action(error_detail)
                        logger.error(f"Preprocessing error on {protein_id}: {item['error_type']}: {item['error']}")
                        # Limit error details list size
                        if len(details) < MAX_ERROR_DETAILS:
                            details.append(error_detail)
                        elif not error_details_truncated:
                            logger.warning(f"Error details list truncated at {MAX_ERROR_DETAILS} entries")
                            error_details_truncated = True
                        continue

                    # Run GPU inference on preprocessed features
                    result = self._run_gpu_inference_and_save(
                        features_dict=item["features_dict"],
                        atom_array=item["atom_array"],
                        token_array=item["token_array"],
                        protein_id=protein_id,
                        database=database,
                        source_file=file_path,
                        force=force,
                        asym_id_to_auth_asym_id=item.get("asym_id_to_auth_asym_id"),
                    )

                    if result["action"] == "dump":
                        stats["dumped"] += 1
                    elif result["action"] == "skip":
                        stats["skipped"] += 1
                    else:
                        stats["errors"] += 1
                        # Limit error details list size
                        if len(details) < MAX_ERROR_DETAILS:
                            details.append(result)
                        elif not error_details_truncated:
                            logger.warning(f"Error details list truncated at {MAX_ERROR_DETAILS} entries")
                            error_details_truncated = True

                # If we completed the loop without exception, we're done
                break

            except Exception as e:
                # Check if this is a DataLoader worker error that we can recover from
                if not enable_crash_recovery or not is_dataloader_worker_error(e):
                    # Not a recoverable error, re-raise
                    logger.error(f"Unrecoverable error: {type(e).__name__}: {e}")
                    raise

                # Worker crash detected
                stats["worker_crashes"] += 1
                dataloader_rebuild_count += 1

                logger.warning(
                    f"DataLoader worker crash detected ({dataloader_rebuild_count}/{MAX_DATALOADER_REBUILDS}): "
                    f"{type(e).__name__}: {e}"
                )

                if dataloader_rebuild_count > MAX_DATALOADER_REBUILDS:
                    logger.error(
                        f"Exceeded maximum DataLoader rebuilds ({MAX_DATALOADER_REBUILDS}). "
                        "Too many worker crashes, aborting."
                    )
                    raise RuntimeError(
                        f"Too many DataLoader worker crashes ({dataloader_rebuild_count})"
                    ) from e

                # Identify the file that caused the crash
                # With num_workers<=1 and prefetch disabled, we can precisely identify
                # the file that caused the crash
                if len(remaining_files) > 0:
                    # The crash occurred while processing, so the culprit is at local_idx
                    # in the current remaining_files list
                    crash_file_idx = min(local_idx, len(remaining_files) - 1)
                    crash_file = remaining_files[crash_file_idx]
                    crash_protein_id = os.path.basename(crash_file).split(".")[0].upper()

                    logger.warning(f"Suspected crash-causing file: {crash_file} (protein_id={crash_protein_id})")

                    # Record failure and potentially blacklist
                    was_blacklisted = CIFDataset.record_failure(crash_file, crash_protein_id)

                    # Record as error
                    stats["errors"] += 1
                    error_detail = {
                        "action": "error",
                        "protein_id": crash_protein_id,
                        "file_path": crash_file,
                        "error": str(e),
                        "error_type": "WorkerCrash",
                        "stage": "worker_crash",
                        "was_blacklisted": was_blacklisted,
                    }
                    self._log_action(error_detail)
                    if len(details) < MAX_ERROR_DETAILS:
                        details.append(error_detail)

                # Clean up memory before rebuilding
                gc.collect()
                safe_cuda_empty_cache()

                # Delete old dataloader to release resources
                del dataloader

                # Calculate remaining files (skip already processed ones)
                # local_idx items were processed successfully before the crash
                # Plus 1 to skip the crash-causing file
                skip_count = local_idx + 1
                remaining_files = remaining_files[skip_count:]

                if len(remaining_files) == 0:
                    logger.info("No more files to process after crash recovery")
                    break

                logger.info(
                    f"Rebuilding DataLoader with {len(remaining_files)} remaining files "
                    f"(skipped {skip_count} files)"
                )

                # Rebuild DataLoader with remaining files
                dataloader = self._create_dataloader(remaining_files, num_workers, force)

        # Final memory log
        log_memory_usage(f"Completed: {processed_count} proteins")

        # Add blacklist statistics
        blacklist_stats = CIFDataset.get_blacklist_stats()
        stats["details"] = details
        stats["error_details_truncated"] = error_details_truncated
        stats["blacklist_stats"] = blacklist_stats

        return stats


# ============================================================================
# Statistics
# ============================================================================

def compute_full_stats(output_dir: str, model_name: str) -> dict:
    """
    Compute statistics for full embeddings
    """
    model_dir = os.path.join(output_dir, model_name)
    if not os.path.isdir(model_dir):
        logger.warning(f"Model directory not found: {model_dir}")
        return {}

    stats = {
        "total_proteins": 0,
        "with_a_token": 0,
        "without_a_token": 0,
        "by_database": {},
        "total_size_bytes": 0,
    }

    pattern = os.path.join(model_dir, "**", "*.safetensors")
    files = glob(pattern, recursive=True)

    for fpath in tqdm(files, desc="Scanning files"):
        try:
            rel_path = os.path.relpath(fpath, model_dir)
            parts = rel_path.split(os.sep)
            db_name = parts[0] if len(parts) >= 2 else "unknown"

            file_size = os.path.getsize(fpath)
            has_atoken = has_a_token(fpath)

            stats["total_proteins"] += 1
            stats["total_size_bytes"] += file_size
            stats["by_database"][db_name] = stats["by_database"].get(db_name, 0) + 1

            if has_atoken:
                stats["with_a_token"] += 1
            else:
                stats["without_a_token"] += 1

        except Exception as e:
            logger.warning(f"Failed to read {fpath}: {e}")

    return stats


def print_full_stats(stats: dict):
    """Print full statistics"""
    logger.info("=" * 50)
    logger.info("Full Embedding Statistics")
    logger.info("=" * 50)
    logger.info(f"Total proteins: {stats.get('total_proteins', 0)}")
    logger.info(f"With a_token: {stats.get('with_a_token', 0)}")
    logger.info(f"Without a_token: {stats.get('without_a_token', 0)}")
    logger.info(f"Total size: {stats.get('total_size_bytes', 0) / 1e9:.2f} GB")
    logger.info("")
    logger.info("By database:")
    for db, count in stats.get("by_database", {}).items():
        logger.info(f"  {db}: {count}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump Protenix full embeddings (s, esm_embedding, a_token) to safetensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Directory containing protein structure files (.cif, .cif.gz, .pdb, .pdb.gz)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/feat_dump",
        help="Output directory for embeddings",
    )
    parser.add_argument(
        "--protenix_path",
        type=str,
        default="pretrained/protenix_mini_ism_v0.5.0",
        help="Path to pretrained Protenix encoder",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="Noise level for diffusion (default: 0.0, no noise)",
    )
    parser.add_argument(
        "--dump-intermediate-layers",
        type=int,
        nargs="+",
        default=None,
        metavar="LAYER_IDX",
        dest="dump_intermediate_layers",
        help=(
            "Extract and save intermediate DiffusionTransformerBlock outputs alongside a_token. "
            "Accepts one or more block indices (0-23, where 0=first, 11=middle, 23=last before LN). "
            "Outputs are raw tensor values without LayerNorm. "
            "Output directory will have suffix '_layers{i}_{j}...' appended. "
            "Example: --dump-intermediate-layers 0 5 11 17"
        ),
    )
    parser.add_argument(
        "--ref_pos_augment",
        action="store_true",
        help="Enable ref_pos random augmentation (default: disabled for determinism)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="GPU rank for multi-GPU processing",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of jobs for data sharding",
    )
    parser.add_argument(
        "--base_rank",
        type=int,
        default=None,
        help="Base rank offset for torchrun mode",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp32"],
        help="Output dtype (default: bf16)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing embeddings",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics of existing embeddings (no dumping)",
    )
    parser.add_argument(
        "--triangle_by_torch",
        action="store_true",
        help="Use PyTorch for triangle operations (for older GPUs or WSL2)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="[Debug] Only process the first N proteins",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of DataLoader workers. "
             "When --precomputed_dir is given, higher values (2-8) are allowed for parallel pkl.gz loading. "
             "Without precomputed_dir, only 0 or 1 is allowed for precise crash tracking. "
             "0 = sequential processing in main process, "
             "1 = single worker for crash recovery (default).",
    )
    parser.add_argument(
        "--no-crash-recovery",
        action="store_true",
        dest="no_crash_recovery",
        help="Disable automatic crash recovery for DataLoader worker crashes (default: enabled)",
    )
    parser.add_argument(
        "--no-bioassembly",
        action="store_true",
        dest="no_bioassembly",
        help="Disable bioassembly processing; use asymmetric unit instead (default: use bioassembly)",
    )
    parser.add_argument(
        "--protein-only",
        action="store_true",
        dest="protein_only",
        help="Filter to protein-only tokens, excluding ligands, DNA/RNA, ions (default: keep all tokens)",
    )
    parser.add_argument(
        "--nan_ids",
        type=str,
        default=None,
        help="JSON file containing protein IDs to EXCLUDE from processing. "
             "Proteins with IDs in this file will be skipped (e.g., proteins that always produce NaN). "
             "Protein ID is extracted from filename as: basename.split('.')[0].upper()",
    )
    parser.add_argument(
        "--include_ids",
        type=str,
        default=None,
        help="JSON file containing protein IDs to INCLUDE in processing. "
             "Only proteins with IDs in this file will be processed (whitelist mode). "
             "Protein ID is extracted from filename as: basename.split('.')[0].upper(). "
             "Note: --nan_ids is applied AFTER --include_ids if both are specified.",
    )
    parser.add_argument(
        "--precomputed_dir",
        type=str,
        default=None,
        help="Root directory containing precomputed processor outputs. "
             "The full path is auto-constructed as: {precomputed_dir}/{model_name}/{suffix}/ "
             "where suffix is determined by use_bioassembly and protein_only settings. "
             "Falls back to process_from_cif if precomputed file not found.",
    )
    parser.add_argument(
        "--max-esm-seq-length",
        type=int,
        default=None,
        dest="max_esm_seq_length",
        help="Maximum allowed ESM sequence length. Samples with esm_input_ids length "
             "exceeding this threshold will be skipped. If not specified, no filtering is applied.",
    )
    parser.add_argument(
        "--max-chain-count",
        type=int,
        default=None,
        dest="max_chain_count",
        help="Maximum allowed number of chains (based on asym_id_to_auth_asym_id length). "
             "Samples with chain count exceeding this threshold will be skipped. "
             "If not specified, no upper limit filtering is applied.",
    )
    parser.add_argument(
        "--min-chain-count",
        type=int,
        default=None,
        dest="min_chain_count",
        help="Minimum required number of chains (based on asym_id_to_auth_asym_id length). "
             "Samples with chain count below this threshold will be skipped. "
             "If not specified, no lower limit filtering is applied.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )

    # Calculate actual rank for torchrun mode
    if args.base_rank is not None:
        torchrun_rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = args.base_rank + torchrun_rank
        torch.cuda.set_device(local_rank)
        logger.info(
            f"Torchrun mode: base_rank={args.base_rank}, RANK={torchrun_rank}, "
            f"LOCAL_RANK={local_rank}, actual rank={rank}"
        )
    else:
        rank = args.rank

    args.rank = rank

    model_name = get_model_name(args.protenix_path)
    use_bioassembly = not args.no_bioassembly
    protein_only = args.protein_only
    logger.info(f"Model name: {model_name}")
    logger.info(f"Output directory: {args.output}/{model_name}/")
    logger.info(f"Settings: sigma={args.sigma}, ref_pos_augment={args.ref_pos_augment}, use_bioassembly={use_bioassembly}, protein_only={protein_only}")

    # Auto-construct full precomputed path: {precomputed_dir}/{model_name}/{suffix}/
    # This matches the directory structure created by precompute_protenix_features.py
    precomputed_dir = args.precomputed_dir
    if precomputed_dir:
        suffix = get_precomputed_suffix(use_bioassembly, protein_only)
        precomputed_dir = os.path.join(precomputed_dir, model_name, suffix)
        logger.info(f"Precomputed directory: {precomputed_dir}")

    # Stats mode
    if args.stats:
        stats = compute_full_stats(args.output, model_name)
        print_full_stats(stats)
        return

    # Validate input directory
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)

    # Create dumper
    dumper = FullEmbeddingDumper(
        output_dir=args.output,
        protenix_path=args.protenix_path,
        device="cuda",
        dtype=args.dtype,
        rank=args.rank,
        world_size=args.world_size,
        triangle_by_torch=args.triangle_by_torch if args.triangle_by_torch else None,
        ref_pos_augment=args.ref_pos_augment,
        sigma=args.sigma,
        use_bioassembly=use_bioassembly,
        protein_only=protein_only,
        precomputed_dir=precomputed_dir,
        max_esm_seq_length=args.max_esm_seq_length,
        max_chain_count=args.max_chain_count,
        min_chain_count=args.min_chain_count,
        intermediate_layers=args.dump_intermediate_layers,
    )

    # Collect protein structure files (CIF and PDB formats)
    protein_files = []
    for ext in ["*.cif", "*.cif.gz", "*.pdb", "*.pdb.gz"]:
        protein_files.extend(glob(os.path.join(args.input_dir, ext)))
        protein_files.extend(glob(os.path.join(args.input_dir, "**", ext), recursive=True))
    protein_files = sorted(set(protein_files))

    if len(protein_files) == 0:
        logger.error(f"No protein structure files found in {args.input_dir}")
        sys.exit(1)

    # Filter by include_ids if provided (whitelist mode, applied first)
    # Only proteins with IDs in this file will be processed
    if args.include_ids is not None:
        with open(args.include_ids, "r") as f:
            include_ids = set(json.load(f))
        logger.info(f"Loaded {len(include_ids)} IDs to include from {args.include_ids}")

        # Filter protein_files to only those with IDs in include_ids
        # Protein ID extraction: basename.split('.')[0].upper()
        filtered_files = []
        excluded_count = 0
        for file_path in protein_files:
            protein_id = os.path.basename(file_path).split(".")[0].upper()
            if protein_id in include_ids:
                filtered_files.append(file_path)
            else:
                excluded_count += 1

        logger.info(f"Filtered by include_ids: {len(protein_files)} -> {len(filtered_files)} protein files ({excluded_count} excluded)")
        protein_files = filtered_files

    # Filter out nan_ids if provided (before sharding and skip check)
    # These IDs are excluded from processing (e.g., proteins that always produce NaN)
    if args.nan_ids is not None:
        with open(args.nan_ids, "r") as f:
            nan_ids = set(json.load(f))
        logger.info(f"Loaded {len(nan_ids)} IDs to exclude from {args.nan_ids}")

        # Filter out protein_files with IDs in nan_ids
        # Protein ID extraction: basename.split('.')[0].upper()
        filtered_files = []
        excluded_count = 0
        for file_path in protein_files:
            protein_id = os.path.basename(file_path).split(".")[0].upper()
            if protein_id not in nan_ids:
                filtered_files.append(file_path)
            else:
                excluded_count += 1

        logger.info(f"Excluded {excluded_count} files by nan_ids: {len(protein_files)} -> {len(filtered_files)} protein files")
        protein_files = filtered_files

    # Filter by precomputed_dir: only keep files that have precomputed features
    # This avoids falling back to slow CIF processing for files without precomputed data
    if precomputed_dir is not None:
        if not os.path.isdir(precomputed_dir):
            raise FileNotFoundError(f"Precomputed directory not found: {precomputed_dir}")
        # Collect all precomputed IDs from .pkl.gz files
        precomputed_files = glob(os.path.join(precomputed_dir, "*.pkl.gz"))
        precomputed_ids = set()
        for pf in precomputed_files:
            # Extract protein_id: basename without .pkl.gz extension
            pid = os.path.basename(pf).removesuffix(".pkl.gz").upper()
            precomputed_ids.add(pid)
        logger.info(f"Found {len(precomputed_ids)} precomputed files in {precomputed_dir}")

        # Filter protein_files to only those with precomputed features
        filtered_files = []
        excluded_count = 0
        for file_path in protein_files:
            protein_id = os.path.basename(file_path).split(".")[0].upper()
            if protein_id in precomputed_ids:
                filtered_files.append(file_path)
            else:
                excluded_count += 1

        logger.info(f"Filtered by precomputed: {len(protein_files)} -> {len(filtered_files)} protein files ({excluded_count} excluded)")
        protein_files = filtered_files

    if args.limit:
        protein_files = protein_files[:args.limit]
        logger.info(f"[Debug] Limited to first {args.limit} protein files")

    logger.info(f"Found {len(protein_files)} protein structure files in {args.input_dir}")
    logger.info(f"Starting dump (rank {args.rank}/{args.world_size})...")

    stats = dumper.dump_protein_files_parallel(
        protein_files=protein_files,
        num_workers=args.num_workers,
        force=args.force,
        enable_crash_recovery=not args.no_crash_recovery,
    )

    logger.info("=" * 50)
    logger.info(f"Full dump completed (rank {args.rank})")
    logger.info(f"  Dumped: {stats['dumped']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info(f"  Errors: {stats['errors']}")
    if stats.get("preprocess_errors", 0) > 0:
        logger.info(f"  Preprocess errors: {stats['preprocess_errors']}")
    if stats.get("worker_crashes", 0) > 0:
        logger.info(f"  Worker crashes: {stats['worker_crashes']}")
    if stats.get("blacklisted_skipped", 0) > 0:
        logger.info(f"  Blacklisted files skipped: {stats['blacklisted_skipped']}")

    # Print blacklist information if any files were blacklisted
    blacklist_stats = stats.get("blacklist_stats", {})
    if blacklist_stats.get("blacklist_count", 0) > 0:
        logger.warning("=" * 50)
        logger.warning("Blacklisted files (failed >= 3 times):")
        for filepath in blacklist_stats.get("blacklisted_files", [])[:20]:
            logger.warning(f"  {filepath}")
        if blacklist_stats["blacklist_count"] > 20:
            logger.warning(f"  ... and {blacklist_stats['blacklist_count'] - 20} more")

    if stats["errors"] > 0:
        logger.warning("Errors occurred:")
        for detail in stats["details"][:10]:
            logger.warning(f"  {detail.get('protein_id', 'unknown')}: {detail.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
