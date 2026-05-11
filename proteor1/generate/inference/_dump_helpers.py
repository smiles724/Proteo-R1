#!/usr/bin/env python3
"""
Protenix Embedding Dump Script

Precompute Protenix Encoder outputs and save them as safetensors.

Key features:
1. Automatically skips existing embeddings (default behavior; no --skip_existing flag).
2. Organizes outputs as {model_name}/{database}/{protein_id}.safetensors.
3. Stores n_token so downstream bucketing is cheap (lazy-read from the safetensors header).
4. Supports Multi-GPU parallelism.
5. --check mode validates existing files (corrupt detection, n_token vs. tensor length consistency).
6. --limit is a debug option that processes only the first N entries.

Usage:
    # Single GPU
    python -m proteor1.generate.inference._dump_helpers \
        --input data/proteins.jsonl \
        --output embeddings/ \
        --protenix_path pretrained/protenix_mini_ism_v0.5.0

    # Debug mode: only process the first 10 entries
    python -m proteor1.generate.inference._dump_helpers \
        --input data/proteins.jsonl \
        --output embeddings/ \
        --protenix_path pretrained/protenix_mini_ism_v0.5.0 \
        --limit 10

    # Multi-GPU option A: launch several processes manually
    for RANK in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$RANK python -m proteor1.generate.inference._dump_helpers \
            --input data/proteins.jsonl \
            --output embeddings/ \
            --protenix_path pretrained/protenix_mini_ism_v0.5.0 \
            --rank $RANK --world_size 4 &
    done

    # Multi-GPU option B: use torchrun (recommended)
    # Run 4 jobs (rank 0-3) on a single 4-GPU node
    torchrun --nproc_per_node=4 \
        -m proteor1.generate.inference._dump_helpers \
        --input data/proteins.jsonl \
        --output embeddings/ \
        --protenix_path pretrained/protenix_mini_ism_v0.5.0 \
        --base_rank 0 --world_size 4

    # Distributed: 16 jobs across 4 nodes, 4 GPUs each
    # Node 0: --base_rank 0  (runs rank 0-3)
    # Node 1: --base_rank 4  (runs rank 4-7)
    # Node 2: --base_rank 8  (runs rank 8-11)
    # Node 3: --base_rank 12 (runs rank 12-15)
    torchrun --nproc_per_node=4 \
        -m proteor1.generate.inference._dump_helpers \
        --input data/proteins.jsonl \
        --output embeddings/ \
        --protenix_path pretrained/protenix_mini_ism_v0.5.0 \
        --base_rank 12 --world_size 16

    # Show statistics
    python -m proteor1.generate.inference._dump_helpers \
        --output embeddings/ \
        --protenix_path pretrained/protenix_mini_ism_v0.5.0 \
        --stats

    # Check missing and corrupt files (validates file contents and n_token consistency)
    python -m proteor1.generate.inference._dump_helpers \
        --input data/proteins.jsonl \
        --output embeddings/ \
        --protenix_path pretrained/protenix_mini_ism_v0.5.0 \
        --check missing.json
"""
from typing import Optional

import argparse
import json
import os
import re
import sys
from datetime import datetime
from glob import glob

import torch
from loguru import logger
from safetensors.torch import save_file, safe_open
from tqdm import tqdm

from proteor1.understand.data_collator import move_protenix_features_to_device


# ============================================================================
# Core Utility Functions
# ============================================================================

def get_model_name(protenix_path: str) -> str:
    """
    Extract model_name from protenix_path.

    Examples:
        "pretrained/protenix_mini_ism_v0.5.0" -> "protenix_mini_ism_v0.5.0"
        "/path/to/protenix_mini_ism_v0.5.0/" -> "protenix_mini_ism_v0.5.0"
    """
    path = protenix_path.rstrip("/\\")
    return os.path.basename(path)


def parse_protein_source(protein_json: dict) -> tuple[str, str]:
    """
    Parse the protein source.

    Priority:
    1. _metadata.database or _metadata.source set manually.
    2. Explicit fields in the data (uniprot_id, pdb_id, alphafold_id).
    3. Auto-detection from the name field.

    Returns:
        (database, protein_id)
    """
    metadata = protein_json.get("_metadata", {})
    name = protein_json.get("name", "unknown")

    # Prefer the manually specified database or source.
    if "database" in metadata:
        return metadata["database"], name

    if "source" in metadata:
        return metadata["source"], name

    # Next, prefer explicit fields in the data.
    if "uniprot_id" in metadata:
        return "uniprot", metadata["uniprot_id"]

    if "pdb_id" in metadata:
        return "pdb", metadata["pdb_id"].lower()

    if "alphafold_id" in metadata:
        return "alphafold", metadata["alphafold_id"]

    # Fall back to auto-detecting from the name.
    # PDB: 4 characters, starts with a digit.
    if re.match(r'^[0-9][a-zA-Z0-9]{3}$', name):
        return "pdb", name.lower()

    # AlphaFold: AF-XXXX-F1 format.
    if name.startswith("AF-"):
        match = re.match(r'^AF-([A-Z0-9]+)-F\d+$', name)
        if match:
            return "alphafold", match.group(1)
        return "alphafold", name.replace("AF-", "").split("-")[0]

    # Default: custom.
    return "custom", name


def get_embedding_path(
    model_name: str,
    database: str,
    protein_id: str,
    base_dir: str
) -> str:
    """
    Build the full embedding-file path.

    Returns:
        {base_dir}/{model_name}/{database}/{protein_id}.safetensors
    """
    return os.path.join(base_dir, model_name, database, f"{protein_id}.safetensors")


def embedding_exists(
    model_name: str,
    database: str,
    protein_id: str,
    base_dir: str
) -> bool:
    """Check whether the embedding file already exists."""
    path = get_embedding_path(model_name, database, protein_id, base_dir)
    return os.path.exists(path)


# ============================================================================
# Save / Load Functions
# ============================================================================

def save_embedding(
    path: str,
    s: torch.Tensor,
    esm_embedding: torch.Tensor,
    n_token: int,
    metadata: dict,
    dtype: torch.dtype = torch.bfloat16,
    residue_index: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
):
    """
    Save a single protein's embedding.

    File contents:
    - tensors: s, esm_embedding, n_token, residue_index, asym_id
    - metadata: protein_id, database, n_token, model_name, ...

    residue_index and asym_id are used by the Protenix Position Embedding; storing them
    at dump time means:
    1. Training does not need to recompute SampleDictToFeatures.
    2. Position info is guaranteed to come from the same compute as the embedding.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    tensors = {
        "s": s.detach().cpu().to(dtype),
        "esm_embedding": esm_embedding.detach().cpu().to(dtype),
        "n_token": torch.tensor([n_token], dtype=torch.int64),
    }

    # Save position info when provided.
    if residue_index is not None:
        tensors["residue_index"] = residue_index.detach().cpu().long()
    if asym_id is not None:
        tensors["asym_id"] = asym_id.detach().cpu().long()

    metadata_str = {k: str(v) for k, v in metadata.items()}
    metadata_str["dump_timestamp"] = datetime.now().isoformat()
    metadata_str["dtype"] = str(dtype).split(".")[-1]
    # Flag whether the file contains position info.
    metadata_str["has_position_info"] = str(residue_index is not None and asym_id is not None)

    save_file(tensors, path, metadata=metadata_str)


def load_embedding(path: str) -> dict:
    """
    Load a single protein's embedding.

    Returns:
        {
            "s": Tensor [N_token, 384],
            "esm_embedding": Tensor [N_token, 2560],
            "n_token": int,
            "metadata": dict (from the safetensors header),
            "residue_index": Tensor [N_token] (optional, only in newer dumps),
            "asym_id": Tensor [N_token] (optional, only in newer dumps),
        }
    """
    with safe_open(path, framework="pt") as f:
        s = f.get_tensor("s")
        esm_embedding = f.get_tensor("esm_embedding")
        n_token = f.get_tensor("n_token").item()
        metadata = f.metadata()

        # Load position info when present (older dumps may not have it).
        keys = f.keys()
        residue_index = f.get_tensor("residue_index") if "residue_index" in keys else None
        asym_id = f.get_tensor("asym_id") if "asym_id" in keys else None

    result = {
        "s": s,
        "esm_embedding": esm_embedding,
        "n_token": n_token,
        "metadata": metadata,
    }

    if residue_index is not None:
        result["residue_index"] = residue_index
    if asym_id is not None:
        result["asym_id"] = asym_id

    return result


def load_n_token_only(path: str) -> int:
    """
    Load n_token only (read from the header, do not load tensors).

    Useful for bucketing by sequence length without paying the tensor I/O cost.
    """
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
        return int(metadata.get("n_token", 0))


def get_precomputed_suffix(use_bioassembly: bool, protein_only: bool) -> str:
    """Return the directory suffix used for cached Protenix processor outputs."""

    parts = []
    if not use_bioassembly:
        parts.append("asym")
    if protein_only:
        parts.append("prot")
    return "_".join(parts) if parts else "default"


def get_precomputed_path(
    precomputed_dir: str,
    protein_id: str,
    use_bioassembly: bool,
    protein_only: bool,
) -> str:
    """Return the cached processor-output path for a protein id."""

    suffix = get_precomputed_suffix(use_bioassembly, protein_only)
    return os.path.join(precomputed_dir, suffix, f"{protein_id}.pkl.gz")


def load_processor_output(path: str):
    """Load a cached ProtenixProcessorOutput from a gzip pickle file."""

    from protenix.utils.file_io import load_gzip_pickle
    from proteor1.understand.protenix_encoder.processing_protenix_encoder import ProtenixProcessorOutput

    data = load_gzip_pickle(path)
    return ProtenixProcessorOutput(
        input_feature_dict=data["input_feature_dict"],
        atom_array=data["atom_array"],
        token_array=data["token_array"],
        metadata=data.get("metadata", {}),
    )


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_input_file(input_path: str) -> list[dict]:
    """Load a JSON or JSONL file."""
    proteins = []

    if input_path.endswith(".jsonl"):
        with open(input_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    proteins.append(json.loads(line))
    elif input_path.endswith(".json"):
        with open(input_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                proteins = data
            else:
                proteins = [data]
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    return proteins


def load_multiple_files(input_paths: list[str]) -> list[dict]:
    """Load multiple input files."""
    all_proteins = []
    for path in input_paths:
        logger.info(f"Loading {path}...")
        proteins = load_input_file(path)
        logger.info(f"  Loaded {len(proteins)} proteins")
        all_proteins.extend(proteins)
    return all_proteins


# ============================================================================
# EmbeddingDumper Class
# ============================================================================

class EmbeddingDumper:
    """
    Protenix Embedding dumper.

    Key features:
    1. Automatically skips existing embeddings (default behavior).
    2. Organizes outputs as {model_name}/{database}/{id}.
    3. Stores n_token in every file for cheap downstream bucketing.
    4. Multi-GPU support.
    """

    def __init__(
        self,
        output_dir: str,
        protenix_path: str,
        device: str = "cuda",
        dtype: str = "bf16",
        rank: int = 0,
        world_size: int = 1,
        triangle_by_torch: Optional[bool] = None
    ):
        """
        Args:
            output_dir: output directory
            protenix_path: path to the pretrained Protenix model
            device: device ("cuda" or "cpu")
            dtype: dtype ("bf16" or "fp32")
            rank: this process's rank (for Multi-GPU)
            world_size: total number of processes (for Multi-GPU)
            triangle_by_torch: whether to use the PyTorch implementation of triangle ops.
                Useful for: 1) debugging, 2) older GPUs (e.g. V100) without Triton,
                3) environments where cuequivariance Triton kernels fail (e.g. WSL2).
        """
        self.output_dir = output_dir
        self.protenix_path = protenix_path
        self.model_name = get_model_name(protenix_path)
        self.device = device
        self.dtype_str = dtype
        self.dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.rank = rank
        self.world_size = world_size
        self.triangle_by_torch = triangle_by_torch

        self.model_output_dir = os.path.join(output_dir, self.model_name)
        # Name files with datetime + world_size + rank so each run creates a fresh file.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(
            self.model_output_dir,
            f"dump_log_{timestamp}_ws{world_size}_rank{rank}.jsonl"
        )

        self._encoder = None
        self._processor = None

    @property
    def encoder(self):
        if self._encoder is None:
            from proteor1.understand.protenix_encoder import ProtenixEncoder

            logger.info(f"Loading ProtenixEncoder from {self.protenix_path}...")
            self._encoder = ProtenixEncoder.from_pretrained(
                self.protenix_path,
                load_esm=True,
                device=self.device,
                dtype=self.dtype_str,
                triangle_by_torch=self.triangle_by_torch
            )
            self._encoder = self._encoder.to(dtype=self.dtype)
            self._encoder.eval()
        return self._encoder

    @property
    def processor(self):
        if self._processor is None:
            from proteor1.understand.protenix_encoder import ProtenixProcessor

            logger.info(f"Loading ProtenixProcessor from {self.protenix_path}...")
            self._processor = ProtenixProcessor.from_pretrained(
                self.protenix_path,
                init_esm_tokenizer=True,
            )
        return self._processor

    def exists(self, database: str, protein_id: str) -> bool:
        """Check whether the embedding already exists."""
        return embedding_exists(self.model_name, database, protein_id, self.output_dir)

    def _log_action(self, action_data: dict):
        """Append an action record to the log file."""
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        action_data["timestamp"] = datetime.now().isoformat()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(action_data, ensure_ascii=False) + "\n")

    def process_protein(
        self,
        protein_data: dict,
        source_file: str = "",
        force: bool = False,
    ) -> dict:
        """Process a single protein."""
        database, protein_id = parse_protein_source(protein_data)
        path = get_embedding_path(self.model_name, database, protein_id, self.output_dir)

        # Auto-skip when already present.
        if not force and os.path.exists(path):
            result = {
                "action": "skip",
                "protein_id": protein_id,
                "database": database,
                "reason": "already_exists",
            }
            self._log_action(result)
            return result

        try:
            # Data is already in Protenix format; use it directly.
            processor_output = self.processor(protein_data)

            # Move to device up front (only once).
            input_feature_dict_on_device = move_protenix_features_to_device(
                processor_output.input_feature_dict, self.encoder.device, dtype=self.encoder.dtype
            )

            # NaN-detection retry loop: try up to 10 times.
            max_retries = 10
            s, esm_embedding = None, None

            for attempt in range(max_retries):
                with torch.inference_mode(), torch.autocast(device_type=self.device, dtype=self.dtype, enabled=torch.cuda.is_available()):
                    s, esm_embedding = self.encoder(
                        input_feature_dict=input_feature_dict_on_device,
                        atom_array=processor_output.atom_array,
                        token_array=processor_output.token_array,
                    )

                # Check for NaN.
                has_nan = torch.isnan(s).any().item() or torch.isnan(esm_embedding).any().item()

                if not has_nan:
                    break  # no NaN, exit the loop

                # NaN encountered: clear cache and retry.
                import gc
                logger.warning(f"NaN detected in {protein_id} (attempt {attempt + 1}/{max_retries}), retrying...")
                gc.collect()
                torch.cuda.empty_cache()

            # All 10 attempts failed: log and skip.
            if torch.isnan(s).any().item() or torch.isnan(esm_embedding).any().item():
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

            n_token = s.shape[0]

            # Pull position info from input_feature_dict (same source as the embedding, guarantees consistency).
            residue_index = processor_output.input_feature_dict.get("residue_index")
            asym_id = processor_output.input_feature_dict.get("asym_id")

            save_embedding(
                path=path,
                s=s,
                esm_embedding=esm_embedding,
                n_token=n_token,
                metadata={
                    "protein_id": protein_id,
                    "database": database,
                    "n_token": n_token,
                    "s_dim": s.shape[-1],
                    "esm_dim": esm_embedding.shape[-1],
                    "source_file": source_file,
                    "model_name": self.model_name,
                },
                dtype=self.dtype,
                residue_index=residue_index,
                asym_id=asym_id,
            )

            result = {
                "action": "dump",
                "protein_id": protein_id,
                "database": database,
                "n_token": n_token,
                "path": path,
            }
            self._log_action(result)
            return result

        except torch.cuda.OutOfMemoryError as e:
            # OOM specific handling: clear the CUDA cache so subsequent items can continue.
            import traceback
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            result = {
                "action": "error",
                "protein_id": protein_id,
                "database": database,
                "error": f"CUDA OOM: {e}",
                "traceback": traceback.format_exc(),
            }
            self._log_action(result)
            logger.warning(f"OOM on {protein_id}, cleared CUDA cache and continuing...")
            return result

        except Exception as e:
            import traceback
            # Record more detailed error info to aid debugging.
            error_detail = {
                "action": "error",
                "protein_id": protein_id,
                "database": database,
                "protein_name": protein_data.get("name", "unknown"),
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
            }
            # Append protein-structure info.
            if "sequences" in protein_data:
                seqs = protein_data["sequences"]
                error_detail["n_chains"] = len(seqs)
                # Protenix format: {"proteinChain": {"sequence": "...", "count": 1}}
                def get_seq_len(s):
                    for key in ["proteinChain", "dnaSequence", "rnaSequence"]:
                        if key in s:
                            return len(s[key].get("sequence", ""))
                    return 0
                error_detail["chain_lengths"] = [get_seq_len(s) for s in seqs]
            self._log_action(error_detail)
            logger.error(
                f"Error on {protein_id} ({protein_data.get('name', 'unknown')}): "
                f"{type(e).__name__}: {e}"
            )
            return error_detail

    def dump_proteins(
        self,
        proteins: list[dict],
        source_file: str = "",
        force: bool = False,
    ) -> dict:
        """Dump a list of proteins."""
        # Multi-GPU sharding: interleaved assignment.
        if self.world_size > 1:
            proteins = proteins[self.rank::self.world_size]
            logger.info(f"Rank {self.rank}/{self.world_size}: assigned {len(proteins)} proteins")

        stats = {"dumped": 0, "skipped": 0, "errors": 0}
        details = []

        # Pre-filter to determine which embeddings already exist.
        if not force:
            proteins_to_process = []
            for protein in tqdm(proteins, desc=f"Checking existing (rank {self.rank})"):
                database, protein_id = parse_protein_source(protein)
                if not self.exists(database, protein_id):
                    proteins_to_process.append(protein)
                else:
                    stats["skipped"] += 1
                    self._log_action({
                        "action": "skip",
                        "protein_id": protein_id,
                        "database": database,
                        "reason": "already_exists",
                    })

            logger.info(
                f"Rank {self.rank}: {len(proteins_to_process)} to process, "
                f"{stats['skipped']} already exist (skipped)"
            )
        else:
            proteins_to_process = proteins
            logger.info(f"Rank {self.rank}: {len(proteins_to_process)} to process (force mode)")

        # Use the actually-to-process count to drive tqdm.
        for protein in tqdm(proteins_to_process, desc=f"Dumping (rank {self.rank})"):
            result = self.process_protein(
                protein_data=protein,
                source_file=source_file,
                force=force,
            )

            if result["action"] == "dump":
                stats["dumped"] += 1
            elif result["action"] == "skip":
                # Should not happen in force mode, but kept defensively.
                stats["skipped"] += 1
            else:
                stats["errors"] += 1
                details.append(result)

        stats["details"] = details
        return stats

    def dump_files(
        self,
        input_paths: list[str],
        force: bool = False,
        limit: Optional[int] = None,
        nan_ids_file: Optional[str] = None,
    ) -> dict:
        """Bulk-dump multiple files.

        Args:
            input_paths: list of input file paths
            force: whether to overwrite existing embeddings
            limit: [Debug] only process the first N entries
            nan_ids_file: JSON file containing NaN IDs to re-dump
        """
        all_proteins = load_multiple_files(input_paths)

        # Dedupe by name, keeping the first occurrence.
        seen_names = set()
        unique_proteins = []
        for p in all_proteins:
            name = p.get("name")
            if name not in seen_names:
                seen_names.add(name)
                unique_proteins.append(p)
        if len(unique_proteins) < len(all_proteins):
            logger.info(f"Deduplicated by name: {len(all_proteins)} -> {len(unique_proteins)} proteins")
        all_proteins = unique_proteins

        # If nan_ids_file is provided, restrict to those IDs.
        if nan_ids_file is not None:
            with open(nan_ids_file, "r") as f:
                nan_ids = set(json.load(f))
            logger.info(f"Loaded {len(nan_ids)} NaN IDs from {nan_ids_file}")

            # Filter to keep only proteins in nan_ids.
            filtered_proteins = [p for p in all_proteins if p.get("name") in nan_ids]
            logger.info(f"Filtered by NaN IDs: {len(all_proteins)} -> {len(filtered_proteins)} proteins")
            all_proteins = filtered_proteins

            # nan_ids mode forces overwriting.
            force = True

        if limit is not None:
            all_proteins = all_proteins[:limit]
            logger.info(f"[Debug] Limited to first {limit} proteins")
        return self.dump_proteins(all_proteins, source_file=",".join(input_paths), force=force)


# ============================================================================
# Statistics (on-demand, no manifest needed)
# ============================================================================

def compute_stats(output_dir: str, model_name: str) -> dict:
    """
    Compute statistics on demand (no manifest required).

    Walks the directory with glob and reads n_token from each safetensors header.
    """
    model_dir = os.path.join(output_dir, model_name)
    if not os.path.isdir(model_dir):
        logger.warning(f"Model directory not found: {model_dir}")
        return {}

    stats = {
        "total_proteins": 0,
        "by_database": {},
        "n_token_distribution": {
            "0-128": 0,
            "128-256": 0,
            "256-512": 0,
            "512-1024": 0,
            "1024-2048": 0,
            "2048+": 0,
        },
        "total_size_bytes": 0,
    }

    # Glob every safetensors file.
    pattern = os.path.join(model_dir, "**", "*.safetensors")
    files = glob(pattern, recursive=True)

    for fpath in tqdm(files, desc="Scanning files"):
        try:
            # Parse the path to obtain the database name.
            rel_path = os.path.relpath(fpath, model_dir)
            parts = rel_path.split(os.sep)
            if len(parts) >= 2:
                db_name = parts[0]
            else:
                db_name = "unknown"

            # Read n_token from the header.
            n_token = load_n_token_only(fpath)
            file_size = os.path.getsize(fpath)

            stats["total_proteins"] += 1
            stats["total_size_bytes"] += file_size
            stats["by_database"][db_name] = stats["by_database"].get(db_name, 0) + 1

            # Update the bucketed distribution.
            if n_token <= 128:
                stats["n_token_distribution"]["0-128"] += 1
            elif n_token <= 256:
                stats["n_token_distribution"]["128-256"] += 1
            elif n_token <= 512:
                stats["n_token_distribution"]["256-512"] += 1
            elif n_token <= 1024:
                stats["n_token_distribution"]["512-1024"] += 1
            elif n_token <= 2048:
                stats["n_token_distribution"]["1024-2048"] += 1
            else:
                stats["n_token_distribution"]["2048+"] += 1

        except Exception as e:
            logger.warning(f"Failed to read {fpath}: {e}")

    return stats


def print_stats(stats: dict):
    """Print statistics."""
    logger.info("=" * 50)
    logger.info("Embedding Statistics")
    logger.info("=" * 50)
    logger.info(f"Total proteins: {stats.get('total_proteins', 0)}")
    logger.info(f"Total size: {stats.get('total_size_bytes', 0) / 1e9:.2f} GB")
    logger.info("")
    logger.info("By database:")
    for db, count in stats.get("by_database", {}).items():
        logger.info(f"  {db}: {count}")
    logger.info("")
    logger.info("N_token distribution:")
    for bucket, count in stats.get("n_token_distribution", {}).items():
        logger.info(f"  {bucket}: {count}")


# ============================================================================
# Check Missing Proteins
# ============================================================================

def load_error_logs(output_dir: str, model_name: str) -> dict[str, dict]:
    """
    Load every error record from the log files.

    Returns:
        {protein_id: {database, error, traceback, timestamp, ...}}
    """
    model_dir = os.path.join(output_dir, model_name)
    log_pattern = os.path.join(model_dir, "dump_log_*.jsonl")
    log_files = glob(log_pattern)

    errors = {}
    for log_file in log_files:
        try:
            with open(log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("action") == "error":
                        pid = record.get("protein_id", "unknown")
                        # Keep the most recent error record.
                        if pid not in errors or record.get("timestamp", "") > errors[pid].get("timestamp", ""):
                            errors[pid] = record
        except Exception as e:
            logger.warning(f"Failed to read log file {log_file}: {e}")

    return errors


def verify_safetensors_readable(path: str) -> tuple[bool, Optional[str]]:
    """
    Verify that a safetensors file is readable and that n_token matches the tensor lengths.

    Returns:
        (is_valid, error_message)
        - (True, None) when the file is readable and lengths agree
        - (False, error_message) when the file is corrupt, unreadable, or has inconsistent lengths
    """
    try:
        data = load_embedding(path)
        # Verify length consistency.
        n_token = data["n_token"]
        s_len = data["s"].shape[0]
        esm_len = data["esm_embedding"].shape[0]
        if s_len != n_token:
            return False, f"n_token mismatch: n_token={n_token}, s.shape[0]={s_len}"
        if esm_len != n_token:
            return False, f"n_token mismatch: n_token={n_token}, esm_embedding.shape[0]={esm_len}"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def check_missing_proteins(
    input_paths: list[str],
    output_dir: str,
    model_name: str,
    output_file: Optional[str] = None,
) -> dict:
    """
    Check which proteins were not dumped successfully and surface their failure reasons.

    Returns:
        {
            "total": int,
            "success": int,
            "missing": int,
            "missing_with_error": int,
            "missing_no_log": int,
            "corrupted": int,
            "details": [...]
        }
    """
    # Load every protein.
    proteins = load_multiple_files(input_paths)
    logger.info(f"Total proteins in input: {len(proteins)}")

    # Load error logs.
    error_logs = load_error_logs(output_dir, model_name)
    logger.info(f"Found {len(error_logs)} error records in logs")

    # Check each protein.
    missing_details = []
    corrupted_details = []
    success_count = 0

    for protein in tqdm(proteins, desc="Checking proteins"):
        database, protein_id = parse_protein_source(protein)
        embedding_path = get_embedding_path(model_name, database, protein_id, output_dir)

        if os.path.exists(embedding_path):
            # File exists; check whether the content is readable.
            is_valid, error_msg = verify_safetensors_readable(embedding_path)
            if is_valid:
                success_count += 1
            else:
                # File exists but is corrupt.
                corrupted_details.append({
                    "protein_id": protein_id,
                    "database": database,
                    "name": protein.get("name", ""),
                    "path": embedding_path,
                    "error": f"Corrupted file: {error_msg}",
                    "is_corrupted": True,
                })
        else:
            # Look up the failure reason.
            error_info = error_logs.get(protein_id)
            detail = {
                "protein_id": protein_id,
                "database": database,
                "name": protein.get("name", ""),
            }
            if error_info:
                detail["error"] = error_info.get("error", "Unknown error")
                detail["traceback"] = error_info.get("traceback", "")
                detail["timestamp"] = error_info.get("timestamp", "")
                detail["has_error_log"] = True
            else:
                detail["error"] = "No error log found (possibly not processed yet)"
                detail["has_error_log"] = False

            missing_details.append(detail)

    # Summary counts.
    missing_with_error = sum(1 for d in missing_details if d.get("has_error_log"))
    missing_no_log = len(missing_details) - missing_with_error

    result = {
        "total": len(proteins),
        "success": success_count,
        "missing": len(missing_details),
        "corrupted": len(corrupted_details),
        "missing_with_error": missing_with_error,
        "missing_no_log": missing_no_log,
        "details": missing_details + corrupted_details,
    }

    # Save to file.
    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved check result to {output_file}")

    return result


def print_check_result(result: dict, verbose: bool = False):
    """Print the check result."""
    logger.info("=" * 60)
    logger.info("Check Result")
    logger.info("=" * 60)
    logger.info(f"Total proteins in input:    {result['total']}")
    logger.info(f"Successfully dumped:        {result['success']}")
    logger.info(f"Missing:                    {result['missing']}")
    logger.info(f"  - With error log:         {result['missing_with_error']}")
    logger.info(f"  - No error log (pending): {result['missing_no_log']}")
    logger.info(f"Corrupted files:            {result.get('corrupted', 0)}")
    logger.info("")

    if result["missing"] == 0 and result.get("corrupted", 0) == 0:
        logger.info("All proteins have been successfully dumped!")
        return

    # Group by error type.
    error_groups: dict[str, list] = {}
    corrupted_list = []
    no_log_count = 0

    for detail in result["details"]:
        # Handle corrupt files.
        if detail.get("is_corrupted"):
            corrupted_list.append(detail)
            continue

        if not detail.get("has_error_log"):
            no_log_count += 1
            continue

        error = detail.get("error", "Unknown")
        # Simplify the error string for grouping.
        if "CUDA" in error or "OOM" in error:
            key = "CUDA OOM"
        elif "timeout" in error.lower():
            key = "Timeout"
        else:
            # First line, truncated, as the group key.
            key = error.split("\n")[0][:80]

        if key not in error_groups:
            error_groups[key] = []
        error_groups[key].append(detail)

    # Print corrupt files.
    if corrupted_list:
        logger.info("Corrupted files (need to re-dump):")
        for d in corrupted_list[:10]:
            logger.info(f"  - {d['protein_id']}: {d['error']}")
        if len(corrupted_list) > 10:
            logger.info(f"  ... and {len(corrupted_list) - 10} more")
        logger.info("")

    # Print error groups.
    if error_groups:
        logger.info("Errors by type:")
        for error_type, details in sorted(error_groups.items(), key=lambda x: -len(x[1])):
            logger.info(f"  [{len(details)}] {error_type}")
            if verbose:
                for d in details[:5]:  # show at most 5 per error type
                    logger.info(f"       - {d['protein_id']}")
                if len(details) > 5:
                    logger.info(f"       ... and {len(details) - 5} more")

    if no_log_count > 0:
        logger.info("")
        logger.info(f"Proteins with no error log (likely not processed yet): {no_log_count}")
        if verbose:
            pending = [d for d in result["details"] if not d.get("has_error_log") and not d.get("is_corrupted")]
            for d in pending[:10]:
                logger.info(f"  - {d['protein_id']}")
            if len(pending) > 10:
                logger.info(f"  ... and {len(pending) - 10} more")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dump Protenix encoder embeddings to safetensors format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        nargs="+",
        help="Input JSON/JSONL file(s)",
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
        help="Path to pretrained Protenix encoder (basename used as model_name)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="GPU rank for multi-GPU processing (default: 0). "
             "Ignored when using torchrun with --base_rank.",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Total number of jobs for data sharding (default: 1). "
             "This is the logical world size for splitting data, not torchrun's WORLD_SIZE.",
    )
    parser.add_argument(
        "--base_rank",
        type=int,
        default=None,
        help="Base rank offset for torchrun mode. When set, actual rank = base_rank + RANK (from torchrun). "
             "Example: To run jobs 12-15 of a 16-job task on a 4-GPU node, use: "
             "torchrun --nproc_per_node=4 ... --base_rank 12 --world_size 16",
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
        "--check",
        nargs="?",
        const=True,
        default=False,
        metavar="OUTPUT_FILE",
        help="Check which proteins are missing and show error reasons from logs. "
             "Optionally specify a file path to save results as JSON (e.g., --check missing.json)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output (e.g., list individual proteins in --check)",
    )
    parser.add_argument(
        "--triangle_by_torch",
        action="store_true",
        help="Use PyTorch implementation for triangle multiplicative update instead of Triton kernels. "
             "Useful for: 1) debugging, 2) older GPUs like V100 that lack Triton support, "
             "3) environments where cuequivariance Triton kernels fail (e.g., WSL2).",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="[Debug] Only process the first N proteins. Useful for testing.",
    )
    parser.add_argument(
        "--nan_ids",
        type=str,
        default=None,
        help="JSON file containing NaN protein IDs to re-dump. "
             "When provided: 1) only proteins with IDs in this file will be processed, "
             "2) existence check is skipped (force re-dump), "
             "3) NaN detection with retry logic is enabled.",
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

    # Compute the effective rank with torchrun support:
    # - Regular mode: use the --rank flag.
    # - Torchrun mode (--base_rank set): rank = base_rank + RANK.
    #   Note: we use torchrun's per-node RANK (starts at 0), not the global RANK,
    #   because we launch multiple processes on a single node via torchrun, each
    #   handling its own data shard.
    if args.base_rank is not None:
        torchrun_rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        rank = args.base_rank + torchrun_rank
        # Critical: set the CUDA device so each process pins to its own GPU; otherwise
        # multiple processes share one GPU and trigger OOM.
        torch.cuda.set_device(local_rank)
        logger.info(f"Torchrun mode: base_rank={args.base_rank}, RANK={torchrun_rank}, LOCAL_RANK={local_rank}, actual rank={rank}")
    else:
        rank = args.rank

    # Update args.rank so downstream code sees the effective value.
    args.rank = rank

    model_name = get_model_name(args.protenix_path)
    logger.info(f"Model name: {model_name}")
    logger.info(f"Output directory: {args.output}/{model_name}/")

    # Statistics mode.
    if args.stats:
        stats = compute_stats(args.output, model_name)
        print_stats(stats)
        return

    # Validate input.
    if not args.input:
        logger.error("No input files specified. Use --input to specify JSON/JSONL files.")
        sys.exit(1)

    for path in args.input:
        if not os.path.exists(path):
            logger.error(f"Input file not found: {path}")
            sys.exit(1)

    # Check mode: analyze missing proteins and show error reasons.
    if args.check:
        logger.info("Check mode: analyzing missing proteins...")
        # args.check is either True or a file-path string.
        output_file = args.check if isinstance(args.check, str) else None
        result = check_missing_proteins(
            input_paths=args.input,
            output_dir=args.output,
            model_name=model_name,
            output_file=output_file,
        )
        print_check_result(result, verbose=args.verbose)
        return

    # Build the dumper and run.
    dumper = EmbeddingDumper(
        output_dir=args.output,
        protenix_path=args.protenix_path,
        device="cuda",
        dtype=args.dtype,
        rank=args.rank,
        world_size=args.world_size,
        triangle_by_torch=args.triangle_by_torch,
    )

    logger.info(f"Starting dump (rank {args.rank}/{args.world_size})...")
    stats = dumper.dump_files(args.input, force=args.force, limit=args.limit, nan_ids_file=args.nan_ids)

    logger.info("=" * 50)
    logger.info(f"Dump completed (rank {args.rank})")
    logger.info(f"  Dumped: {stats['dumped']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info(f"  Errors: {stats['errors']}")

    if stats["errors"] > 0:
        logger.warning("Errors occurred:")
        for detail in stats["details"][:10]:
            logger.warning(f"  {detail['protein_id']}: {detail['error']}")


if __name__ == "__main__":
    main()
