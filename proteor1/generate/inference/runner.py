#!/usr/bin/env python3
"""
ProteoR1 generate inference script.

This script runs inference using the ProteoR1GenerateModel with text conditioning.
It loads a trained HF checkpoint and generates antibody CDR sequences.

Data flow:
1. YAML files are parsed and Structure NPZ files are auto-generated to out_dir/processed/structures/
2. CDR positions in YAML (marked as 'X') become UNK token (res_type=22)
3. CDR sequences are extracted from chain_infos for text conditioning
4. Model predicts sequences for CDR positions

Supports two execution modes:
1. Single-GPU mode with parallel data loading (DataLoader num_workers)
2. Multi-GPU distributed mode with torchrun (DDP + DistributedSampler)

Usage (Single-GPU, with HF checkpoint):
    python -m proteor1.generate.inference.runner \\
        --data datasets/upstream/test_yaml_dir \\
        --checkpoint thinking-bio-lab/proteor1-generate \\
        --out_dir outputs/demo

Usage (Single-GPU, with upstream structure-design checkpoint as baseline):
    python -m proteor1.generate.inference.runner \\
        --data datasets/upstream/test_yaml_dir \\
        --checkpoint upstream \\
        --boltz-ckpt-path ckpts/upstream/stage_4.ckpt \\
        --out_dir outputs/upstream_baseline

Usage (Multi-GPU with torchrun):
    torchrun --nproc_per_node=4 -m proteor1.generate.inference.runner \\
        --data datasets/upstream/test_yaml_dir \\
        --checkpoint thinking-bio-lab/proteor1-generate \\
        --out_dir outputs/demo
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from proteor1.generate.inference.prediction_dataset import BagelPredictionDataset

# Reuse from upstream reference impl
from proteor1.generate.inference._helpers import (
    BoltzDiffusionParams,
    BoltzProcessedInput,
    PredictionDataset,
    check_inputs,
    cleanup_distributed,
    collate,
    get_local_rank,
    is_main_process,
    process_inputs,
    seed_everything,
    send_to_device,
    setup_distributed,
)
from proteor1.generate import (
    ProteoR1GenerateConfig,
    ProteoR1GenerateModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="proteor1-design",
        description="ProteoR1 diffusion inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/output
    parser.add_argument(
        "--data",
        "--input_dir",
        dest="data",
        type=str,
        required=True,
        help="Path to input YAML file or directory",
    )
    parser.add_argument(
        "--out_dir",
        "--output",
        dest="out_dir",
        type=str,
        default="./predictions",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="thinking-bio-lab/proteor1-generate",
        help="HF Hub repo id or local directory for the ProteoR1 generate checkpoint; pass 'upstream' to load a baseline Boltz1 checkpoint via --boltz-ckpt-path instead",
    )
    parser.add_argument(
        "--boltz-ckpt-path",
        type=str,
        default="ckpts/upstream/stage_4.ckpt",
        help="Path to upstream structure-design checkpoint (only used when --checkpoint=upstream)",
    )

    # MSA options
    parser.add_argument(
        "--msa_dir",
        type=str,
        default=None,
        help="Path to raw MSA CSV files directory. If not provided, defaults to out_dir/msa",
    )
    parser.add_argument(
        "--processed_msa_dir",
        type=str,
        default=None,
        help="Path to processed MSA NPZ files directory. If not provided, defaults to out_dir/processed/msa",
    )

    # Ground truth structure for inpainting (matching upstream Boltz1 predict.py)
    parser.add_argument(
        "--ground_truth_structure_dir",
        type=str,
        default="./datasets/upstream/antibody_data/structures",
        help="Path to ground truth structure directory for inpainting evaluation",
    )
    parser.add_argument("--structure_inpainting", action="store_true", help="Whether to perform structure inpainting")

    # Text conditioning
    parser.add_argument(
        "--use_text_conditioning",
        action="store_true",
        help="Use text conditioning from gt CDR sequences (default: disabled)",
    )
    parser.add_argument(
        "--gt_oracle_cdr",
        action="store_true",
        default=False,
        help="Use ground-truth CDR residue types and coordinates as input-side oracle features",
    )

    # Model settings
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu"], help="Accelerator type")
    # Note: --dtype removed because Boltz requires fp32

    # Inference settings
    parser.add_argument("--recycling_steps", type=int, default=3, help="Number of recycling steps")
    parser.add_argument("--sampling_steps", type=int, default=200, help="Number of diffusion sampling steps")
    parser.add_argument("--diffusion_samples", type=int, default=1, help="Number of diffusion samples per input")
    # Diffusion parameters
    parser.add_argument("--step_scale", type=float, default=1.638, help="Step size scale for diffusion")
    parser.add_argument("--gamma_0", type=float, default=0.605, help="Gamma_0 for SDE stochasticity")
    parser.add_argument("--gamma_min", type=float, default=1.107, help="Gamma_min threshold")
    parser.add_argument("--noise_scale", type=float, default=0.901, help="Noise injection scale")
    parser.add_argument("--rho", type=float, default=8.0, help="Rho for noise schedule")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sequence sampling")
    parser.add_argument(
        "--noise_type",
        type=str,
        default="discrete_absorb",
        choices=["discrete_absorb", "discrete_uniform", "continuous"],
        help="Noise type for diffusion process",
    )
    parser.add_argument(
        "--only_structure_prediction",
        action="store_true",
        help="Only perform structure prediction (no sequence generation)",
    )
    parser.add_argument("--write_full_pae", action="store_true", help="Write full PAE matrix to output")
    parser.add_argument("--write_full_pde", action="store_true", help="Write full PDE matrix to output")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum number of samples to process (-1 for all)")

    # Output format options
    parser.add_argument(
        "--output_format", type=str, default="pdb", choices=["npz", "pdb", "both"], help="Output format for structures"
    )

    # Distributed inference options
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="Number of batches to prefetch per worker")

    # Random seed
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")

    # Cache directory
    parser.add_argument(
        "--cache", type=str, default="ckpts/upstream", help="Path to cache directory for CCD and model files"
    )

    # Override existing predictions
    parser.add_argument("--override", action="store_true", help="Override existing predictions")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print a summary without loading a model",
    )

    # Use epitope region information
    parser.add_argument(
        "--use_epitope", action="store_true", default=True, help="Use epitope region information for antigen"
    )
    parser.add_argument("--no_epitope", action="store_false", dest="use_epitope", help="Disable epitope region support")

    # Preprocessed data path (optional legacy dev hook; not consumed by current
    # parse path — kept for backward compatibility with multi-GPU experiments).
    parser.add_argument(
        "--preprocessed_data_path",
        type=str,
        default=None,
        help="Optional JSON file with per-entry preprocessed metadata (legacy dev hook; safe to omit)",
    )

    # CDR JSON output options (enabled by default)
    parser.add_argument(
        "--no_cdr_json", action="store_true", default=False, help="Disable CDR JSON output (enabled by default)"
    )

    # Precomputed CDR hidden states
    parser.add_argument(
        "--precomputed_cdr_dir",
        type=str,
        default=None,
        help="Path to precomputed CDR hidden states directory. "
        "Files should be named {record_id}_SAMPLE_N.safetensors. "
        "When provided, automatically enables text conditioning mode.",
    )

    parsed_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(parsed_argv)
    gt_dir_provided = any(
        arg == "--ground_truth_structure_dir" or arg.startswith("--ground_truth_structure_dir=") for arg in parsed_argv
    )
    if args.gt_oracle_cdr and (not gt_dir_provided or not args.ground_truth_structure_dir):
        parser.error("--gt" "_oracle_cdr requires --ground_truth_structure_dir")

    return args


def build_processor(checkpoint_path: str):
    """
    Build processor (tokenizer) for text conditioning.

    Uses the understanding_model_id from the checkpoint's config.json,
    or falls back to default Qwen3 model.

    Parameters
    ----------
    checkpoint_path : str
        Path to HF checkpoint directory

    Returns
    -------
    processor
        Transformers processor/tokenizer
    """
    from transformers import AutoProcessor, AutoTokenizer

    config_path = Path(checkpoint_path) / "config.json"
    understanding_model_id = "Qwen/Qwen3-4B-Instruct-2507"  # Default

    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
            understanding_model_id = config.get("understanding_model_id", understanding_model_id)

    logger.info(f"Loading tokenizer from: {understanding_model_id}")

    # Try AutoProcessor first, then AutoTokenizer
    try:
        processor = AutoProcessor.from_pretrained(understanding_model_id)
    except Exception:
        processor = AutoTokenizer.from_pretrained(understanding_model_id)

    return processor


def bagel_collate_fn(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for BagelPredictionDataset.

    Parameters
    ----------
    data : List[Dict]
        List of samples, each containing {"text": {...}, "boltz": {...}}
        Text can be either:
        - Normal mode: input_ids, attention_mask, chain_type_ids, cdr_region_type_ids
        - Precomputed mode: precomputed_hidden_states, attention_mask, chain_type_ids, cdr_region_type_ids

    Returns
    -------
    Dict
        Batched data with {"text": {...}, "boltz": {...}}
    """
    from torch.nn.utils.rnn import pad_sequence

    # Separate text and boltz features
    text_features = []
    boltz_features = []

    for item in data:
        text_features.append(item["text"])
        boltz_features.append(item["boltz"])

    # Collate boltz features using existing collate function
    boltz_batch = collate(boltz_features)

    # Collate text features
    text_batch = {}

    if len(text_features) == 0:
        return {"text": text_batch, "boltz": boltz_batch}

    # Check if this is precomputed mode (has precomputed_hidden_states)
    is_precomputed = "precomputed_hidden_states" in text_features[0]

    if is_precomputed:
        # Precomputed mode: pad precomputed_hidden_states and related tensors
        # Check if any sample has non-empty hidden states
        non_empty = [t for t in text_features if t["precomputed_hidden_states"].numel() > 0]

        if len(non_empty) > 0:
            # Pad precomputed_hidden_states [B, L, hidden_dim]
            hidden_states_list = [t["precomputed_hidden_states"] for t in text_features]
            # pad_sequence expects [L, hidden_dim], returns [B, L, hidden_dim] with batch_first=True
            text_batch["precomputed_hidden_states"] = pad_sequence(
                hidden_states_list, batch_first=True, padding_value=0.0
            )

            # Pad other tensors
            for key in ["attention_mask", "chain_type_ids", "cdr_region_type_ids"]:
                tensors = [t[key] for t in text_features]
                if any(t.numel() > 0 for t in tensors):
                    pad_value = 0 if key == "attention_mask" else -1
                    text_batch[key] = pad_sequence(tensors, batch_first=True, padding_value=pad_value)
                else:
                    text_batch[key] = torch.tensor([], dtype=torch.long)

            # Pad cdr_confidence
            cdr_confidence_list = [t.get("cdr_confidence") for t in text_features]
            if cdr_confidence_list[0] is not None:
                text_batch["cdr_confidence"] = pad_sequence(cdr_confidence_list, batch_first=True, padding_value=0.0)
        else:
            # All samples have empty precomputed hidden states
            text_batch["precomputed_hidden_states"] = torch.tensor([], dtype=torch.float32)
            for key in ["attention_mask", "chain_type_ids", "cdr_region_type_ids"]:
                text_batch[key] = torch.tensor([], dtype=torch.long)
            text_batch["cdr_confidence"] = torch.tensor([], dtype=torch.float32)

    elif "input_ids" in text_features[0]:
        # Normal mode: pad input_ids and related tensors
        # Check if any sample has non-empty text
        non_empty = [t for t in text_features if t["input_ids"].numel() > 0]

        if len(non_empty) > 0:
            # Pad text tensors
            for key in ["input_ids", "attention_mask", "chain_type_ids", "cdr_region_type_ids"]:
                tensors = [t[key] for t in text_features]
                # Filter empty tensors and pad
                if any(t.numel() > 0 for t in tensors):
                    # Pad with 0 for input_ids and attention_mask, -1 for type ids
                    pad_value = 0 if key in ["input_ids", "attention_mask"] else -1
                    text_batch[key] = pad_sequence(tensors, batch_first=True, padding_value=pad_value)
                else:
                    text_batch[key] = torch.tensor([], dtype=torch.long)
        else:
            # All samples have empty text
            for key in ["input_ids", "attention_mask", "chain_type_ids", "cdr_region_type_ids"]:
                text_batch[key] = torch.tensor([], dtype=torch.long)

    return {"text": text_batch, "boltz": boltz_batch}


def main(argv: Sequence[str] | None = None):
    """
    Main entry point for ProteoR1 generate inference.

    Supports two modes:
    1. Single-GPU mode: python -m proteor1.generate.inference.runner ...
    2. Multi-GPU mode: torchrun --nproc_per_node=N -m proteor1.generate.inference.runner ...
    """
    args = parse_args(argv)

    # Validate structure_inpainting arguments
    if args.structure_inpainting and args.ground_truth_structure_dir is None:
        logger.error("Please provide the ground truth structure directory if inpainting.")
        return

    if args.ground_truth_structure_dir is not None:
        args.ground_truth_structure_dir = Path(args.ground_truth_structure_dir).expanduser()

    # Set paths
    args.data = Path(args.data).expanduser()
    output_dir = Path(args.out_dir).expanduser()

    # Handle MSA directories
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
    if not data:
        logger.error("No predictions to run, exiting.")
        return

    if args.dry_run:
        summary = {
            "dry_run": True,
            "data": str(args.data),
            "out_dir": str(output_dir),
            "records": [path.stem for path in sorted(data, key=lambda x: x.stem)],
            "checkpoint": args.checkpoint,
            "output_format": args.output_format,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set random seed
    if args.seed is not None:
        seed_everything(args.seed)
        logger.info(f"Set random seed to {args.seed}")

    # Setup distributed environment
    rank, world_size, is_distributed = setup_distributed()

    # Setup device
    if args.accelerator == "gpu" and torch.cuda.is_available():
        if is_distributed:
            local_rank = get_local_rank()
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logger.info("Running on CPU, this will be slow.")

    if is_main_process():
        logger.info(f"Running in {'distributed' if is_distributed else 'single-GPU'} mode")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info(f"Using device: {device}")

    args.cache = Path(args.cache).expanduser()
    args.cache.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort data for consistent ordering
    data = sorted(data, key=lambda x: x.stem)

    # Apply max_samples limit
    if args.max_samples > 0:
        data = data[: args.max_samples]
        logger.info(f"Selected {len(data)} samples (max_samples={args.max_samples})")

    ccd_path = args.cache / "ccd.pkl"

    # Process inputs (only on main process)
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

    processed_dir = output_dir / "processed"

    # Load processed data
    from proteor1.generate.data_load import Manifest

    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=args.processed_msa_dir,
    )

    # Load chain_infos for text conditioning
    chain_infos_path = processed_dir / "chain_infos.json"
    chain_infos = {}
    if chain_infos_path.exists():
        with open(chain_infos_path, "r") as f:
            chain_infos = json.load(f)
        logger.info(f"Loaded chain_infos for {len(chain_infos)} records")
    else:
        logger.warning("chain_infos.json not found, text conditioning will be disabled")
        args.use_text_conditioning = False

    # Handle precomputed CDR directory
    precomputed_cdr_dir = None
    if args.precomputed_cdr_dir is not None:
        precomputed_cdr_dir = Path(args.precomputed_cdr_dir).expanduser()
        if not precomputed_cdr_dir.exists():
            logger.error(f"Precomputed CDR directory not found: {precomputed_cdr_dir}")
            return
        # Precomputed mode automatically enables text conditioning
        args.use_text_conditioning = True
        logger.info(f"Precomputed CDR mode enabled from: {precomputed_cdr_dir}")
        logger.info("Text conditioning automatically enabled for precomputed mode")

    # Create base PredictionDataset
    base_dataset = PredictionDataset(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        inpaint=args.structure_inpainting,
        ground_truth_dir=args.ground_truth_structure_dir,
        use_epitope=args.use_epitope,
        gt_oracle_cdr=args.gt_oracle_cdr,
    )

    # Build processor for text conditioning (not needed for precomputed mode)
    processor = None
    if args.use_text_conditioning and precomputed_cdr_dir is None:
        processor = build_processor(args.checkpoint)

    # Create BagelPredictionDataset
    dataset = BagelPredictionDataset(
        prediction_dataset=base_dataset,
        chain_infos=chain_infos,
        processor=processor,
        use_text_conditioning=args.use_text_conditioning and (processor is not None or precomputed_cdr_dir is not None),
        precomputed_cdr_dir=str(precomputed_cdr_dir) if precomputed_cdr_dir is not None else None,
    )

    # Create sampler for distributed mode
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        collate_fn=bagel_collate_fn,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # Create model
    # BoltzWriter requires confidence_score for ranking samples, so we enable it.
    # The trained checkpoint should have confidence_module inherited from upstream Boltz1.
    write_confidence = True

    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "write_confidence_summary": write_confidence,
        "write_full_pae": write_confidence and args.write_full_pae,
        "write_full_pde": write_confidence and args.write_full_pde,
    }

    diffusion_params = BoltzDiffusionParams()
    diffusion_params.step_scale = args.step_scale
    diffusion_params.gamma_0 = args.gamma_0
    diffusion_params.gamma_min = args.gamma_min
    diffusion_params.noise_scale = args.noise_scale
    diffusion_params.rho = args.rho
    diffusion_params.temperature = args.temperature
    diffusion_params.noise_type = args.noise_type

    # Load model based on checkpoint type
    if args.checkpoint == "upstream":
        # Use original upstream weights (baseline mode)
        logger.info(f"Loading upstream Boltz1 baseline from: {args.boltz_ckpt_path}")
        config = ProteoR1GenerateConfig(
            boltz_ckpt_path=args.boltz_ckpt_path,
            load_pretrained=True,
            ema=False,
            predict_args=predict_args,
            structure_prediction_training=False,
            sequence_prediction_training=not args.only_structure_prediction,
            confidence_prediction=True,
            confidence_imitate_trunk=True,
            structure_inpainting=args.structure_inpainting,
            alpha_pae=1.0,
            diffusion_process_args=asdict(diffusion_params),
            conditioning_method="add_embed" if args.use_text_conditioning else "none",
        )
        model = ProteoR1GenerateModel(config)
    else:
        # Load trained HF checkpoint
        # Strategy: Load config first, modify boltz_hparams, then load model
        # This ensures diffusion_process_args is correct at initialization time
        # (AtomDiffusion uses these params in __init__, so post-hoc modification doesn't work)
        logger.info(f"Loading HF checkpoint from: {args.checkpoint}")

        # Step 1: Load config and modify boltz_hparams before model initialization
        config = ProteoR1GenerateConfig.from_pretrained(args.checkpoint)

        # Step 2: Override parameters in boltz_hparams BEFORE model initialization
        config.conditioning_method = "add_embed" if args.use_text_conditioning else "none"
        if config.boltz_hparams:
            config.boltz_hparams["predict_args"] = predict_args
            config.boltz_hparams["structure_inpainting"] = args.structure_inpainting
            config.boltz_hparams["diffusion_process_args"] = asdict(diffusion_params)
            config.boltz_hparams["confidence_prediction"] = True
            config.boltz_hparams["confidence_imitate_trunk"] = True
            config.boltz_hparams["alpha_pae"] = 1.0
            if args.only_structure_prediction:
                config.boltz_hparams["sequence_prediction_training"] = False

        # Step 3: Load model with modified config
        model = ProteoR1GenerateModel.from_pretrained(
            args.checkpoint,
            config=config,
            torch_dtype=torch.float32,  # Boltz requires fp32
        )

    logger.info(f"model.config={model.config}")

    # Move to device
    model = model.to(device)  # type: ignore[arg-type]
    model.eval()

    # Create writer
    from proteor1.generate.data_load.write.writer import BoltzWriter

    pred_writer = BoltzWriter(
        data_dir=str(processed.targets_dir),
        output_dir=str(output_dir / "predictions"),
        output_format=args.output_format,
        seq_info_path=str(chain_infos_path) if chain_infos_path.exists() else None,
        save_cdr_json=not args.no_cdr_json,
    )

    # Process samples
    success_count = 0
    error_count = 0

    pbar = tqdm(
        dataloader,
        desc=f"Rank {rank}/{world_size}" if is_distributed else "Inference",
        unit="sample",
        position=rank if is_distributed else 0,
        leave=True,
    )

    for batch in pbar:
        # Move to device
        batch = send_to_device(batch, device=device)

        # Run prediction
        with torch.no_grad():
            prediction = model.predict(
                text=batch.get("text"),
                boltz=batch.get("boltz"),
                recycling_steps=args.recycling_steps,
                sampling_steps=args.sampling_steps,
                diffusion_samples=args.diffusion_samples,
            )

        if prediction.get("exception"):
            logger.error(f"[Rank {rank}] Prediction failed")
            error_count += 1
            continue

        # Write output
        pred_writer.write_on_batch_end(prediction=prediction, batch=batch["boltz"])
        success_count += 1

    logger.info(f"[Rank {rank}] Inference complete! Success={success_count}, Errors={error_count}")

    if is_distributed:
        cleanup_distributed()


if __name__ == "__main__":
    main()
