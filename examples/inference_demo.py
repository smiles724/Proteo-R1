"""ProteoR1 demo wrapper for CDR preparation and structure design.

The dry-run path is CPU-only: it prepares demo CDR inputs and validates the
design CLI invocation without loading the generation checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = REPO_ROOT / "data" / "demo"
DEMO_CANONICAL_YAMLS = REPO_ROOT / "data" / "demo_canonical_yamls"
DEMO_GT_STRUCTURES = REPO_ROOT / "data" / "demo_gt_structures"
DEMO_MSA_DIR = REPO_ROOT / "data" / "demo_msa"

FALLBACK_DEMO_ENTRIES = (
    "8q7o_A__C",
    "8sxp_H_L_C",
    "8slb_H_L_A",
    "8r9y_H_L_A",
    "8tg9_H_L_A",
)


def _demo_entries() -> list[str]:
    if not DEMO_ROOT.exists():
        return sorted(FALLBACK_DEMO_ENTRIES)
    return sorted(path.name for path in DEMO_ROOT.iterdir() if path.is_dir())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ProteoR1 demo: CDR preparation + structure design.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--demo-entry", choices=_demo_entries(), default="8q7o_A__C")
    parser.add_argument("--out", type=Path, default=Path("work/demo"))
    parser.add_argument("--dry-run", action="store_true", help="Skip the GPU Protenix dump step")
    parser.add_argument(
        "--prepare-cdr-bin",
        default="proteor1-prepare-cdr",
        help="prepare-cdr executable; falls back to python -m proteor1.cli.prepare_cdr if missing",
    )
    parser.add_argument(
        "--design-bin",
        default="proteor1-design",
        help="design executable; falls back to python -m proteor1.cli.design if missing",
    )
    parser.add_argument(
        "--no-inpainting",
        action="store_true",
        help="Disable --structure_inpainting on the design step. Default is ON because the "
             "bundled GT npz / canonical YAML / MSA assets under data/demo_* make inpainting "
             "free for the shipped demo entries (and inpaint produces the published quality).",
    )
    parser.add_argument(
        "--ground-truth-structure-dir",
        type=Path,
        default=DEMO_GT_STRUCTURES,
        help="Directory of <record_id>.npz Boltz1 Structure-format files (default: bundled "
             "data/demo_gt_structures/). Ignored when --no-inpainting is set.",
    )
    parser.add_argument(
        "--processed-msa-dir",
        type=Path,
        default=DEMO_MSA_DIR,
        help="Directory of preprocessed MSA .npz shards per chain (default: bundled data/demo_msa/).",
    )
    parser.add_argument(
        "--canonical-yaml-dir",
        type=Path,
        default=DEMO_CANONICAL_YAMLS,
        help="Directory of canonical spec_mask YAMLs (default: bundled data/demo_canonical_yamls/). "
             "Token-count-aligned with the GT npz, required for inpainting.",
    )
    return parser


def _read_required_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"{path} is empty")
    return text


def _prepare_cdr_command(binary: str) -> list[str]:
    resolved = shutil.which(binary)
    if resolved:
        return [resolved]
    return [sys.executable, "-m", "proteor1.cli.prepare_cdr"]


def _design_command(binary: str) -> list[str]:
    resolved = shutil.which(binary)
    if resolved:
        return [resolved]
    return [sys.executable, "-m", "proteor1.cli.design"]


def _extract_json(stdout: str) -> dict[str, object]:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise RuntimeError(f"prepare-cdr did not print a JSON summary:\n{stdout}")
    return json.loads(stdout[start : end + 1])


def _print_prepare_summary(summary: dict[str, object]) -> None:
    print("\n--- prepare-cdr summary ---")
    for key in ("record_id", "yaml", "masked_json", "x_mask_cif", "hidden_dump", "dry_run"):
        print(f"{key}: {summary.get(key)}")


def _print_design_summary(summary: dict[str, object]) -> None:
    print("\n--- design summary ---")
    for key in ("dry_run", "data", "out_dir", "records", "checkpoint", "output_format"):
        print(f"{key}: {summary.get(key)}")
    print("\nExpected non-dry-run output: PDB file with redesigned CDR sequences.")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    entry_dir = DEMO_ROOT / args.demo_entry
    cif_path = entry_dir / f"{args.demo_entry}.cif"
    design_points_path = entry_dir / "design_points.txt"
    if not cif_path.exists():
        raise FileNotFoundError(f"demo CIF not found: {cif_path}")
    if not design_points_path.exists():
        raise FileNotFoundError(f"demo design points not found: {design_points_path}")

    out_dir = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    design_points = _read_required_text(design_points_path)

    command = [
        *_prepare_cdr_command(args.prepare_cdr_bin),
        "--cif",
        str(cif_path),
        "--design-points",
        design_points,
        "--out",
        str(out_dir),
        "--record-id",
        args.demo_entry,
    ]
    if args.dry_run:
        command.append("--dry-run")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, text=True, capture_output=True, check=True)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    prepare_summary = _extract_json(result.stdout)
    _print_prepare_summary(prepare_summary)

    inpaint_on = not args.no_inpainting and not args.dry_run
    if inpaint_on:
        # Inpaint path uses the bundled canonical YAML (token-aligned with the GT npz)
        # instead of prepare-cdr's emitted YAML — see README §Quickstart.
        canonical_yaml = args.canonical_yaml_dir / f"{args.demo_entry}.yaml"
        if not canonical_yaml.exists():
            raise FileNotFoundError(
                f"canonical YAML for inpaint not found: {canonical_yaml}. "
                f"Pass --no-inpainting to fall back to prepare-cdr's emitted YAML "
                f"(produces lower-quality structure)."
            )
        yaml_path = canonical_yaml
    else:
        yaml_path = Path(str(prepare_summary["yaml"]))

    design_command = [
        *_design_command(args.design_bin),
        "--input_dir",
        str(yaml_path),
        "--output",
        str(out_dir / "predictions"),
    ]
    if args.dry_run:
        design_command.append("--dry-run")
    if inpaint_on:
        if not args.ground_truth_structure_dir.exists():
            raise FileNotFoundError(
                f"--ground-truth-structure-dir {args.ground_truth_structure_dir} does not exist; "
                f"either point to a dir of <record_id>.npz files or pass --no-inpainting."
            )
        design_command.extend([
            "--structure_inpainting",
            "--ground_truth_structure_dir",
            str(args.ground_truth_structure_dir),
            "--processed_msa_dir",
            str(args.processed_msa_dir),
        ])

    design_result = subprocess.run(
        design_command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    if design_result.stderr.strip():
        print(design_result.stderr.strip(), file=sys.stderr)
    if args.dry_run:
        _print_design_summary(_extract_json(design_result.stdout))
    elif design_result.stdout.strip():
        print(design_result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
