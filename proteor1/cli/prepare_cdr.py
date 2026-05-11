"""Prepare CDR redesign inputs for the ProteoR1 OSS workflow."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ._protenix_dump import run_protenix_encoder_dump
from ._yaml_emit import ChainSpec, emit_oss_yaml


@dataclass(frozen=True)
class ParsedDesignPoints:
    raw: str


@dataclass(frozen=True)
class ChainSelection:
    heavy_chain: str
    light_chain: str | None
    antigen_chains: list[str]


_HOTSPOT_ITEM_RE = re.compile(r"\[[A-Za-z0-9]+,\s*[1-9][0-9]*\]")
_HOTSPOT_LIST_RE = re.compile(
    rf"\s*{_HOTSPOT_ITEM_RE.pattern}\s*(?:,\s*{_HOTSPOT_ITEM_RE.pattern}\s*)*"
)


def parse_design_points(value: str) -> ParsedDesignPoints:
    """Parse antigen hotspot tuples."""

    text = value.strip()
    if not text:
        raise argparse.ArgumentTypeError("at least one design point is required")

    if _HOTSPOT_LIST_RE.fullmatch(text) is None:
        raise argparse.ArgumentTypeError(
            "design points must use antigen hotspot tuples, for example [C,4], [C,1], [C,71]"
        )
    return ParsedDesignPoints(raw=text)


def infer_chain_selection(points: ParsedDesignPoints, record_id: str | None = None) -> ChainSelection:
    """Infer cdr_eval chain IDs from the record id."""

    if record_id:
        parts = record_id.split("_")
        if len(parts) >= 2 and parts[1]:
            heavy_chain = parts[1]
            light_chain = parts[2] if len(parts) >= 3 and parts[2] else None
            antigen_chains = list(parts[3]) if len(parts) >= 4 and parts[3] else []
            return ChainSelection(heavy_chain=heavy_chain, light_chain=light_chain, antigen_chains=antigen_chains)
    heavy_chain = "H"
    light_chain = "L"
    return ChainSelection(heavy_chain=heavy_chain, light_chain=light_chain, antigen_chains=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proteor1-prepare-cdr",
        description="Prepare ProteoR1 CDR redesign artifacts from a user GT CIF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cif", type=Path, required=True, help="User-provided ground-truth CIF path")
    parser.add_argument(
        "--design-points",
        type=parse_design_points,
        required=True,
        help="antigen hotspot tuples, for example [C,4], [C,1], [C,71]",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--record-id", default=None, help="Record id; defaults to the CIF filename stem")
    parser.add_argument(
        "--no-cdr-json",
        dest="no_cdr_json",
        action="store_true",
        default=True,
        help="Skip generation CDR JSON output in downstream OSS inference",
    )
    parser.add_argument(
        "--cdr-json",
        dest="no_cdr_json",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--triangle-by-torch",
        choices=("auto", "true", "false"),
        default="auto",
        help="Use PyTorch triangle ops for Protenix, or auto-detect cuequivariance",
    )
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index used by downstream single-sample design")
    parser.add_argument("--override", dest="override", action="store_true", default=True, help="Overwrite prior outputs")
    parser.add_argument("--no-override", dest="override", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument(
        "--emit-cdr-hidden",
        action="store_true",
        help="After the Protenix dump, run the understand model to emit real CDR hidden states",
    )
    parser.add_argument(
        "--understand-ckpt",
        type=str,
        default="thinking-bio-lab/proteor1-understand",
        help="HF Hub repo id or local directory for the ProteoR1 understand checkpoint, used with --emit-cdr-hidden",
    )
    parser.add_argument("--cdr-hidden-device", default="cuda", help="Torch device for --emit-cdr-hidden")
    parser.add_argument(
        "--cdr-hidden-max-new-tokens",
        type=int,
        default=2048,
        help="Maximum understand-model generation tokens for --emit-cdr-hidden",
    )
    parser.add_argument("--cdr-hidden-temperature", type=float, default=0.7, help="CDR hidden generation temperature")
    parser.add_argument("--cdr-hidden-top-p", type=float, default=0.8, help="CDR hidden generation top-p")
    parser.add_argument("--cdr-hidden-top-k", type=int, default=20, help="CDR hidden generation top-k")
    parser.add_argument("--cdr-hidden-seed", type=int, default=2025, help="Torch seed for reproducible CDR hidden generation")
    parser.add_argument(
        "--cdr-hidden-num-beams",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip the GPU Protenix dump step")
    return parser


def format_design_points(points: ParsedDesignPoints) -> str:
    return points.raw


def _protein_chain_ids(protein: dict, fallback: str) -> str:
    chain_id = protein.get("id") or protein.get("chain_id") or protein.get("auth_asym_id") or fallback
    if isinstance(chain_id, list):
        return ",".join(str(item) for item in chain_id)
    return str(chain_id)


def _protein_sequence_entry(item: dict) -> dict | None:
    if "protein" in item:
        return item["protein"]
    if "proteinChain" in item:
        return item["proteinChain"]
    return None


def chain_specs_from_masking_result(result: Any) -> list[ChainSpec]:
    """Convert cdr_eval masked JSON output into YAML chain specs."""

    if not result.json_data:
        raise ValueError("CDR masking result has no JSON data")
    data = result.json_data[0]
    specs: list[ChainSpec] = []

    original_by_entity = {}
    if result.matched_heavy_entity is not None and result.heavy_chain_info is not None:
        original_by_entity[result.matched_heavy_entity] = result.heavy_chain_info["original_seq"]
    if result.matched_light_entity is not None and result.light_chain_info is not None:
        original_by_entity[result.matched_light_entity] = result.light_chain_info["original_seq"]

    for idx, item in enumerate(data.get("sequences", [])):
        protein = _protein_sequence_entry(item)
        if protein is None:
            continue
        sequence = str(protein["sequence"])
        ground_truth = original_by_entity.get(idx, sequence)
        if len(sequence) != len(ground_truth):
            raise ValueError(
                f"Masked sequence and ground truth length mismatch for entity {idx}: "
                f"{len(sequence)} != {len(ground_truth)}"
            )
        spec_mask = "".join("1" if seq_char != gt_char else "0" for seq_char, gt_char in zip(sequence, ground_truth))
        specs.append(
            ChainSpec(
                id=_protein_chain_ids(protein, fallback=str(idx + 1)),
                sequence=sequence,
                spec_mask=spec_mask,
                ground_truth=ground_truth,
            )
        )

    if not specs:
        raise ValueError("CDR masking result did not contain protein sequences")
    return specs


def write_x_mask_cif(
    extracted_cif: str | Path,
    output_cif: str | Path,
    mask_result: Any,
    record_id: str,
    heavy_chain_id: str = "H",
    light_chain_id: str | None = None,
) -> Path:
    """Write an X-masked CIF by marking masked CDR residues as UNK in atom_site."""

    from protenix.data.parser import MMCIFParser
    from protenix.data.utils import save_atoms_to_cif

    extracted_cif = Path(extracted_cif)
    output_cif = Path(output_cif)
    parser = MMCIFParser(extracted_cif)
    atom_array = parser.get_structure(altloc="first", model=1, bond_lenth_threshold=None)

    chain_masks = _chain_residue_masks(atom_array, mask_result, heavy_chain_id=heavy_chain_id, light_chain_id=light_chain_id)
    masked_atom_count = 0
    for atom_mask in chain_masks:
        if not np.any(atom_mask):
            continue
        _set_atom_annotation(atom_array, "res_name", atom_mask, "UNK")
        _set_atom_annotation(atom_array, "label_comp_id", atom_mask, "UNK")
        _set_atom_annotation(atom_array, "auth_comp_id", atom_mask, "UNK")
        masked_atom_count += int(np.sum(atom_mask))

    if masked_atom_count == 0:
        raise RuntimeError("CDR masking produced no residue positions for the X-mask CIF")

    output_cif.parent.mkdir(parents=True, exist_ok=True)
    save_atoms_to_cif(
        output_cif_file=str(output_cif),
        atom_array=atom_array,
        entity_poly_type=parser.entity_poly_type,
        pdb_id=record_id,
    )
    return output_cif


def _chain_residue_masks(
    atom_array: Any,
    mask_result: Any,
    heavy_chain_id: str = "H",
    light_chain_id: str | None = None,
) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    if mask_result.heavy_chain_info is not None:
        masks.append(_residue_mask_for_chain(atom_array, heavy_chain_id, mask_result.heavy_chain_info["cdr_indices"]))
    if mask_result.light_chain_info is not None and light_chain_id is not None:
        masks.append(_residue_mask_for_chain(atom_array, light_chain_id, mask_result.light_chain_info["cdr_indices"]))
    return masks


def _residue_mask_for_chain(atom_array: Any, chain_id: str, residue_indices: Sequence[int]) -> np.ndarray:
    atom_count = len(atom_array)
    result = np.zeros(atom_count, dtype=bool)
    if not residue_indices:
        return result

    chain_values = _get_atom_annotation(atom_array, "auth_asym_id")
    if chain_values is None:
        chain_values = _get_atom_annotation(atom_array, "chain_id")
    if chain_values is None:
        chain_values = _get_atom_annotation(atom_array, "label_asym_id")
    if chain_values is None:
        raise RuntimeError("X-mask CIF writing requires chain annotations")

    res_ids = _get_atom_annotation(atom_array, "res_id")
    if res_ids is None:
        raise RuntimeError("X-mask CIF writing requires residue id annotations")
    ins_codes = _get_atom_annotation(atom_array, "ins_code")

    residue_keys: list[Any] = []
    atom_key_by_index: list[Any] = []
    for index in range(atom_count):
        if str(chain_values[index]) != chain_id:
            atom_key_by_index.append(None)
            continue
        key = (str(res_ids[index]), str(ins_codes[index]) if ins_codes is not None else "")
        atom_key_by_index.append(key)
        if not residue_keys or residue_keys[-1] != key:
            residue_keys.append(key)

    selected_keys = {residue_keys[idx] for idx in residue_indices if idx < len(residue_keys)}
    for index, key in enumerate(atom_key_by_index):
        if key in selected_keys:
            result[index] = True
    return result


def _get_atom_annotation(atom_array: Any, name: str) -> np.ndarray | None:
    if hasattr(atom_array, name):
        return np.asarray(getattr(atom_array, name))
    if hasattr(atom_array, "get_annotation") and hasattr(atom_array, "get_annotation_categories"):
        if name in atom_array.get_annotation_categories():
            return np.asarray(atom_array.get_annotation(name))
    return None


def _set_atom_annotation(atom_array: Any, name: str, atom_mask: np.ndarray, value: str) -> None:
    values = _get_atom_annotation(atom_array, name)
    if values is None:
        return
    values = np.array(values, copy=True)
    values[atom_mask] = value
    if hasattr(atom_array, "set_annotation") and hasattr(atom_array, "get_annotation_categories"):
        if name in atom_array.get_annotation_categories():
            atom_array.set_annotation(name, values)
            return
    setattr(atom_array, name, values)


def prepare_cdr(args: argparse.Namespace) -> dict[str, str | int | bool | None]:
    from proteor1.cdr_eval import (
        cif_to_protenix_json,
        extract_and_mask_cdr,
        extract_chains_from_cif,
    )

    record_id = args.record_id or args.cif.stem
    out_dir = args.out
    artifacts_dir = out_dir / record_id
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    chain_selection = infer_chain_selection(args.design_points, record_id=record_id)
    extracted_cif = artifacts_dir / f"{record_id}_chains.cif"
    x_mask_cif = artifacts_dir / f"{record_id}_xmask.cif"
    masked_json = artifacts_dir / f"{record_id}_masked.json"

    extraction = extract_chains_from_cif(
        cif_path=args.cif,
        heavy_chain=chain_selection.heavy_chain,
        light_chain=chain_selection.light_chain,
        antigen_chains=chain_selection.antigen_chains,
        output_path=extracted_cif,
        entry_name=record_id,
        apply_filters=True,
    )
    if not extraction.success:
        raise RuntimeError(f"CDR chain extraction failed: {extraction.error_message}")

    json_data = cif_to_protenix_json(extracted_cif, sample_name=record_id, get_entity_seq_with_coords=False)
    mask_result = extract_and_mask_cdr(
        json_data=json_data,
        heavy_chain_id=chain_selection.heavy_chain,
        light_chain_id=chain_selection.light_chain,
        mask_token="X",
    )
    if not mask_result.success:
        raise RuntimeError(f"CDR masking failed: {mask_result.error_message}")

    with masked_json.open("w", encoding="utf-8") as handle:
        json.dump(mask_result.json_data, handle, indent=2)

    x_mask_cif = write_x_mask_cif(
        extracted_cif,
        x_mask_cif,
        mask_result,
        record_id,
        heavy_chain_id=chain_selection.heavy_chain,
        light_chain_id=chain_selection.light_chain,
    )
    yaml_path = emit_oss_yaml(record_id, chain_specs_from_masking_result(mask_result), artifacts_dir)
    hidden_dump = None
    cdr_hidden = None
    if not args.dry_run:
        hidden_dump = run_protenix_encoder_dump(
            x_mask_cif,
            artifacts_dir / "protenix_dump",
            triangle_by_torch=args.triangle_by_torch,
            override=args.override,
        )
        if args.emit_cdr_hidden:
            from proteor1.generate.inference.cdr_hidden_emit import emit_cdr_hidden_states

            cdr_hidden = emit_cdr_hidden_states(
                input_dir=artifacts_dir,
                protenix_dump_dir=hidden_dump,
                checkpoint=args.understand_ckpt,
                output=artifacts_dir / "cdr_hidden",
                record_id=record_id,
                design_points=format_design_points(args.design_points),
                masked_json=masked_json,
                sample_idx=args.sample_idx,
                device=args.cdr_hidden_device,
                max_new_tokens=args.cdr_hidden_max_new_tokens,
                temperature=args.cdr_hidden_temperature,
                top_p=args.cdr_hidden_top_p,
                top_k=args.cdr_hidden_top_k,
                seed=args.cdr_hidden_seed,
                override=args.override,
            )

    return {
        "record_id": record_id,
        "yaml": str(yaml_path),
        "masked_json": str(masked_json),
        "x_mask_cif": str(x_mask_cif),
        "hidden_dump": str(hidden_dump) if hidden_dump is not None else None,
        "cdr_hidden": str(cdr_hidden) if cdr_hidden is not None else None,
        "dry_run": args.dry_run,
        "emit_cdr_hidden": args.emit_cdr_hidden,
        "no_cdr_json": args.no_cdr_json,
        "sample_idx": args.sample_idx,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = prepare_cdr(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
