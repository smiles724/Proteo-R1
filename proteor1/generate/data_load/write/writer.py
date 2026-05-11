from dataclasses import asdict, replace
import json
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from proteor1.generate.data_load.types import (
    Interface,
    Record,
    Structure,
)
from proteor1.generate.data_load import const
from proteor1.generate.data_load.write.mmcif import to_mmcif
from proteor1.generate.data_load.write.pdb import to_pdb


def find_cdr_segments(spec_mask: str) -> List[Tuple[int, int]]:
    """
    Find continuous CDR segments from spec_mask.

    Parameters
    ----------
    spec_mask : str
        '0'/'1' string where '1' indicates CDR position

    Returns
    -------
    List[Tuple[int, int]]
        List of (start, end) tuples for each continuous CDR segment.
        Note: end is inclusive.

    Example
    -------
    >>> find_cdr_segments("000111110000011110000111000")
    [(3, 7), (13, 16), (21, 23)]
    """
    segments = []
    start = -1
    for i, char in enumerate(spec_mask):
        if char == '1' and start == -1:
            start = i
        elif char == '0' and start != -1:
            segments.append((start, i - 1))
            start = -1
    # Handle segment at the end
    if start != -1:
        segments.append((start, len(spec_mask) - 1))
    return segments


def extract_cdr_sequences_by_segment_order(
    seq: str,
    spec_mask: str,
    h_chain_length: int,
) -> Tuple[Dict, Dict]:
    """
    Extract CDR sequences using the simplified segment order method.

    Assumptions (based on upstream Boltz1 YAML format):
    - seq and spec_mask contain only H+L chains (no Antigen)
    - H chain comes first, L chain follows
    - CDR segments in order: HCDR1, HCDR2, HCDR3, LCDR1, LCDR2, LCDR3

    Parameters
    ----------
    seq : str
        Full sequence (H+L concatenated)
    spec_mask : str
        '0'/'1' string where '1' indicates CDR position
    h_chain_length : int
        Length of H chain, used to separate H/L chains

    Returns
    -------
    Tuple[Dict, Dict]
        (heavy_chain_info, light_chain_info)
        Each dict contains "sequence", "length", "cdr_regions"
    """
    # Validate inputs
    if len(seq) != len(spec_mask):
        raise ValueError(f"seq length ({len(seq)}) != spec_mask length ({len(spec_mask)})")

    if h_chain_length > len(seq):
        raise ValueError(f"h_chain_length ({h_chain_length}) > seq length ({len(seq)})")

    # Find all CDR segments (global coordinates)
    segments = find_cdr_segments(spec_mask)

    # CDR labels in expected order
    cdr_labels = ["HCDR1", "HCDR2", "HCDR3", "LCDR1", "LCDR2", "LCDR3"]

    # Initialize results
    h_chain = {
        "sequence": seq[:h_chain_length],
        "length": h_chain_length,
        "cdr_regions": {}
    }
    l_chain = {
        "sequence": seq[h_chain_length:],
        "length": len(seq) - h_chain_length,
        "cdr_regions": {}
    }

    # Validate that H chain CDRs (first 3 segments) fall within h_chain_length
    for i, (start, end) in enumerate(segments[:3]):
        if end >= h_chain_length:
            logging.warning(
                f"CDR segment {i+1} (start={start}, end={end}) exceeds h_chain_length={h_chain_length}. "
                f"This may indicate incorrect CDR ordering assumption."
            )

    # Process each CDR segment
    for i, (start, end) in enumerate(segments):
        if i >= len(cdr_labels):
            logging.warning(f"Found more than 6 CDR segments, ignoring segment {i+1}: ({start}, {end})")
            break

        label = cdr_labels[i]
        cdr_seq = seq[start:end+1]

        if label.startswith("H"):
            # H chain CDR, coordinates are already relative to H chain
            cdr_name = label[1:]  # "CDR1", "CDR2", "CDR3"
            h_chain["cdr_regions"][cdr_name] = {
                "sequence": cdr_seq,
                "start": start,
                "end": end,
            }
        else:
            # L chain CDR, convert to L chain relative coordinates
            cdr_name = label[1:]  # "CDR1", "CDR2", "CDR3"
            l_start = start - h_chain_length
            l_end = end - h_chain_length
            l_chain["cdr_regions"][cdr_name] = {
                "sequence": cdr_seq,
                "start": l_start,
                "end": l_end,
            }

    return h_chain, l_chain


def calculate_per_cdr_aar(
    pred_h_chain: Dict,
    pred_l_chain: Dict,
    gt_h_chain: Dict,
    gt_l_chain: Dict,
) -> Dict[str, Optional[float]]:
    """
    Calculate AAR for each CDR region.

    Parameters
    ----------
    pred_h_chain, pred_l_chain : Dict
        Predicted H/L chain info with cdr_regions
    gt_h_chain, gt_l_chain : Dict
        Ground truth H/L chain info with cdr_regions

    Returns
    -------
    Dict[str, Optional[float]]
        Key is CDR label (HCDR1, HCDR2, ...), value is AAR (0.0-1.0) or None
    """
    aar_dict = {}

    # H chain CDRs
    for cdr_name in ["CDR1", "CDR2", "CDR3"]:
        label = f"H{cdr_name}"
        pred_cdr = pred_h_chain.get("cdr_regions", {}).get(cdr_name, {})
        gt_cdr = gt_h_chain.get("cdr_regions", {}).get(cdr_name, {})

        pred_seq = pred_cdr.get("sequence", "")
        gt_seq = gt_cdr.get("sequence", "")

        if len(pred_seq) == 0 or len(gt_seq) == 0 or len(pred_seq) != len(gt_seq):
            aar_dict[label] = None
        else:
            matches = sum(1 for p, g in zip(pred_seq, gt_seq) if p == g)
            aar_dict[label] = matches / len(pred_seq)

    # L chain CDRs
    for cdr_name in ["CDR1", "CDR2", "CDR3"]:
        label = f"L{cdr_name}"
        pred_cdr = pred_l_chain.get("cdr_regions", {}).get(cdr_name, {})
        gt_cdr = gt_l_chain.get("cdr_regions", {}).get(cdr_name, {})

        pred_seq = pred_cdr.get("sequence", "")
        gt_seq = gt_cdr.get("sequence", "")

        if len(pred_seq) == 0 or len(gt_seq) == 0 or len(pred_seq) != len(gt_seq):
            aar_dict[label] = None
        else:
            matches = sum(1 for p, g in zip(pred_seq, gt_seq) if p == g)
            aar_dict[label] = matches / len(pred_seq)

    return aar_dict


def calculate_aar(seq_predict, seq_truth, seq_mask):
    assert len(seq_predict) == len(seq_truth) == len(seq_mask)

    segments = []
    start = -1
    for i in range(len(seq_mask)):
        if seq_mask[i] == '1' and start == -1:
            start = i
        elif seq_mask[i] == '0' and start != -1:
            segments.append((start, i - 1))
            start = -1
    if start != -1:
        segments.append((start, len(seq_mask) - 1))

    accuracies = []
    total_matches = 0
    total_count = 0

    for (start, end) in segments:
        seg_len = end - start + 1
        matches = sum(1 for i in range(start, end + 1) if seq_predict[i] == seq_truth[i])
        accuracies.append(matches / seg_len)

        total_matches += matches
        total_count += seg_len

    total_accuracy = total_matches / total_count

    h_acc_segments = segments[:3]
    h_acc_matches = sum(
        sum(1 for i in range(start, end + 1) if seq_predict[i] == seq_truth[i])
        for (start, end) in h_acc_segments
    )
    h_acc_len = sum(end - start + 1 for (start, end) in h_acc_segments)
    h_acc = h_acc_matches / h_acc_len if h_acc_len > 0 else 0

    l_acc_segments = segments[-3:]
    l_acc_matches = sum(
        sum(1 for i in range(start, end + 1) if seq_predict[i] == seq_truth[i])
        for (start, end) in l_acc_segments
    )
    l_acc_len = sum(end - start + 1 for (start, end) in l_acc_segments)
    l_acc = l_acc_matches / l_acc_len if l_acc_len > 0 else 0

    return accuracies, total_accuracy, h_acc, l_acc


class BoltzWriter:
    """Custom writer for predictions."""

    def __init__(
            self,
            data_dir: str,
            output_dir: str,
            output_format: Literal["pdb", "mmcif"] = "mmcif",
            seq_info_path: Optional[str] = None,
            save_cdr_json: bool = True,
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        data_dir : str
            The directory containing input structure NPZ files.
        output_dir : str
            The directory to save the predictions.
        output_format : str
            Output format for structures ("pdb" or "mmcif").
        seq_info_path : str, optional
            Path to chain_infos.json containing ground truth info.
        save_cdr_json : bool
            Whether to save CDR prediction JSON files (default: True).

        """
        super().__init__()
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0
        self.seq_info_path = seq_info_path
        self.save_cdr_json = save_cdr_json

        # Create the output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_cdr_json(
        self,
        record_id: str,
        seq: Tensor,
        seq_gt: str,
        spec_mask: str,
        h_chain_length: int,
        confidence_score: Optional[Tensor],
        idx_to_rank: Dict[int, int],
    ) -> Dict:
        """
        Build CDR prediction JSON data structure (chain-separated format).

        Parameters
        ----------
        record_id : str
            Record ID
        seq : Tensor
            Predicted sequence tensor, shape: [num_samples, seq_len]
        seq_gt : str
            Ground truth sequence
        spec_mask : str
            CDR mask string ('0'/'1')
        h_chain_length : int
            H chain length
        confidence_score : Optional[Tensor]
            Confidence scores for ranking
        idx_to_rank : Dict[int, int]
            Model index to rank mapping

        Returns
        -------
        Dict
            JSON data structure
        """
        # Extract GT CDR regions
        gt_h_chain, gt_l_chain = extract_cdr_sequences_by_segment_order(
            seq=seq_gt,
            spec_mask=spec_mask,
            h_chain_length=h_chain_length,
        )

        result = {
            "record_id": record_id,
            "version": "2.0",
            "ground_truth": {
                "heavy_chain": gt_h_chain,
                "light_chain": gt_l_chain,
            },
            "predictions": [],
        }

        # Process each diffusion sample
        for model_idx in range(seq.shape[0]):
            # Convert predicted sequence to string
            pred_seq_str = "".join([
                const.prot_token_to_letter.get(const.tokens[int(x.item())], "X")
                for x in seq[model_idx]
            ])
            pred_seq_str = pred_seq_str[:len(seq_gt)]  # Truncate to GT length

            # Extract predicted CDR regions
            pred_h_chain, pred_l_chain = extract_cdr_sequences_by_segment_order(
                seq=pred_seq_str,
                spec_mask=spec_mask,
                h_chain_length=h_chain_length,
            )

            # Calculate per-CDR AAR
            per_cdr_aar = calculate_per_cdr_aar(
                pred_h_chain, pred_l_chain,
                gt_h_chain, gt_l_chain,
            )

            # Calculate overall AAR (reuse existing function)
            _, total_aar, h_aar, l_aar = calculate_aar(
                pred_seq_str, seq_gt, spec_mask
            )

            # Add AAR to predicted CDR regions
            for cdr_name in pred_h_chain.get("cdr_regions", {}):
                pred_h_chain["cdr_regions"][cdr_name]["aar"] = per_cdr_aar.get(f"H{cdr_name}")
            for cdr_name in pred_l_chain.get("cdr_regions", {}):
                pred_l_chain["cdr_regions"][cdr_name]["aar"] = per_cdr_aar.get(f"L{cdr_name}")

            prediction_entry = {
                "rank": idx_to_rank.get(model_idx, model_idx),
                "model_idx": model_idx,
                "confidence_score": (
                    confidence_score[model_idx].item()
                    if confidence_score is not None
                    else None
                ),
                "heavy_chain": pred_h_chain,
                "light_chain": pred_l_chain,
                "metrics": {
                    "total_aar": total_aar,
                    "h_chain_aar": h_aar,
                    "l_chain_aar": l_aar,
                    "per_cdr_aar": per_cdr_aar,
                },
            }
            result["predictions"].append(prediction_entry)

        # Sort by rank
        result["predictions"].sort(key=lambda x: x["rank"])

        return result

    def write_on_batch_end(
            self,
            prediction: dict[str, Tensor],
            batch: dict[str, Tensor],
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]

        seqs = prediction["seqs"]

        if seqs is not None:
            seqs = seqs.unsqueeze(0)
            assert seqs.shape[0] == coords.shape[0]
        else:
            seqs = [""] * len(records)

        # Get ranking
        argsort = torch.argsort(prediction["confidence_score"], descending=True)
        idx_to_rank = {idx.item(): rank for rank, idx in enumerate(argsort)}

        seqs_info = {}
        if self.seq_info_path is not None:
            seqs_info = json.load(open(self.seq_info_path))

        # Iterate over the records
        for record, coord, pad_mask, seq in zip(records, coords, pad_masks, seqs):
            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            # Save the structure
            struct_dir = self.output_dir / record.id
            struct_dir.mkdir(exist_ok=True)

            if len(seq) > 0:
                seq_info = seqs_info.get(record.id, None)
                seq_path = struct_dir / f"{record.id}.seq"
                with seq_path.open("w") as f:
                    seq_gt = seq_info["seq_gt"] if seq_info is not None else None
                    spec_mask = seq_info["spec_mask"] if seq_info is not None else None
                    title_str = "Rank\tSequence\tTotal\tH\tL\tH1\tH2\tH3\tL1\tL2\tL3\n" if seq_gt else "Rank\tSequence\n"
                    f.write(title_str)
                    lines = {}
                    for model_idx in range(seq.shape[0]):
                        seq_str = "".join(
                            [const.prot_token_to_letter[const.tokens[int(x.item())]] for x in seq[model_idx]])
                        if seq_gt:
                            seq_str = seq_str[:len(seq_gt)] if seq_gt else seq_str
                            accuracies, total_accuracy, h_acc, l_acc = calculate_aar(seq_str, seq_gt, spec_mask)
                            aar_str = "\t".join([f"{acc:.3f}" for acc in accuracies])
                            lines[idx_to_rank[
                                model_idx]] = f"{idx_to_rank[model_idx]}\t{seq_str}\t{total_accuracy:.3f}\t{h_acc:.3f}\t{l_acc:.3f}\t{aar_str}\n"
                        else:
                            lines[idx_to_rank[model_idx]] = f"{idx_to_rank[model_idx]}\t{seq_str}\n"

                    sorted_lines = {k: lines[k] for k in sorted(lines)}
                    for line in sorted_lines.values():
                        f.write(line)

                # Save CDR JSON if enabled
                if self.save_cdr_json and seq_info is not None:
                    h_chain_length = seq_info.get("h_chain_length")

                    if h_chain_length is not None:
                        cdr_json = self._build_cdr_json(
                            record_id=record.id,
                            seq=seq,
                            seq_gt=seq_gt,
                            spec_mask=spec_mask,
                            h_chain_length=h_chain_length,
                            confidence_score=prediction.get("confidence_score"),
                            idx_to_rank=idx_to_rank,
                        )

                        cdr_json_path = struct_dir / f"{record.id}_cdr_predictions.json"
                        with cdr_json_path.open("w") as f:
                            json.dump(cdr_json, f, indent=2)
                    else:
                        logging.warning(
                            f"Cannot save CDR JSON for {record.id}: h_chain_length not available"
                        )

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                if len(seq) > 0:
                    residues["res_type"] = seq[model_idx].cpu().numpy()
                    res_name = [const.tokens[int(x.item())] for x in seq[model_idx]]
                    residues["name"] = np.array(res_name, dtype=np.dtype("<U5"))

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                new_structure: Structure = replace(
                    structure,
                    atoms=atoms,
                    residues=residues,
                    interfaces=interfaces,
                )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                if self.output_format == "pdb":
                    path = (
                            struct_dir / f"{record.id}_model_{idx_to_rank[model_idx]}.pdb"
                    )
                    with path.open("w") as f:
                        f.write(to_pdb(new_structure))
                elif self.output_format == "mmcif":
                    path = (
                            struct_dir / f"{record.id}_model_{idx_to_rank[model_idx]}.cif"
                    )
                    with path.open("w") as f:
                        if "plddt" in prediction:
                            f.write(
                                to_mmcif(new_structure, prediction["plddt"][model_idx])
                            )
                        else:
                            f.write(to_mmcif(new_structure))
                else:
                    path = (
                            struct_dir / f"{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, **asdict(new_structure))

                # Save confidence summary
                if "plddt" in prediction:
                    path = (
                            struct_dir
                            / f"confidence_{record.id}_model_{idx_to_rank[model_idx]}.json"
                    )
                    confidence_summary_dict = {}
                    for key in [
                        "confidence_score",
                        "ptm",
                        "iptm",
                        "ligand_iptm",
                        "protein_iptm",
                        "complex_plddt",
                        "complex_iplddt",
                        "complex_pde",
                        "complex_ipde",
                    ]:
                        confidence_summary_dict[key] = prediction[key][model_idx].item()
                    confidence_summary_dict["chains_ptm"] = {
                        idx: prediction["pair_chains_iptm"][idx][idx][model_idx].item()
                        for idx in prediction["pair_chains_iptm"]
                    }
                    confidence_summary_dict["pair_chains_iptm"] = {
                        idx1: {
                            idx2: prediction["pair_chains_iptm"][idx1][idx2][
                                model_idx
                            ].item()
                            for idx2 in prediction["pair_chains_iptm"][idx1]
                        }
                        for idx1 in prediction["pair_chains_iptm"]
                    }
                    with path.open("w") as f:
                        f.write(
                            json.dumps(
                                confidence_summary_dict,
                                indent=4,
                            )
                        )

                    # Save plddt
                    plddt = prediction["plddt"][model_idx]
                    path = (
                            struct_dir
                            / f"plddt_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, plddt=plddt.cpu().numpy())

                # Save pae
                if "pae" in prediction:
                    pae = prediction["pae"][model_idx]
                    path = (
                            struct_dir
                            / f"pae_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pae=pae.cpu().numpy())

                # Save pde
                if "pde" in prediction:
                    pde = prediction["pde"][model_idx]
                    path = (
                            struct_dir
                            / f"pde_{record.id}_model_{idx_to_rank[model_idx]}.npz"
                    )
                    np.savez_compressed(path, pde=pde.cpu().numpy())

    def on_predict_epoch_end(
            self,
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201

