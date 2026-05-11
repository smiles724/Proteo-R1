"""Emit ProteoR1 OSS design YAML files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class ChainSpec:
    """Protein chain fields required by the spec_mask schema."""

    id: str
    sequence: str
    spec_mask: str
    ground_truth: str
    msa: str = "empty"

    def validate(self) -> None:
        lengths = {len(self.sequence), len(self.spec_mask), len(self.ground_truth)}
        if len(lengths) != 1:
            raise ValueError(
                f"Chain {self.id!r} has mismatched sequence/spec_mask/ground_truth lengths: "
                f"{len(self.sequence)}/{len(self.spec_mask)}/{len(self.ground_truth)}"
            )
        if any(char not in {"0", "1"} for char in self.spec_mask):
            raise ValueError(f"Chain {self.id!r} spec_mask must contain only 0/1 characters")


def emit_oss_yaml(record_id: str, chains: Iterable[ChainSpec], out_dir: str | Path) -> Path:
    """Write a spec_mask YAML file and return its path."""

    record_id = record_id.strip()
    if not record_id:
        raise ValueError("record_id must be non-empty")

    chain_list = list(chains)
    if not chain_list:
        raise ValueError("at least one chain is required")

    sequences = []
    for chain in chain_list:
        chain.validate()
        sequences.append(
            {
                "protein": {
                    "id": chain.id,
                    "sequence": chain.sequence,
                    "spec_mask": chain.spec_mask,
                    "ground_truth": chain.ground_truth,
                    "msa": chain.msa,
                }
            }
        )

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{record_id}.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump({"version": 1, "sequences": sequences}, handle, sort_keys=False)
    return output_path
