"""Thin console-script shim for ProteoR1 diffusion inference."""

from __future__ import annotations

import sys
from collections.abc import Sequence


_HELP = """usage: proteor1-design --data DATA [--out_dir OUT_DIR] [options]

ProteoR1 diffusion inference.

options:
  -h, --help                         show this help message and exit
  --data DATA, --input_dir DATA      path to input YAML file or directory
  --out_dir OUT_DIR, --output OUT_DIR
                                      output directory for predictions
  --checkpoint CHECKPOINT            HF Hub repo id or local path, or 'upstream' for baseline
  --boltz-ckpt-path PATH          upstream Boltz1 checkpoint path
  --accelerator {gpu,cpu}            accelerator type
  --recycling_steps N                number of recycling steps
  --sampling_steps N                 number of diffusion sampling steps
  --diffusion_samples N              number of diffusion samples per input
  --output_format {npz,pdb,both}     output format for structures
  --no_cdr_json                      disable CDR JSON output
  --precomputed_cdr_dir DIR          precomputed CDR hidden-state directory
  --override                         override existing predictions
  --dry-run                          validate inputs and print a summary without loading a model
"""


def main(argv: Sequence[str] | None = None) -> int | None:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in args):
        print(_HELP)
        return 0

    from proteor1.generate.inference.runner import main as inference_main

    return inference_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
