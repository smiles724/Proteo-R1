"""Subprocess wrapper for the Protenix encoder dump step."""

from __future__ import annotations

import importlib.util
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Sequence

RESULT_PATH_PATTERNS = (
    re.compile(r"Result will be saved to:\s*(?P<path>.+)"),
    re.compile(r"Output directory:\s*(?P<path>.+)"),
    re.compile(r"result(?:s)? (?:will be )?(?:saved|written) to:?\s*(?P<path>.+)", re.IGNORECASE),
)


def _tail_output(stdout: str, stderr: str, max_lines: int = 20) -> str:
    lines = [line for text in (stdout, stderr) for line in text.splitlines()]
    return "\n".join(lines[-max_lines:])


def _validate_dump_artifacts(result_path: Path, stdout: str, stderr: str) -> None:
    if not result_path.is_dir():
        raise RuntimeError(f"Protenix dump result path does not exist: {result_path}")
    if any(result_path.rglob("*.safetensors")):
        return
    tail = _tail_output(stdout, stderr)
    message = f"Protenix dump completed without producing safetensors under {result_path}"
    if tail:
        message = f"{message}\nLast subprocess output:\n{tail}"
    raise RuntimeError(message)


def resolve_triangle_by_torch(value: str | bool) -> bool:
    """Resolve auto/true/false into the boolean flag expected by the dump command."""

    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    if normalized != "auto":
        raise ValueError("triangle_by_torch must be one of: auto, true, false")
    if importlib.util.find_spec("cuequivariance") is None:
        return True

    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (12, 0):
            return True
    except Exception:
        return True

    return False


def parse_result_path(stdout: str, stderr: str = "") -> Path:
    """Extract the result path printed by the Protenix dump command."""

    text = "\n".join(part for part in (stdout, stderr) if part)
    for line in text.splitlines():
        for pattern in RESULT_PATH_PATTERNS:
            match = pattern.search(line)
            if match:
                return Path(match.group("path").strip())
    raise RuntimeError("Could not parse Protenix dump output path from subprocess output")


def build_protenix_encoder_dump_command(
    input_dir: str | Path,
    output_dir: str | Path,
    triangle_by_torch: str | bool = "auto",
    runner: str | Sequence[str] | None = None,
) -> list[str]:
    """Build the subprocess command without executing it."""

    if runner is None:
        runner = os.environ.get("PROTEOR1_PROTENIX_DUMP_CMD", "proteor1-protenix-dump")
    command = shlex.split(runner) if isinstance(runner, str) else list(runner)
    if not command:
        raise ValueError("runner command must be non-empty")

    command.extend(
        [
            "--input_dir",
            str(input_dir),
            "--output",
            str(output_dir),
            "--no-bioassembly",
            "--protein-only",
        ]
    )
    if resolve_triangle_by_torch(triangle_by_torch):
        command.append("--triangle_by_torch")
    return command


def run_protenix_encoder_dump(
    cif_path: str | Path,
    out_dir: str | Path,
    triangle_by_torch: str | bool = "auto",
    override: bool = True,
    runner: str | Sequence[str] | None = None,
) -> Path:
    """Run the Protenix dump command and return the output path it reports."""

    output_dir = Path(out_dir)
    if override and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = output_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source = Path(cif_path)
    if source.is_dir():
        input_dir = source
    else:
        shutil.copy2(source, input_dir / source.name)

    command = build_protenix_encoder_dump_command(input_dir, output_dir, triangle_by_torch, runner=runner)
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Protenix dump command was not found. Install the ProteoR1 inference entry point "
            "or set PROTEOR1_PROTENIX_DUMP_CMD."
        ) from exc

    result_path = parse_result_path(completed.stdout, completed.stderr)
    _validate_dump_artifacts(result_path, completed.stdout, completed.stderr)
    return result_path
