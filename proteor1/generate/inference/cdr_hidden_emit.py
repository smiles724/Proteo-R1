"""Emit Qwen-side CDR hidden states for ProteoR1 design inference."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from safetensors.torch import load_file, save_file

from proteor1.understand import ProteoR1UnderstandModel, ProteoR1UnderstandProcessor
from proteor1.understand.data_collator import prepare_batch_for_model


LOGGER = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
CDR_REGION_TYPES_INV = {"CDR1": 2, "CDR2": 4, "CDR3": 6}
CHAIN_TYPE_NONE = 0
CHAIN_TYPE_HEAVY = 1
CHAIN_TYPE_LIGHT = 2
CDR_REGION_TYPE_NONE = 0
CDR_TAGS = (
    "<HCDR1>",
    "</HCDR1>",
    "<HCDR2>",
    "</HCDR2>",
    "<HCDR3>",
    "</HCDR3>",
    "<LCDR1>",
    "</LCDR1>",
    "<LCDR2>",
    "</LCDR2>",
    "<LCDR3>",
    "</LCDR3>",
)
GENERATION_DEFAULTS = dict(temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=2048, do_sample=True)
GENERATION_TEMPERATURE = GENERATION_DEFAULTS["temperature"]
GENERATION_TOP_P = GENERATION_DEFAULTS["top_p"]
GENERATION_TOP_K = GENERATION_DEFAULTS["top_k"]
GENERATION_MAX_NEW_TOKENS = GENERATION_DEFAULTS["max_new_tokens"]
DEFAULT_GENERATION_SEED = 2025

CHAT_TEMPLATE_NO_SYSTEM = (
    "{% set audio_count = namespace(value=0) %}"
    "{% set image_count = namespace(value=0) %}"
    "{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}<|im_end|>\n"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if 'audio' in content or 'audio_url' in content %}"
    "{% set audio_count.value = audio_count.value + 1 %}"
    "<|AUDIO|>\n"
    "{% elif content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}"
    "Picture {{ image_count.value }}: "
    "{% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>\n"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}"
    "Video {{ video_count.value }}: "
    "{% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>\n"
    "{% elif 'text' in content %}"
    "{{ content['text'] }}"
    "{% endif %}"
    "{% endfor %}"
    "<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)


@dataclass(frozen=True)
class CdrEmitInputs:
    record_id: str
    yaml_path: Path
    masked_json_path: Path | None
    protenix_dump_path: Path
    output_dir: Path
    checkpoint: Path
    design_points: str | None
    sample_idx: int
    device: str
    batch_size: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    override: bool


@dataclass(frozen=True)
class CdrSegment:
    label: str
    tag: str
    expected_sequence: str

    @property
    def expected_len(self) -> int:
        return len(self.expected_sequence)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="proteor1-cdr-hidden-emit",
        description="Run the ProteoR1 understand model and save CDR hidden-state safetensors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="prepare-cdr record directory or YAML path")
    parser.add_argument("--protenix_dump_dir", type=Path, required=True, help="Protenix dump result directory")
    parser.add_argument("--checkpoint", type=Path, default=Path("pretrained/proteor1_understand"))
    parser.add_argument("--output", type=Path, default=None, help="Output directory; defaults to <input_dir>/cdr_hidden")
    parser.add_argument("--record-id", default=None, help="Record id; defaults to the YAML stem")
    parser.add_argument("--design-points", default=None, help="Design point text to include in the teacher-forced prompt")
    parser.add_argument("--masked-json", type=Path, default=None, help="Optional masked JSON artifact from prepare-cdr")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index for the output filename")
    parser.add_argument("--device", default="cuda", help="Torch device for the understand forward pass")
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=2,
        help="Minimum batch size; single samples are duplicated",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=GENERATION_MAX_NEW_TOKENS,
        help="Maximum autoregressive tokens for default CDR generation",
    )
    parser.add_argument("--temperature", type=float, default=GENERATION_TEMPERATURE, help="generation temperature")
    parser.add_argument("--top-p", type=float, default=GENERATION_TOP_P, help="nucleus sampling top-p")
    parser.add_argument("--top-k", type=int, default=GENERATION_TOP_K, help="top-k sampling cutoff")
    parser.add_argument("--seed", type=int, default=DEFAULT_GENERATION_SEED, help="Torch seed for reproducible sampling")
    parser.add_argument("--override", action="store_true", default=False, help="Overwrite existing output")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading the model")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    return parser


def _record_dir(input_dir: Path) -> Path:
    return input_dir.parent if input_dir.suffix in {".yaml", ".yml"} else input_dir


def _resolve_yaml(input_dir: Path, record_id: str | None) -> Path:
    if input_dir.suffix in {".yaml", ".yml"}:
        if not input_dir.is_file():
            raise FileNotFoundError(input_dir)
        return input_dir
    if record_id is not None:
        candidate = input_dir / f"{record_id}.yaml"
        if candidate.is_file():
            return candidate
    candidates = sorted(input_dir.glob("*.yaml"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one YAML under {input_dir}, found {len(candidates)}")
    return candidates[0]


def _resolve_masked_json(record_dir: Path, record_id: str, explicit: Path | None) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        return explicit
    candidate = record_dir / f"{record_id}_masked.json"
    return candidate if candidate.is_file() else None


def _resolve_output(record_dir: Path, output: Path | None) -> Path:
    return output if output is not None else record_dir / "cdr_hidden"


def resolve_inputs(args: argparse.Namespace) -> CdrEmitInputs:
    yaml_path = _resolve_yaml(args.input_dir, args.record_id)
    record_id = args.record_id or yaml_path.stem
    record_dir = _record_dir(args.input_dir)
    return CdrEmitInputs(
        record_id=record_id,
        yaml_path=yaml_path,
        masked_json_path=_resolve_masked_json(record_dir, record_id, args.masked_json),
        protenix_dump_path=args.protenix_dump_dir,
        output_dir=_resolve_output(record_dir, args.output),
        checkpoint=args.checkpoint,
        design_points=args.design_points,
        sample_idx=args.sample_idx,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        override=args.override,
    )


def _protein_entry(item: dict[str, Any]) -> dict[str, Any] | None:
    if "protein" in item:
        return item["protein"]
    if "proteinChain" in item:
        return item["proteinChain"]
    return None


def load_chain_specs(yaml_path: Path) -> list[dict[str, str]]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    specs: list[dict[str, str]] = []
    for item in data.get("sequences", []):
        protein = _protein_entry(item)
        if protein is None:
            continue
        chain_id = protein.get("id") or protein.get("chain_id") or protein.get("auth_asym_id")
        if isinstance(chain_id, list):
            chain_id = ",".join(str(part) for part in chain_id)
        specs.append(
            {
                "id": str(chain_id),
                "sequence": str(protein["sequence"]),
                "spec_mask": str(protein.get("spec_mask", "0" * len(str(protein["sequence"])))),
                "ground_truth": str(protein.get("ground_truth", protein["sequence"])),
            }
        )
    if not specs:
        raise ValueError(f"YAML contains no protein sequences: {yaml_path}")
    return specs


def load_masked_protein_json(masked_json_path: Path | None) -> dict[str, Any] | None:
    if masked_json_path is None:
        return None
    data = json.loads(masked_json_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            raise ValueError(f"masked JSON is empty: {masked_json_path}")
        return data[0]
    if isinstance(data, dict):
        return data
    raise ValueError(f"masked JSON must contain a dict or list[dict]: {masked_json_path}")


def _masked_segments(sequence: str, mask: str) -> list[tuple[int, int, str]]:
    segments: list[tuple[int, int, str]] = []
    start: int | None = None
    for idx, flag in enumerate(mask):
        if flag == "1" and start is None:
            start = idx
        elif flag != "1" and start is not None:
            segments.append((start, idx, sequence[start:idx]))
            start = None
    if start is not None:
        segments.append((start, len(mask), sequence[start:]))
    return segments


def _record_antibody_chain_ids(record_id: str) -> tuple[str | None, str | None]:
    parts = record_id.split("_")
    heavy = parts[1] if len(parts) >= 2 and parts[1] else None
    light = parts[2] if len(parts) >= 3 and parts[2] else None
    return heavy, light


def collect_cdr_segments(chain_specs: Sequence[dict[str, str]], record_id: str) -> list[CdrSegment]:
    heavy_id, light_id = _record_antibody_chain_ids(record_id)
    cdr_segments: list[CdrSegment] = []

    for spec in chain_specs:
        chain_id = spec["id"].split(",")[0].upper()
        if chain_id.startswith("H") or (heavy_id is not None and chain_id == heavy_id.upper()):
            chain_prefix = "H"
        elif chain_id.startswith("L") or (light_id is not None and chain_id == light_id.upper()):
            chain_prefix = "L"
        else:
            continue
        segments = _masked_segments(spec["ground_truth"], spec["spec_mask"])
        for segment_index, (_start, _end, seq) in enumerate(segments[:3], start=1):
            label = f"{chain_prefix}{segment_index}"
            tag = f"{chain_prefix}CDR{segment_index}"
            cdr_segments.append(CdrSegment(label=label, tag=tag, expected_sequence=seq))

    if not cdr_segments:
        raise ValueError("No H/L CDR segments were found in YAML spec_mask fields")
    return cdr_segments


def build_target_text(
    chain_specs: Sequence[dict[str, str]],
    record_id: str,
    predicted_cdr_sequences: Mapping[str, str] | None = None,
) -> str:
    cdr_segments = collect_cdr_segments(chain_specs, record_id)
    cdr_sequences: dict[str, dict[str, Any]] = {}

    for segment in cdr_segments:
        seq = segment.expected_sequence if predicted_cdr_sequences is None else predicted_cdr_sequences[segment.label]
        cdr_sequences[segment.label] = {
            "len": len(seq),
            "filled_positions": [{"pos": pos, "aa": aa} for pos, aa in enumerate(seq, start=1)],
            "seq": f"<{segment.tag}>{seq}</{segment.tag}>",
        }

    answer = {
        "task": "AB_CDR_REDESIGN_SFT_V2",
        "thinking": "",
        "answer": {
            "cdrs_present": [segment.label for segment in cdr_segments],
            "cdr_sequences": cdr_sequences,
        },
    }
    return json.dumps(answer, ensure_ascii=False)


def build_input_text(design_points: str | None) -> str:
    design_line = "Design points (antigen hotspots): not provided."
    if design_points:
        design_line = f"Design points (antigen hotspots): {design_points}."
    return (
        "<TASK=AB_CDR_REDESIGN_SFT_V2>\n"
        "You are redesigning masked CDR regions of an antibody to improve binding to the antigen.\n"
        f"{design_line}\n"
        "Output JSON with keys: task, thinking, answer."
    )


def _chain_id_for_spec(spec: dict[str, str], fallback: str) -> str:
    chain_id = spec["id"].split(",")[0].strip()
    return chain_id or fallback


def _chain_token_counts(asym_id: torch.Tensor) -> list[int]:
    values = asym_id.detach().cpu().tolist()
    counts: list[int] = []
    seen_order: list[int] = []
    for value in values:
        int_value = int(value)
        if int_value not in seen_order:
            seen_order.append(int_value)
            counts.append(0)
        counts[seen_order.index(int_value)] += 1
    return counts


def _protein_replacement(processor: ProteoR1UnderstandProcessor, chain_ids: Sequence[str], counts: Sequence[int]) -> str:
    lines = []
    for chain_id, count in zip(chain_ids, counts):
        expanded = processor.protein_start_token + processor.protein_token * int(count) + processor.protein_end_token
        lines.append(f"Protein chain {chain_id}: {expanded}")
    return "\n".join(lines)


def _chain_context(
    chain_specs: Sequence[dict[str, str]],
    embedding: dict[str, torch.Tensor],
) -> tuple[list[str], list[int]]:
    chain_counts = _chain_token_counts(embedding["asym_id"])
    chain_ids = [_chain_id_for_spec(spec, fallback=f"chain_{idx + 1}") for idx, spec in enumerate(chain_specs)]
    if len(chain_ids) < len(chain_counts):
        chain_ids.extend(f"chain_{idx + 1}" for idx in range(len(chain_ids), len(chain_counts)))
    return chain_ids, chain_counts


class AntibodySftEncoder:
    def __init__(self, processor: ProteoR1UnderstandProcessor) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.tokenizer.chat_template = CHAT_TEMPLATE_NO_SYSTEM

    def encode(self, input_text: str, target_text: str, chain_ids: Sequence[str], chain_token_counts: Sequence[int]) -> dict[str, torch.Tensor]:
        replacement = _protein_replacement(self.processor, chain_ids, chain_token_counts)
        messages = [
            {"role": "user", "content": f"{self.processor.protein_token} {input_text}"},
            {"role": "assistant", "content": target_text},
        ]
        messages[0]["content"] = messages[0]["content"].replace(self.processor.protein_token, replacement, 1)
        return self._process_messages_with_labels(messages)

    def encode_generation_prompt(
        self,
        input_text: str,
        chain_ids: Sequence[str],
        chain_token_counts: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        replacement = _protein_replacement(self.processor, chain_ids, chain_token_counts)
        messages = [{"role": "user", "content": f"{self.processor.protein_token} {input_text}"}]
        messages[0]["content"] = messages[0]["content"].replace(self.processor.protein_token, replacement, 1)
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids_tensor = torch.tensor(self.tokenizer.encode(rendered), dtype=torch.long)
        return {
            "input_ids": input_ids_tensor,
            "attention_mask": torch.ones_like(input_ids_tensor),
            "position_ids": self.processor.compute_compressed_position_ids(input_ids_tensor),
        }

    def _encode_with_cdr_char_level(self, text: str) -> dict[str, list[int]]:
        pattern = re.compile(r"(<(H|L)(CDR[123])>)([A-Za-z]*)(</\2\3>)")
        input_ids: list[int] = []
        chain_type_ids: list[int] = []
        cdr_region_type_ids: list[int] = []
        last_end = 0

        for match in pattern.finditer(text):
            if match.start() > last_end:
                prefix_ids = self.tokenizer.encode(text[last_end:match.start()], add_special_tokens=False)
                input_ids.extend(prefix_ids)
                chain_type_ids.extend([CHAIN_TYPE_NONE] * len(prefix_ids))
                cdr_region_type_ids.extend([CDR_REGION_TYPE_NONE] * len(prefix_ids))

            start_ids = self.tokenizer.encode(match.group(1), add_special_tokens=False)
            input_ids.extend(start_ids)
            chain_type_ids.extend([CHAIN_TYPE_NONE] * len(start_ids))
            cdr_region_type_ids.extend([CDR_REGION_TYPE_NONE] * len(start_ids))

            chain_type = CHAIN_TYPE_HEAVY if match.group(2) == "H" else CHAIN_TYPE_LIGHT
            cdr_region_type = CDR_REGION_TYPES_INV[match.group(3)]
            for aa in match.group(4):
                aa_ids = self.tokenizer.encode(aa, add_special_tokens=False)
                input_ids.append(aa_ids[0] if aa_ids else (self.tokenizer.pad_token_id or 0))
                chain_type_ids.append(chain_type)
                cdr_region_type_ids.append(cdr_region_type)

            end_ids = self.tokenizer.encode(match.group(5), add_special_tokens=False)
            input_ids.extend(end_ids)
            chain_type_ids.extend([CHAIN_TYPE_NONE] * len(end_ids))
            cdr_region_type_ids.extend([CDR_REGION_TYPE_NONE] * len(end_ids))
            last_end = match.end()

        if last_end < len(text):
            suffix_ids = self.tokenizer.encode(text[last_end:], add_special_tokens=False)
            input_ids.extend(suffix_ids)
            chain_type_ids.extend([CHAIN_TYPE_NONE] * len(suffix_ids))
            cdr_region_type_ids.extend([CDR_REGION_TYPE_NONE] * len(suffix_ids))

        if last_end == 0:
            input_ids = self.tokenizer.encode(text)
            chain_type_ids = [CHAIN_TYPE_NONE] * len(input_ids)
            cdr_region_type_ids = [CDR_REGION_TYPE_NONE] * len(input_ids)

        return {
            "input_ids": input_ids,
            "chain_type_ids": chain_type_ids,
            "cdr_region_type_ids": cdr_region_type_ids,
        }

    def _process_messages_with_labels(self, messages: Sequence[dict[str, str]]) -> dict[str, torch.Tensor]:
        input_ids: list[int] = []
        labels: list[int] = []
        chain_type_ids: list[int] = []
        cdr_region_type_ids: list[int] = []

        special_tokens = list(getattr(self.tokenizer, "additional_special_tokens", []))
        special_tokens.extend(["<|im_start|>", "<|im_end|>"])
        unmask_token_ids = {self.tokenizer.convert_tokens_to_ids(token) for token in special_tokens}

        for message in messages:
            rendered = self.tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False)
            if any(token in rendered for token in CDR_TAGS):
                encoded = self._encode_with_cdr_char_level(rendered)
                message_ids = encoded["input_ids"]
                message_chain_type_ids = encoded["chain_type_ids"]
                message_cdr_region_type_ids = encoded["cdr_region_type_ids"]
            else:
                message_ids = self.tokenizer.encode(rendered)
                message_chain_type_ids = [CHAIN_TYPE_NONE] * len(message_ids)
                message_cdr_region_type_ids = [CDR_REGION_TYPE_NONE] * len(message_ids)

            input_ids.extend(message_ids)
            chain_type_ids.extend(message_chain_type_ids)
            cdr_region_type_ids.extend(message_cdr_region_type_ids)
            if message["role"] in {"user", "system"}:
                labels.extend([-100] * len(message_ids))
            else:
                message_labels = list(message_ids)
                message_labels[:3] = [-100] * min(3, len(message_labels))
                labels.extend(message_labels)

        protein_token_ids = {
            self.processor.protein_token_id,
            self.processor.protein_start_token_id,
            self.processor.protein_end_token_id,
        }
        for idx, token_id in enumerate(input_ids):
            if token_id in unmask_token_ids:
                labels[idx] = token_id
            if token_id in protein_token_ids:
                labels[idx] = -100

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        return {
            "input_ids": input_ids_tensor,
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones_like(input_ids_tensor),
            "position_ids": self.processor.compute_compressed_position_ids(input_ids_tensor),
            "chain_type_ids": torch.tensor(chain_type_ids, dtype=torch.long),
            "cdr_region_type_ids": torch.tensor(cdr_region_type_ids, dtype=torch.long),
        }


def locate_protenix_dump_file(protenix_dump_dir: Path) -> Path:
    if protenix_dump_dir.is_file():
        return protenix_dump_dir
    if not protenix_dump_dir.is_dir():
        raise FileNotFoundError(protenix_dump_dir)
    candidates = sorted(
        path
        for path in protenix_dump_dir.rglob("*.safetensors")
        if "_SAMPLE_" not in path.name
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected exactly one full Protenix safetensors under {protenix_dump_dir}, found {len(candidates)}")
    return candidates[0]


def load_precomputed_embedding(path: Path) -> dict[str, torch.Tensor]:
    data = load_file(path)
    required = {"s", "esm_embedding", "a_token", "residue_index", "asym_id"}
    missing = sorted(required - set(data))
    if missing:
        raise RuntimeError(f"Protenix dump {path} is missing keys: {missing}")
    return data


def build_feature(
    processor: ProteoR1UnderstandProcessor,
    chain_specs: Sequence[dict[str, str]],
    embedding: dict[str, torch.Tensor],
    design_points: str | None,
    record_id: str,
    model_path: Path,
    predicted_cdr_sequences: Mapping[str, str],
) -> dict[str, Any]:
    chain_ids, chain_counts = _chain_context(chain_specs, embedding)
    encoder = AntibodySftEncoder(processor)
    text = encoder.encode(
        input_text=build_input_text(design_points),
        target_text=build_target_text(
            chain_specs,
            record_id=record_id,
            predicted_cdr_sequences=predicted_cdr_sequences,
        ),
        chain_ids=chain_ids,
        chain_token_counts=chain_counts,
    )
    text.update(
        {
            "protenix_s_embedding": embedding["s"],
            "protenix_esm_embedding": embedding["esm_embedding"],
            "protenix_a_token": embedding["a_token"],
            "protenix_embedding_attention_mask": torch.ones(embedding["s"].shape[0], dtype=torch.long),
            "protenix_residue_index": embedding["residue_index"],
            "protenix_asym_id": embedding["asym_id"],
            "metadata": {
                "name": f"{record_id}_SAMPLE_0",
                "record_id": record_id,
                "model_path": str(model_path),
            },
        }
    )
    return text


def build_generation_feature(
    processor: ProteoR1UnderstandProcessor,
    chain_specs: Sequence[dict[str, str]],
    embedding: dict[str, torch.Tensor],
    design_points: str | None,
    record_id: str,
    model_path: Path,
) -> dict[str, Any]:
    chain_ids, chain_counts = _chain_context(chain_specs, embedding)
    encoder = AntibodySftEncoder(processor)
    text = encoder.encode_generation_prompt(
        input_text=build_input_text(design_points),
        chain_ids=chain_ids,
        chain_token_counts=chain_counts,
    )
    text.update(
        {
            "protenix_s_embedding": embedding["s"],
            "protenix_esm_embedding": embedding["esm_embedding"],
            "protenix_a_token": embedding["a_token"],
            "protenix_embedding_attention_mask": torch.ones(embedding["s"].shape[0], dtype=torch.long),
            "protenix_residue_index": embedding["residue_index"],
            "protenix_asym_id": embedding["asym_id"],
            "metadata": {
                "name": f"{record_id}_SAMPLE_0",
                "record_id": record_id,
                "model_path": str(model_path),
            },
        }
    )
    return text


def _decoded_sample(text: str, limit: int = 600) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _load_generated_json(decoded_text: str) -> dict[str, Any]:
    if not decoded_text.strip():
        raise RuntimeError("model.generate output was empty or all padding")
    start = decoded_text.find("{")
    if start < 0:
        raise RuntimeError(f"model.generate output contained no JSON object; decoded_sample={_decoded_sample(decoded_text)}")
    try:
        payload, _end = json.JSONDecoder().raw_decode(decoded_text[start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"model.generate output JSON parse failed; decoded_sample={_decoded_sample(decoded_text)}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"model.generate JSON root was not an object; decoded_sample={_decoded_sample(decoded_text)}")
    return payload


def _strip_segment_tags(seq_text: str, segment: CdrSegment, decoded_text: str) -> str:
    start_tag = f"<{segment.tag}>"
    end_tag = f"</{segment.tag}>"
    stripped = seq_text.strip()
    if stripped.startswith(start_tag) and stripped.endswith(end_tag):
        inner = stripped[len(start_tag) : -len(end_tag)]
    else:
        match = re.match(r"<([A-Z]+\d?)>([A-Z]*)</([A-Z]+\d?)>", stripped)
        inner = match.group(2) if match else re.sub(r"</?[A-Z]+\d?>", "", stripped)
    return "".join(aa for aa in inner.upper() if aa in AMINO_ACIDS)


def parse_predicted_cdr_sequences(decoded_text: str, expected_segments: Sequence[CdrSegment]) -> dict[str, str]:
    payload = _load_generated_json(decoded_text)
    answer = payload.get("answer")
    if not isinstance(answer, dict):
        raise RuntimeError(
            "model.generate JSON missed answer; "
            f"decoded_sample={_decoded_sample(decoded_text)}"
        )
    # default ckpts emit nested `answer.cdr_sequences.<label>.seq`; some
    # checkpoints emit direct keys `answer.<label>` (either a string or a
    # `{"seq": ...}` dict). Accept both.
    cdr_sequences = answer.get("cdr_sequences") if isinstance(answer.get("cdr_sequences"), dict) else {}

    def _resolve_seq_text(label: str) -> str | None:
        nested = cdr_sequences.get(label)
        if isinstance(nested, dict) and isinstance(nested.get("seq"), str):
            return nested["seq"]
        if isinstance(nested, str):
            return nested
        direct = answer.get(label)
        if isinstance(direct, dict) and isinstance(direct.get("seq"), str):
            return direct["seq"]
        if isinstance(direct, str):
            return direct
        return None

    predicted: dict[str, str] = {}
    for segment in expected_segments:
        seq_text = _resolve_seq_text(segment.label)
        if not isinstance(seq_text, str):
            continue
        seq = _strip_segment_tags(seq_text, segment, decoded_text)
        if seq:
            predicted[segment.label] = seq

    for segment in expected_segments:
        if segment.label not in predicted:
            continue
        pred_len = len(predicted[segment.label])
        expected_len = segment.expected_len
        if pred_len == 0 or pred_len > expected_len * 2 or pred_len < expected_len * 0.5:
            raise RuntimeError(
                f"predicted {segment.label} length {pred_len} is wildly different from "
                f"mask span {expected_len}; decoded_sample={_decoded_sample(decoded_text)}"
            )

    return {segment.label: predicted[segment.label] for segment in expected_segments if segment.label in predicted}


def complete_cdr_sequences_for_teacher_forcing(
    predicted_cdr_sequences: Mapping[str, str],
    expected_segments: Sequence[CdrSegment],
) -> dict[str, str]:
    """Fill missing generated CDRs with an X-string of the expected length.

    Using ``X`` (unknown-amino-acid placeholder) avoids leaking ground-truth
    CDR identity into the teacher-forced hidden states that condition generation
    diffusion downstream. Length is preserved so the prompt template + token
    alignment stay intact even when model.generate omits a CDR label.
    """
    return {
        segment.label: predicted_cdr_sequences.get(segment.label, "X" * segment.expected_len)
        for segment in expected_segments
    }


def seed_generation(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_predicted_cdr_sequences(
    model: ProteoR1UnderstandModel,
    processor: ProteoR1UnderstandProcessor,
    feature: dict[str, Any],
    expected_segments: Sequence[CdrSegment],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
) -> tuple[dict[str, str], str, str]:
    features, _original_size = pad_batch_if_needed([feature], min_batch_size=max(1, batch_size))
    batch = collate_precomputed(features)
    batch.pop("metadata", None)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype if device.type != "cpu" else None
    batch = prepare_batch_for_model(batch, device=device, dtype=dtype)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id

    generation_config = model.generation_config
    generation_config.max_new_tokens = max_new_tokens
    generation_config.temperature = temperature
    generation_config.top_p = top_p
    generation_config.top_k = top_k

    valid_input_tokens = batch["input_ids"][0][batch["attention_mask"][0].bool()]
    prompt_text = processor.tokenizer.decode(valid_input_tokens, skip_special_tokens=False)
    prompt_text = re.sub(r"(<protein>)+", "<protein>", prompt_text)

    seed_generation(seed)
    with torch.no_grad():
        generated_ids = model.generate(
            **batch,
            do_sample=True,
            pad_token_id=pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_tokens = generated_ids[0, batch["input_ids"].shape[1] :]
    if pad_token_id is not None:
        generated_tokens = generated_tokens[generated_tokens != pad_token_id]
    decoded_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=False)
    return parse_predicted_cdr_sequences(decoded_text, expected_segments), decoded_text, prompt_text


def pad_batch_if_needed(features: list[dict[str, Any]], min_batch_size: int = 2) -> tuple[list[dict[str, Any]], int]:
    original_size = len(features)
    if original_size >= min_batch_size:
        return features, original_size
    if original_size == 0:
        raise ValueError("cannot pad an empty feature list")
    padded = list(features)
    while len(padded) < min_batch_size:
        padded.append({key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in features[-1].items()})
    return padded, original_size


def collate_precomputed(features: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from torch.nn.utils.rnn import pad_sequence

    tensor_1d_keys = [
        "input_ids",
        "labels",
        "attention_mask",
        "position_ids",
        "chain_type_ids",
        "cdr_region_type_ids",
        "protenix_residue_index",
        "protenix_asym_id",
        "protenix_embedding_attention_mask",
    ]
    batch: dict[str, Any] = {}
    for key in tensor_1d_keys:
        if not all(key in feature for feature in features):
            continue
        pad_value = -100 if key == "labels" else (-1 if key in {"chain_type_ids", "cdr_region_type_ids"} else 0)
        batch[key] = pad_sequence([feature[key] for feature in features], batch_first=True, padding_value=pad_value)

    for key in ["protenix_s_embedding", "protenix_esm_embedding", "protenix_a_token"]:
        tensors = [feature[key] for feature in features]
        max_len = max(tensor.shape[0] for tensor in tensors)
        shape = (len(tensors), max_len, tensors[0].shape[1])
        padded = torch.zeros(shape, dtype=tensors[0].dtype)
        for idx, tensor in enumerate(tensors):
            padded[idx, : tensor.shape[0]] = tensor
        batch[key] = padded

    batch["metadata"] = [feature["metadata"] for feature in features]
    return batch


def get_amino_acid_token_ids(processor: ProteoR1UnderstandProcessor) -> list[int]:
    token_ids: list[int] = []
    for aa in AMINO_ACIDS:
        ids = processor.tokenizer.encode(aa, add_special_tokens=False)
        if ids:
            token_ids.append(ids[0])
    return token_ids


def compute_cdr_confidence(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    cdr_indices: torch.Tensor,
    aa_token_ids: list[int],
) -> torch.Tensor:
    import torch.nn.functional as F

    n_cdr = len(cdr_indices)
    confidence = torch.zeros(n_cdr, device=logits.device, dtype=torch.float32)
    prev_indices = cdr_indices - 1
    valid_prev_mask = prev_indices >= 0
    if not valid_prev_mask.any():
        confidence.fill_(-1.0)
        return confidence

    safe_prev_indices = prev_indices.clamp(min=0)
    aa_indices = torch.tensor(aa_token_ids, device=logits.device, dtype=torch.long)
    aa_logits = logits[safe_prev_indices][:, aa_indices]
    aa_probs = F.softmax(aa_logits.float(), dim=-1)
    cdr_token_ids = input_ids[cdr_indices]
    token_to_aa_idx = {token_id: idx for idx, token_id in enumerate(aa_token_ids)}
    aa_positions = torch.tensor(
        [token_to_aa_idx.get(token_id.item(), -1) for token_id in cdr_token_ids],
        device=logits.device,
        dtype=torch.long,
    )
    valid_aa_mask = aa_positions >= 0
    combined_valid_mask = valid_prev_mask & valid_aa_mask
    if combined_valid_mask.any():
        valid_indices = torch.arange(n_cdr, device=logits.device)[combined_valid_mask]
        confidence[combined_valid_mask] = aa_probs[valid_indices, aa_positions[combined_valid_mask]]
    confidence[~combined_valid_mask] = -1.0
    return confidence


def extract_cdr_hidden_states_for_sample(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    chain_type_ids: torch.Tensor,
    cdr_region_type_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    aa_token_ids: list[int],
) -> dict[str, torch.Tensor] | None:
    valid_mask = attention_mask.bool()
    cdr_mask = (cdr_region_type_ids > 0) & valid_mask
    if not cdr_mask.any():
        return None

    cdr_token_indices = torch.where(cdr_mask)[0]
    cdr_confidence = compute_cdr_confidence(
        logits=logits,
        input_ids=input_ids,
        cdr_indices=cdr_token_indices,
        aa_token_ids=aa_token_ids,
    )
    return {
        "cdr_hidden_states": hidden_states[cdr_mask].detach().cpu().contiguous(),
        "cdr_chain_type": chain_type_ids[cdr_mask].detach().cpu().long().contiguous(),
        "cdr_region_type": cdr_region_type_ids[cdr_mask].detach().cpu().long().contiguous(),
        "cdr_token_indices": cdr_token_indices.detach().cpu().long().contiguous(),
        "cdr_confidence": cdr_confidence.detach().cpu().float().contiguous(),
    }


def save_cdr_hidden_states(output_path: Path, cdr_data: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(cdr_data, output_path, metadata=metadata)


def process_batch(
    model: ProteoR1UnderstandModel,
    batch: dict[str, Any],
    output_dir: Path,
    model_path: Path,
    aa_token_ids: list[int],
    original_batch_size: int,
    sample_idx: int,
    force: bool,
    extra_metadata: Mapping[str, str] | None = None,
) -> Path:
    metadata_list = batch.pop("metadata")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype if device.type != "cpu" else None
    batch = prepare_batch_for_model(batch, device=device, dtype=dtype)
    chain_type_ids = batch.pop("chain_type_ids")
    cdr_region_type_ids = batch.pop("cdr_region_type_ids")

    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True, use_cache=False)

    output_path: Path | None = None
    last_hidden_states = outputs.hidden_states[-1]
    for index in range(original_batch_size):
        meta = metadata_list[index]
        record_id = meta["record_id"]
        output_path = output_dir / f"{record_id}_SAMPLE_{sample_idx}.safetensors"
        if output_path.exists() and not force:
            raise FileExistsError(f"output exists; pass --override to replace: {output_path}")
        cdr_data = extract_cdr_hidden_states_for_sample(
            hidden_states=last_hidden_states[index],
            logits=outputs.logits[index],
            input_ids=batch["input_ids"][index],
            chain_type_ids=chain_type_ids[index],
            cdr_region_type_ids=cdr_region_type_ids[index],
            attention_mask=batch["attention_mask"][index],
            aa_token_ids=aa_token_ids,
        )
        if cdr_data is None:
            raise RuntimeError(f"No CDR tokens found in teacher-forced text for {record_id}")
        metadata = {
            "sample_id": f"{record_id}_SAMPLE_{sample_idx}",
            "record_id": record_id,
            "n_total_tokens": str(int(batch["attention_mask"][index].sum().item())),
            "n_cdr_tokens": str(int(cdr_data["cdr_hidden_states"].shape[0])),
            "hidden_dim": str(int(cdr_data["cdr_hidden_states"].shape[1])),
            "model_path": str(model_path),
        }
        if extra_metadata is not None:
            metadata.update(extra_metadata)
        save_cdr_hidden_states(output_path, cdr_data, metadata)
    if output_path is None:
        raise RuntimeError("empty batch")
    return output_path


def load_processor(checkpoint: Path) -> ProteoR1UnderstandProcessor:
    processor = ProteoR1UnderstandProcessor.from_pretrained(checkpoint)
    missing_cdr_tokens = [token for token in CDR_TAGS if token not in processor.tokenizer.get_vocab()]
    if missing_cdr_tokens:
        processor.tokenizer.add_tokens(missing_cdr_tokens)
        LOGGER.warning("Added missing CDR tag tokens to tokenizer: %s", ", ".join(missing_cdr_tokens))
    return processor


def load_model(checkpoint: Path, device: str) -> ProteoR1UnderstandModel:
    model = ProteoR1UnderstandModel.from_pretrained(checkpoint, low_cpu_mem_usage=True)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    torch_device = torch.device(device)
    dtype = torch.bfloat16 if torch_device.type == "cuda" else torch.float32
    model.to(device=torch_device, dtype=dtype)
    model.eval()
    return model


def emit_cdr_hidden_states(
    input_dir: str | Path,
    protenix_dump_dir: str | Path,
    checkpoint: str | Path = "pretrained/proteor1_understand",
    output: str | Path | None = None,
    record_id: str | None = None,
    design_points: str | None = None,
    masked_json: str | Path | None = None,
    sample_idx: int = 0,
    device: str = "cuda",
    batch_size: int = 2,
    max_new_tokens: int = GENERATION_MAX_NEW_TOKENS,
    temperature: float = GENERATION_TEMPERATURE,
    top_p: float = GENERATION_TOP_P,
    top_k: int = GENERATION_TOP_K,
    seed: int = DEFAULT_GENERATION_SEED,
    override: bool = False,
    dry_run: bool = False,
) -> Path | None:
    args = argparse.Namespace(
        input_dir=Path(input_dir),
        protenix_dump_dir=Path(protenix_dump_dir),
        checkpoint=Path(checkpoint),
        output=Path(output) if output is not None else None,
        record_id=record_id,
        design_points=design_points,
        masked_json=Path(masked_json) if masked_json is not None else None,
        sample_idx=sample_idx,
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        override=override,
    )
    resolved = resolve_inputs(args)
    if dry_run:
        locate_protenix_dump_file(resolved.protenix_dump_path)
        load_chain_specs(resolved.yaml_path)
        load_masked_protein_json(resolved.masked_json_path)
        return None

    if resolved.output_dir.exists() and resolved.override:
        shutil.rmtree(resolved.output_dir)
    resolved.output_dir.mkdir(parents=True, exist_ok=True)

    processor = load_processor(resolved.checkpoint)
    dump_file = locate_protenix_dump_file(resolved.protenix_dump_path)
    embedding = load_precomputed_embedding(dump_file)
    chain_specs = load_chain_specs(resolved.yaml_path)
    expected_segments = collect_cdr_segments(chain_specs, resolved.record_id)
    model = load_model(resolved.checkpoint, resolved.device)
    generation_feature = build_generation_feature(
        processor=processor,
        chain_specs=chain_specs,
        embedding=embedding,
        design_points=resolved.design_points,
        record_id=resolved.record_id,
        model_path=resolved.checkpoint,
    )
    generated_cdr_sequences, decoded_text, prompt_text = generate_predicted_cdr_sequences(
        model=model,
        processor=processor,
        feature=generation_feature,
        expected_segments=expected_segments,
        batch_size=resolved.batch_size,
        max_new_tokens=resolved.max_new_tokens,
        temperature=resolved.temperature,
        top_p=resolved.top_p,
        top_k=resolved.top_k,
        seed=resolved.seed,
    )
    missing_generated_cdrs = [
        segment.label for segment in expected_segments if segment.label not in generated_cdr_sequences
    ]
    teacher_forced_cdr_sequences = complete_cdr_sequences_for_teacher_forcing(
        generated_cdr_sequences,
        expected_segments,
    )
    feature = build_feature(
        processor=processor,
        chain_specs=chain_specs,
        embedding=embedding,
        design_points=resolved.design_points,
        record_id=resolved.record_id,
        model_path=resolved.checkpoint,
        predicted_cdr_sequences=teacher_forced_cdr_sequences,
    )
    features, original_size = pad_batch_if_needed([feature], min_batch_size=max(1, resolved.batch_size))
    batch = collate_precomputed(features)
    return process_batch(
        model=model,
        batch=batch,
        output_dir=resolved.output_dir,
        model_path=resolved.checkpoint,
        aa_token_ids=get_amino_acid_token_ids(processor),
        original_batch_size=original_size,
        sample_idx=resolved.sample_idx,
        force=resolved.override,
        extra_metadata={
            "target_sequence_source": "model.generate"
            if not missing_generated_cdrs
            else "model.generate_with_x_placeholder_fallback",
            "generated_cdr_lengths": json.dumps(
                {label: len(seq) for label, seq in generated_cdr_sequences.items()},
                sort_keys=True,
            ),
            "generated_cdr_sequences": json.dumps(generated_cdr_sequences, sort_keys=True),
            "missing_generated_cdrs": json.dumps(missing_generated_cdrs),
            "teacher_forced_cdr_lengths": json.dumps(
                {label: len(seq) for label, seq in teacher_forced_cdr_sequences.items()},
                sort_keys=True,
            ),
            "generated_decode_chars": str(len(decoded_text)),
            "generated_response_sample": _decoded_sample(decoded_text),
            "generation_prompt_sample": _decoded_sample(prompt_text),
            "generation_seed": str(resolved.seed),
            "generation_config": json.dumps(
                {
                    "temperature": resolved.temperature,
                    "top_p": resolved.top_p,
                    "top_k": resolved.top_k,
                    "max_new_tokens": resolved.max_new_tokens,
                    "do_sample": True,
                },
                sort_keys=True,
            ),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    output = emit_cdr_hidden_states(
        input_dir=args.input_dir,
        protenix_dump_dir=args.protenix_dump_dir,
        checkpoint=args.checkpoint,
        output=args.output,
        record_id=args.record_id,
        design_points=args.design_points,
        masked_json=args.masked_json,
        sample_idx=args.sample_idx,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        override=args.override,
        dry_run=args.dry_run,
    )
    if output is not None:
        print(json.dumps({"cdr_hidden": str(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
