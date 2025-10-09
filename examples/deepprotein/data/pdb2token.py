# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-----------------------
What it does:
  1) build_worklist         : create a list of UniProt IDs to process
  2) download_af2_structures: download AlphaFold PDBs for those IDs
  3) build_foldseek_3di     : generate 3Di tokens using foldseek
  4) assemble_sft_jsonl     : assemble final JSONL from intermediate pieces
  5) show_progress          : quick status summary
  6) run_all                : run the full pipeline

Notes:
- Requires `foldseek` in PATH for 3Di generation.
- AlphaFold PDBs are fetched from EBI; we try v4 then v3.
- Concurrency is exposed via `--jobs`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("pdb2token")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass
class WorkItem:
    uniprot_id: str
    pdb_path: Path
    fasta_path: Path
    three_di_path: Path


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
AF2_URL_PATTERNS = ["https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v6.pdb", "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v5.pdb",
                    "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb", "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v3.pdb",
                    "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v2.pdb", "https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v1.pdb", ]

UID_RE = re.compile(r"^[A-Z0-9]{1,10}$", re.IGNORECASE)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def write_jsonl(records: Iterable[dict], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def _normalize_uid(uid: str) -> str:
    uid = uid.strip()
    # strip UniProt-style prefixes and isoform suffixes
    if '|' in uid:
        uid = uid.split('|')[-1]
    uid = uid.split('-')[0]
    return uid.upper()


def _dest_for(out_path: Path, base_uid: str, url: str) -> Path:
    # If out_path looks like a directory (no suffix), write canonical AF filename inside it
    if out_path.suffix == "":
        fname = url.rsplit("/", 1)[-1]  # e.g., AF-Q39E95-F1-model_v2.pdb
        return out_path / fname
    # If caller passed a file path, use it as-is (but fix extension to match url)
    ext = url.rsplit(".", 1)[-1]
    if out_path.suffix.lstrip(".").lower() != ext.lower():
        return out_path.with_suffix("." + ext)
    return out_path


def _download_one(uid: str, out_path: Path, timeout: int = 60, retry: int = 2) -> Tuple[str, bool, str]:
    base_uid = _normalize_uid(uid)

    # allow dirs or files
    if out_path.suffix == "":
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    last_msg = "unknown_error"
    for attempt in range(retry + 1):
        all_404 = True
        for tmpl in AF2_URL_PATTERNS:
            url = tmpl.format(uid=base_uid)
            dest = _dest_for(out_path, base_uid, url)
            try:
                req = Request(url, headers={"User-Agent": "pdb2token/1.1"})
                with urlopen(req, timeout=timeout) as resp:
                    if resp.status != 200:
                        last_msg = f"http_{resp.status}"
                        if resp.status != 404:
                            all_404 = False
                        continue
                    data = resp.read()
                with dest.open("wb") as f:
                    f.write(data)
                return uid, True, f"ok:{dest.name}"
            except HTTPError as e:
                last_msg = f"http_error_{e.code}"
                if e.code != 404:
                    all_404 = False
            except URLError as e:
                last_msg = f"url_error_{getattr(e, 'reason', 'unknown')}"
                all_404 = False
            except Exception as e:
                last_msg = f"error_{type(e).__name__}:{e}"
                all_404 = False
        if all_404:
            return uid, False, "not_found_v4_v3_v2_v1"
        if attempt < retry:
            time.sleep(1.0 * (attempt + 1))
    return uid, False, last_msg


def build_worklist(ids_file: Path, out_dir: Path) -> List[WorkItem]:
    """
    Create WorkItem stubs pointing to where each artifact would live.
    """
    ids = read_lines(ids_file)
    items: List[WorkItem] = []
    for uid in ids:
        pdb = out_dir / "pdb" / f"{uid}.pdb"
        fasta = out_dir / "fasta" / f"{uid}.fasta"
        three_di = out_dir / "3di" / f"{uid}.3di"
        items.append(WorkItem(uid, pdb, fasta, three_di))
    logger.info("Prepared %d work items", len(items))
    return items


def download_af2_structures(ids_file: Path, out_dir: Path, jobs: int = 8) -> None:
    items = build_worklist(ids_file, out_dir)
    todo = [(wi.uniprot_id, wi.pdb_path) for wi in items if not wi.pdb_path.exists()]
    if not todo:
        logger.info("All PDBs already present; nothing to do.")
        return

    logger.info("Downloading %d PDBs from AlphaFold (jobs=%d)...", len(todo), jobs)
    ok, fail = 0, 0
    with cf.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
        futs = [ex.submit(_download_one, uid, path) for uid, path in todo]
        for fut in cf.as_completed(futs):
            uid, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                logger.warning("Failed %s: %s", uid, msg)
    logger.info("Download done. success=%d fail=%d", ok, fail)


def build_foldseek_3di(pdb_dir: Path, out_dir: Path, jobs: int = 8) -> None:
    """
    Use foldseek to compute 3Di tokens for all PDBs in pdb_dir.
    Produces .3di files in out_dir/3di/
    """
    foldseek_bin = which("foldseek")
    if not foldseek_bin:
        raise RuntimeError("foldseek not found in PATH. Install it first: https://github.com/steineggerlab/foldseek")

    pdb_dir = Path(pdb_dir)
    out_dir = Path(out_dir)
    db_dir = out_dir / "foldseek_db"
    three_di_dir = out_dir / "3di"
    ensure_dir(db_dir)
    ensure_dir(three_di_dir)

    # 1) createdb
    logger.info("Building foldseek DB from PDBs in %s ...", pdb_dir)
    createdb_cmd = [foldseek_bin, "createdb", str(pdb_dir), str(db_dir / "struct"), "--dbtype", "pdb"]
    subprocess.run(createdb_cmd, check=True)

    # 2) convert to 3Di (foldseek lndb or tsv via easy-search pipeline); we'll use `easy-search` to get 3Di aln
    #    but for pure 3Di strings, use `align` with --format-output. Here we extract per-entry 3Di via `convertalis`.
    tmp_dir = out_dir / "tmp"
    ensure_dir(tmp_dir)

    search_cmd = [foldseek_bin, "easy-search", str(db_dir / "struct"), str(db_dir / "struct"), str(tmp_dir / "aln.m8"), str(tmp_dir), "--format-output",
                  "query,target,qcov,tcov,qaln,qaln3di,taln,taln3di", "--threads", str(max(1, jobs))]
    logger.info("Running foldseek easy-search to produce 3Di tokens...")
    subprocess.run(search_cmd, check=True)

    # 3) Parse the output and write 3Di strings. For self-self matches, qaln3di is what we want.
    aln_path = tmp_dir / "aln.m8"
    if not aln_path.exists():
        raise RuntimeError(f"foldseek output missing: {aln_path}")

    # m8 with custom columns; we keep only lines where query==target and dump qaln3di
    written = 0
    with aln_path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            # columns we requested: query,target,qcov,tcov,qaln,qaln3di,taln,taln3di
            if len(parts) < 8:
                continue
            q, t, *_rest = parts
            qaln3di = parts[5]
            if q == t and qaln3di:
                uid = Path(q).stem
                out_path = three_di_dir / f"{uid}.3di"
                with out_path.open("w", encoding="utf-8") as g:
                    g.write(qaln3di + "\n")
                written += 1

    logger.info("Wrote %d 3Di files to %s", written, three_di_dir)


def assemble_sft_jsonl(three_di_dir: Path, out_path: Path, meta_dir: Optional[Path] = None) -> None:
    """
    Assemble simple JSONL records with:
      {
        "id": "<uid>",
        "three_di": "<3di string>",
        "meta": {...}  # optional
      }
    If meta_dir is provided and contains <uid>.json, it will be embedded.
    """
    three_di_dir = Path(three_di_dir)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    recs = []
    cnt = 0
    for fp in sorted(three_di_dir.glob("*.3di")):
        uid = fp.stem
        three_di = fp.read_text(encoding="utf-8").strip()
        rec = {"id": uid, "three_di": three_di}
        if meta_dir:
            meta_fp = Path(meta_dir) / f"{uid}.json"
            if meta_fp.exists():
                try:
                    rec["meta"] = json.loads(meta_fp.read_text(encoding="utf-8"))
                except Exception as e:  # noqa
                    logger.warning("Bad meta for %s: %s", uid, e)
        recs.append(rec)
        cnt += 1
        if cnt % 1000 == 0:
            logger.info("Prepared %d records...", cnt)

    write_jsonl(recs, out_path)
    logger.info("Assembled %d records into %s", len(recs), out_path)


def show_progress(root: Path) -> None:
    """
    Print a quick progress summary.
    """
    pdbs = list((root / "pdb").glob("*.pdb"))
    threes = list((root / "3di").glob("*.3di"))
    fastas = list((root / "fasta").glob("*.fasta"))
    logger.info("Progress: PDB=%d | 3Di=%d | FASTA=%d", len(pdbs), len(threes), len(fastas))


def run_all(ids_file: Path, out_dir: Path, jobs: int = 8) -> None:
    download_af2_structures(ids_file, out_dir, jobs=jobs)
    pdb_dir = out_dir / "pdb"
    build_foldseek_3di(pdb_dir=pdb_dir, out_dir=out_dir, jobs=jobs)
    assemble_sft_jsonl(three_di_dir=out_dir / "3di", out_path=out_dir / "sft.jsonl")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pdb2token", description="Pipeline to download AlphaFold PDBs and convert to 3Di tokens (foldseek), then assemble JSONL.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v, -vv).")
    sub = p.add_subparsers(dest="cmd", required=True)

    # worklist
    sp = sub.add_parser("worklist", help="Build worklist from UniProt ID file.")
    sp.add_argument("--ids", type=Path, required=True, help="Path to text file of UniProt IDs (one per line).")
    sp.add_argument("--out", type=Path, required=True, help="Output root directory.")
    sp.set_defaults(func=lambda a: build_worklist(a.ids, a.out))

    # download
    sp = sub.add_parser("download", help="Download AlphaFold PDBs.")
    sp.add_argument("--ids", type=Path, required=True, help="Path to text file of UniProt IDs (one per line).")
    sp.add_argument("--out", type=Path, required=True, help="Output root directory.")
    sp.add_argument("--jobs", type=int, default=8, help="Parallel downloads.")
    sp.set_defaults(func=lambda a: download_af2_structures(a.ids, a.out, jobs=a.jobs))

    # build-3di
    sp = sub.add_parser("build-3di", help="Generate 3Di tokens via foldseek.")
    sp.add_argument("--pdb-dir", type=Path, required=True, help="Directory containing .pdb files.")
    sp.add_argument("--out", type=Path, required=True, help="Output root directory (stores foldseek DB and 3di).")
    sp.add_argument("--jobs", type=int, default=8, help="Threads for foldseek.")
    sp.set_defaults(func=lambda a: build_foldseek_3di(a.pdb_dir, a.out, jobs=a.jobs))

    # assemble
    sp = sub.add_parser("assemble", help="Assemble SFT JSONL from 3Di outputs.")
    sp.add_argument("--three-di-dir", type=Path, required=True, help="Directory with *.3di files.")
    sp.add_argument("--out", type=Path, required=True, help="Output JSONL path.")
    sp.add_argument("--meta-dir", type=Path, default=None, help="Optional directory with per-UID meta JSON.")
    sp.set_defaults(func=lambda a: assemble_sft_jsonl(a.three_di_dir, a.out, a.meta_dir))

    # progress
    sp = sub.add_parser("progress", help="Show quick progress summary.")
    sp.add_argument("--root", type=Path, required=True, help="Pipeline root directory.")
    sp.set_defaults(func=lambda a: show_progress(a.root))

    # all
    sp = sub.add_parser("all", help="Run the full pipeline (download -> 3di -> assemble).")
    sp.add_argument("--ids", type=Path, required=True, help="Path to text file of UniProt IDs (one per line).")
    sp.add_argument("--out", type=Path, required=True, help="Output root directory.")
    sp.add_argument("--jobs", type=int, default=8, help="Parallelism.")
    sp.set_defaults(func=lambda a: run_all(a.ids, a.out, jobs=a.jobs))

    return p


def _set_loglevel(verbosity: int) -> None:
    if verbosity >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbosity == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)


def main(argv: Optional[List[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    _set_loglevel(getattr(args, "verbose", 0))
    try:
        result = args.func(args)
        if result is not None:
            # e.g., worklist returns a list; we don't print it unless verbose
            logger.debug("Result: %s", str(result)[:500])
        return 0
    except subprocess.CalledProcessError as e:
        logger.error("Subprocess failed: %s", e)
        return e.returncode or 1
    except KeyboardInterrupt:
        logger.error("Interrupted.")
        return 130
    except Exception as e:  # noqa
        logger.exception("Fatal error: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
