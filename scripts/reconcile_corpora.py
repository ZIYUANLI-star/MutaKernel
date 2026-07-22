#!/usr/bin/env python3
"""Stable-ID reconciliation and collection-frame construction for C2-C5.

Blueprint Table 1: C4/C5 counts are "pending stable-ID reconciliation of the
collection frames"; all corpora must be frozen with content-addressed stable
IDs *before* any V2 validator outcome is observed.  This script builds, per
corpus, a collection frame JSONL where every candidate kernel carries:

  * ``stable_id``   — sha256 over the whitespace-normalized candidate source
                      (the content address; byte-level duplicates collapse),
  * ``raw_sha256``  — sha256 over the raw bytes (audit trail),
  * ``task_key``    — the underlying benchmark task (clustering unit for all
                      statistical inference, §5.1.5),
  * ``accepted``    — the shipping validator's own verdict where the corpus
                      records one (C3 ``Correct``, C5 ``correct``); C2/C4
                      publish only accepted kernels, marked ``true``,
  * duplicate flags — byte-identical candidates *across* corpora are
                      deduplicated and flagged, never silently dropped.

Corpora (paths relative to --external-root, default ``external/``):
  C2  CUDA-L1               optimized_cuda_code/a100.json  (JSONL)
  C3  AI-CUDA-Engineer      level_{1,2,3}.parquet          (needs pyarrow)
  C4  TritonBench-G         data/TritonBench_G_v1.json     (184 entries)
  C5  KernelBench-samples   baseline_eval/level*/<model>/problem_*/sample_*/kernel.json

Runs CPU-only, executes no kernel code.  ``--materialize`` additionally
writes each unique candidate to ``<out>/<corpus>/candidates/<stable_id>.py``
so the FSE subject manifest can reference frozen files.

Usage:
  python scripts/reconcile_corpora.py --external-root external \
      --out-dir MutakernelV2/实验/补充实验数据/collection_frames \
      --corpora C2 C3 C4 C5 [--materialize]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

FRAME_SCHEMA_VERSION = "1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_source(source: str) -> str:
    """Deterministic whitespace normalization for content addressing.

    Only trailing per-line whitespace and leading/trailing blank lines are
    stripped — semantic content is untouched, so two byte-different but
    normalized-identical candidates are genuinely the same program text.
    """
    lines = [line.rstrip() for line in source.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def stable_id(source: str) -> str:
    return hashlib.sha256(normalize_source(source).encode("utf-8")).hexdigest()


def collect_c2(root: Path):
    path = root / "C2_cuda_l1" / "optimized_cuda_code" / "a100.json"
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source = record["custom_code"]
            rows.append({
                "corpus": "C2",
                "dataset": "cuda_l1",
                "language": "cuda",
                "task_key": f"KB_L{record['level_id']}_P{record['task_id']}",
                "origin": f"a100.json:L{record['level_id']}/T{record['task_id']}",
                "accepted": True,
                "source": source,
                "reference_source": record.get("ref_code"),
            })
    return rows, {"input": str(path)}


def collect_c3(root: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "C3 requires pyarrow (pip install pyarrow); "
            "run with --corpora excluding C3 until installed") from exc
    rows = []
    inputs = []
    for level in (1, 2, 3):
        path = root / "C3_ai_cuda_engineer" / f"level_{level}.parquet"
        if not path.exists():
            continue
        inputs.append(str(path))
        table = pq.read_table(path, columns=[
            "Op_Name", "Level_ID", "Task_ID", "CUDA_Code", "Correct"])
        data = table.to_pylist()
        for record in data:
            source = record.get("CUDA_Code") or ""
            if not source.strip():
                continue
            rows.append({
                "corpus": "C3",
                "dataset": "ai_cuda_engineer",
                "language": "cuda",
                "task_key": f"KB_L{record['Level_ID']}_P{record['Task_ID']}",
                "origin": f"level_{level}.parquet:{record.get('Op_Name', '')}",
                "accepted": bool(record.get("Correct")),
                "source": source,
            })
    return rows, {"inputs": inputs}


def collect_c4(root: Path):
    manifest = root / "C4_tritonbench" / "data" / "TritonBench_G_v1.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    kernels_dir = root / "C4_tritonbench" / "data" / "TritonBench_G_v1"
    rows = []
    for record in data:
        file_name = record["file"]
        kernel_file = kernels_dir / file_name
        source = (
            kernel_file.read_text(encoding="utf-8", errors="replace")
            if kernel_file.exists() else record.get("output", "")
        )
        rows.append({
            "corpus": "C4",
            "dataset": "tritonbench_g",
            "language": "triton",
            "task_key": f"TBG_{Path(file_name).stem}",
            "origin": f"TritonBench_G_v1/{file_name}",
            "accepted": True,
            "source": source,
        })
    return rows, {"input": str(manifest), "manifest_entries": len(data)}


def collect_c5(root: Path):
    base = root / "C5_kernelbench_samples" / "baseline_eval"
    rows = []
    files = sorted(base.rglob("kernel.json"))
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "corpus": "C5",
            "dataset": "kernelbench_samples",
            "language": "cuda",
            "task_key": f"KB_L{record['level']}_P{record['problem_id']}",
            "origin": str(path.relative_to(base)),
            "accepted": bool(record.get("correct")),
            "generator_model": record.get("model_name"),
            "source": record["kernel"],
        })
    return rows, {"input": str(base), "files": len(files)}


COLLECTORS = {"C2": collect_c2, "C3": collect_c3, "C4": collect_c4, "C5": collect_c5}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--external-root", type=Path, default=PROJECT_ROOT / "external")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--corpora", nargs="+", default=["C2", "C3", "C4", "C5"],
                    choices=sorted(COLLECTORS))
    ap.add_argument("--materialize", action="store_true",
                    help="write unique candidate sources to <out>/<corpus>/candidates/")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seen_global = {}  # stable_id -> first (corpus, frame_id)
    summary = {}
    for corpus in args.corpora:
        rows, provenance = COLLECTORS[corpus](args.external_root)
        frame_path = args.out_dir / f"{corpus}_collection_frame.jsonl"
        candidates_dir = args.out_dir / corpus / "candidates"
        if args.materialize:
            candidates_dir.mkdir(parents=True, exist_ok=True)

        counts = Counter()
        seen_in_corpus = set()
        with open(frame_path, "w", encoding="utf-8") as fh:
            for index, row in enumerate(rows):
                source = row.pop("source")
                reference_source = row.pop("reference_source", None)
                if not source or not str(source).strip():
                    counts["skipped_empty_source"] += 1
                    continue
                sid = stable_id(source)
                frame_id = f"{corpus}-{index:05d}"
                duplicate_of = seen_global.get(sid)
                entry = {
                    "frame_schema_version": FRAME_SCHEMA_VERSION,
                    "frame_id": frame_id,
                    "stable_id": sid,
                    "raw_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    **row,
                    "duplicate_within_corpus": sid in seen_in_corpus,
                    "duplicate_of": (
                        None if duplicate_of is None or duplicate_of[1] == frame_id
                        else {"corpus": duplicate_of[0], "frame_id": duplicate_of[1]}),
                    "source_bytes": len(source.encode("utf-8")),
                }
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
                counts["total"] += 1
                counts["accepted"] += bool(row.get("accepted"))
                counts["dup_within"] += sid in seen_in_corpus
                counts["dup_cross"] += (
                    duplicate_of is not None and duplicate_of[0] != corpus)
                seen_in_corpus.add(sid)
                seen_global.setdefault(sid, (corpus, frame_id))
                if args.materialize and not entry["duplicate_within_corpus"]:
                    (candidates_dir / f"{sid}.py").write_text(source, encoding="utf-8")
                    if reference_source:
                        ref_dir = args.out_dir / corpus / "references"
                        ref_dir.mkdir(parents=True, exist_ok=True)
                        ref_sid = stable_id(reference_source)
                        (ref_dir / f"{ref_sid}.py").write_text(
                            reference_source, encoding="utf-8")

        unique = len(seen_in_corpus)
        summary[corpus] = {
            "provenance": provenance,
            "rows": counts["total"],
            "skipped_empty_source": counts["skipped_empty_source"],
            "unique_stable_ids": unique,
            "accepted": counts["accepted"],
            "duplicates_within_corpus": counts["dup_within"],
            "duplicates_cross_corpus": counts["dup_cross"],
            "frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
            "frame_path": str(frame_path),
        }
        print(f"[{corpus}] rows={counts['total']} unique={unique} "
              f"accepted={counts['accepted']} dup_within={counts['dup_within']} "
              f"dup_cross={counts['dup_cross']}", flush=True)

    freeze = {
        "created_at": _now(),
        "frame_schema_version": FRAME_SCHEMA_VERSION,
        "normalization": "strip trailing per-line whitespace + outer blank lines",
        "corpora": summary,
        "note": "freeze these digests before observing any V2 validator outcome",
    }
    freeze_path = args.out_dir / "collection_frames.freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps({c: {k: v for k, v in s.items() if k != "provenance"}
                      for c, s in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
