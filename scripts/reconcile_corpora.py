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

Inclusion rules (E2 reconciliation, v1.0).  A frame row (= one collected
candidate) *enters the study* only if it passes, in order:

  R1 runnable_source   — non-empty candidate kernel source.  Records with an
                         empty payload are collection-time skips (they never
                         were candidates) and are excluded from the frame and
                         from the Table 1 denominator.
  R2 task_resolvable   — the row's ``task_key`` resolves to an underlying
                         benchmark task: ``KB_L{1..3}_P{n}`` within the
                         KernelBench level sizes (L1:100, L2:100, L3:50) for
                         C2/C3/C5; a frozen TritonBench_G_v1.json manifest
                         entry for C4.
  R3 not_duplicate     — not a normalized-source duplicate of an earlier row
                         in the same corpus (first occurrence is kept; later
                         occurrences are excluded with reason ``duplicate``).
                         Cross-corpus duplicates are flagged, never dropped.
  R4 language_declared — static language evidence matches the corpus
                         declaration (C2/C3/C5: CUDA; C4: Triton).

The first failing rule is recorded as ``exclusion_reason``.  For C3 the
Table 1 unit is the *task* (one representative accepted kernel per
KernelBench task, "229 collected"): after the candidate-level pass a subject
frame is built with, per task, the accepted rule-passing candidate with the
lexicographically smallest stable_id (content-addressed; no validator or
performance signal is consulted).

Runs CPU-only, executes no kernel code.  ``--materialize`` additionally
writes each unique candidate to ``<out>/<corpus>/candidates/<stable_id>.py``
so the FSE subject manifest can reference frozen files.

Usage:
  python scripts/reconcile_corpora.py --external-root external \
      --out-dir MutakernelV2/实验/补充实验数据/collection_frames \
      --corpora C2 C3 C4 C5 [--materialize]

The freeze file is merged per corpus, so C3 (needs pyarrow, runs on the
remote host) and C2/C4/C5 (local) may be reconciled in separate invocations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

FRAME_SCHEMA_VERSION = "1.1"
INCLUSION_RULES_VERSION = "1.0"

# Corpora whose Table 1 unit is the underlying benchmark task, not the
# individual candidate: a per-task subject frame is derived after the
# candidate-level pass.
TASK_LEVEL_SUBJECT_CORPORA = {"C3"}

CORPUS_DECLARED_LANGUAGE = {"C2": "cuda", "C3": "cuda", "C4": "triton", "C5": "cuda"}

KB_TASK_RE = re.compile(r"^KB_L(\d)_P(\d+)$")
KB_LEVEL_SIZES = {1: 100, 2: 100, 3: 50}

_TRITON_EVIDENCE = re.compile(r"@triton\.jit|import\s+triton|triton\.language")
_CUDA_EVIDENCE = re.compile(
    r"__global__|__device__\s|load_inline|cpp_extension|cuda_sources|"
    r"CUDAExtension|cudaMemcpy|<<<"
)


def detect_language(source: str) -> str:
    """Static language evidence: 'triton', 'cuda', or 'python_only'.

    Triton evidence takes precedence: a Triton kernel necessarily also
    imports torch, while an embedded-CUDA candidate never imports triton.
    """
    if _TRITON_EVIDENCE.search(source):
        return "triton"
    if _CUDA_EVIDENCE.search(source):
        return "cuda"
    return "python_only"


def task_resolvable(corpus: str, task_key: str) -> bool:
    if corpus in {"C2", "C3", "C5"}:
        match = KB_TASK_RE.match(task_key)
        if not match:
            return False
        level, problem = int(match.group(1)), int(match.group(2))
        return level in KB_LEVEL_SIZES and 1 <= problem <= KB_LEVEL_SIZES[level]
    if corpus == "C4":
        # C4 task keys are minted from the frozen manifest itself, so a
        # non-empty TBG_* key is resolvable by construction.
        return task_key.startswith("TBG_") and len(task_key) > 4
    return bool(task_key)


def inclusion_check(corpus: str, source: str, task_key: str,
                    is_duplicate_within: bool):
    """Apply rules R2-R4 to one frame row (R1 handled at collection time).

    Returns (included: bool, exclusion_reason: str | None,
    detected_language: str).
    """
    detected = detect_language(source)
    if not task_resolvable(corpus, task_key):
        return False, "task_unresolvable", detected
    if is_duplicate_within:
        return False, "duplicate", detected
    if detected != CORPUS_DECLARED_LANGUAGE[corpus]:
        return False, "language_mismatch", detected
    return True, None, detected


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


def build_task_subject_frame(corpus: str, frame_rows: list, out_dir: Path):
    """One accepted, rule-passing representative per benchmark task.

    Representative rule (frozen, content-addressed): among the task's
    accepted candidates with ``included=True``, choose the smallest
    stable_id.  No validator outcome or performance signal is consulted.
    """
    by_task = defaultdict(list)
    tasks_seen = set()
    for row in frame_rows:
        tasks_seen.add(row["task_key"])
        if row.get("accepted") and row.get("included"):
            by_task[row["task_key"]].append(row)

    subject_rows = []
    excluded_tasks = []
    for task_key in sorted(tasks_seen):
        eligible = by_task.get(task_key)
        if not eligible:
            excluded_tasks.append(task_key)
            continue
        representative = min(eligible, key=lambda r: r["stable_id"])
        subject_rows.append({
            "frame_schema_version": FRAME_SCHEMA_VERSION,
            "corpus": corpus,
            "task_key": task_key,
            "stable_id": representative["stable_id"],
            "frame_id": representative["frame_id"],
            "origin": representative["origin"],
            "candidates_eligible": len(eligible),
            "representative_rule": "min_stable_id_over_accepted_included",
        })

    subject_path = out_dir / f"{corpus}_subject_frame.jsonl"
    with open(subject_path, "w", encoding="utf-8") as fh:
        for row in subject_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return {
        "unit": "benchmark_task",
        "tasks_collected": len(tasks_seen),
        "tasks_included": len(subject_rows),
        "tasks_excluded_no_eligible_accepted": len(excluded_tasks),
        "excluded_tasks": excluded_tasks,
        "subject_frame_path": str(subject_path),
        "subject_frame_sha256": hashlib.sha256(
            subject_path.read_bytes()).hexdigest(),
    }


def merge_freeze(freeze_path: Path, summary: dict) -> dict:
    """Update only the corpora reconciled in this invocation."""
    existing = {}
    if freeze_path.exists():
        try:
            existing = json.loads(freeze_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
    corpora = existing.get("corpora", {})
    corpora.update(summary)
    freeze = {
        "created_at": _now(),
        "frame_schema_version": FRAME_SCHEMA_VERSION,
        "inclusion_rules_version": INCLUSION_RULES_VERSION,
        "inclusion_rules": [
            "R1 runnable_source: non-empty candidate kernel source "
            "(empty records are collection-time skips, outside the frame)",
            "R2 task_resolvable: task_key resolves to the underlying "
            "benchmark task (KB_L{1..3}_P{n} in level sizes 100/100/50 for "
            "C2/C3/C5; frozen TritonBench_G_v1 manifest entry for C4)",
            "R3 not_duplicate: not a normalized-source duplicate of an "
            "earlier same-corpus row (first kept; cross-corpus duplicates "
            "flagged, never dropped)",
            "R4 language_declared: static evidence matches the corpus "
            "language (C2/C3/C5 CUDA, C4 Triton)",
            "C3 unit is the task: representative = smallest stable_id among "
            "accepted included candidates of the task",
        ],
        "normalization": "strip trailing per-line whitespace + outer blank lines",
        "corpora": corpora,
        "note": "freeze these digests before observing any V2 validator outcome",
    }
    freeze_path.write_text(json.dumps(freeze, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    return freeze


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
        exclusions = Counter()
        language_detected = Counter()
        seen_in_corpus = set()
        frame_rows = []
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
                dup_within = sid in seen_in_corpus
                included, exclusion_reason, detected = inclusion_check(
                    corpus, source, row["task_key"], dup_within)
                language_detected[detected] += 1
                entry = {
                    "frame_schema_version": FRAME_SCHEMA_VERSION,
                    "frame_id": frame_id,
                    "stable_id": sid,
                    "raw_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                    **row,
                    "duplicate_within_corpus": dup_within,
                    "duplicate_of": (
                        None if duplicate_of is None or duplicate_of[1] == frame_id
                        else {"corpus": duplicate_of[0], "frame_id": duplicate_of[1]}),
                    "source_bytes": len(source.encode("utf-8")),
                    "language_detected": detected,
                    "included": included,
                    "exclusion_reason": exclusion_reason,
                    "inclusion_rules_version": INCLUSION_RULES_VERSION,
                }
                fh.write(json.dumps(entry, sort_keys=True) + "\n")
                frame_rows.append(entry)
                counts["total"] += 1
                counts["accepted"] += bool(row.get("accepted"))
                counts["dup_within"] += dup_within
                counts["dup_cross"] += (
                    duplicate_of is not None and duplicate_of[0] != corpus)
                if included:
                    counts["included"] += 1
                    counts["included_accepted"] += bool(row.get("accepted"))
                else:
                    exclusions[exclusion_reason] += 1
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
            "included": counts["included"],
            "included_accepted": counts["included_accepted"],
            "excluded_by_reason": dict(exclusions),
            "language_detected": dict(language_detected),
            "inclusion_rules_version": INCLUSION_RULES_VERSION,
            "frame_sha256": hashlib.sha256(frame_path.read_bytes()).hexdigest(),
            "frame_path": str(frame_path),
        }
        if corpus in TASK_LEVEL_SUBJECT_CORPORA:
            summary[corpus]["subject_frame"] = build_task_subject_frame(
                corpus, frame_rows, args.out_dir)
        print(f"[{corpus}] rows={counts['total']} unique={unique} "
              f"accepted={counts['accepted']} dup_within={counts['dup_within']} "
              f"dup_cross={counts['dup_cross']} included={counts['included']} "
              f"excluded={dict(exclusions)}", flush=True)

    merge_freeze(args.out_dir / "collection_frames.freeze.json", summary)
    print(json.dumps({c: {k: v for k, v in s.items() if k != "provenance"}
                      for c, s in summary.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
