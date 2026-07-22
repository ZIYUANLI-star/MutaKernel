#!/usr/bin/env python3
"""E3 external-validity harness: C6 gpuemu corpus + B7/B8/B9 port skeletons.

Status (2026-07-22):
  IMPLEMENTED  - C6 corpus loader (26 ops: 16 correct controls + 10 seeded
                 bugs) and the per-op runner scaffold that executes candidate
                 vs the corpus's own fp64 reference under the *unified*
                 judging pipeline (src.validation.compare_outputs with the
                 per-op calibrated tolerances from meta.json).
  IMPLEMENTED  - dry-run mode (--dry-run) that validates the corpus layout
                 and prints the task table CPU-only.
  SKELETON     - B7 (robust-kbench) / B8 (KernelBenchX) / B9 (gpuemu seeded
                 differential fuzzing) protocol ports: entry points +
                 clause-by-clause alignment checklists are laid out below;
                 each raises NotImplementedError with its TODO list so a
                 partial port can never silently produce table rows.

Port-fidelity rules (blueprint §5.1.2): every port must (i) release full
source, (ii) document clause-by-clause alignment, (iii) reproduce the
original tool's published results on its native dataset before any
budget-matched row is reported.  Native and port rows never share a table
row.

Usage:
  # CPU-only structural check (safe while run5 owns the GPU):
  python scripts/run_e3_external.py c6 --corpus-root external/C6_gpuemu \
      --out-dir /tmp/e3_c6 --dry-run
  # GPU run (after run5):
  python scripts/run_e3_external.py c6 --corpus-root external/C6_gpuemu \
      --out-dir /root/mk_v2_runs/e3_c6
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# C6: gpuemu seeded-bug corpus (third-party labels; RQ3, Table 9)
# ---------------------------------------------------------------------------

def load_c6_corpus(corpus_root: Path):
    """Load the 26-op gpuemu corpus (meta.json + ref_fp64.py + kernel.py)."""
    data_dir = corpus_root / "gpuemu-corpus" / "gpuemu_corpus" / "data"
    if not data_dir.is_dir():
        candidates = list(corpus_root.rglob("gpuemu_corpus/data"))
        if len(candidates) == 1:
            data_dir = candidates[0]
        else:
            raise FileNotFoundError(f"gpuemu corpus data dir not found under {corpus_root}")
    ops = []
    for op_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        meta_path = op_dir / "meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # meta.reference/kernel are paths relative to the data dir; buggy
        # variants share the correct op's fp64 reference (e.g.
        # "softmax/ref_fp64.py").
        ref_path = data_dir / str(meta.get("reference", f"{op_dir.name}/ref_fp64.py"))
        kernel_path = data_dir / str(meta.get("kernel", f"{op_dir.name}/kernel.py"))
        ops.append({
            "op_name": op_dir.name,
            "meta": meta,
            "third_party_label": (
                "seeded_bug" if op_dir.name.endswith("_buggy")
                else "correct_control"),
            "source_tag": meta.get("source"),
            "tolerances": meta.get("tolerances"),
            "dtypes": meta.get("dtypes"),
            "ref_path": str(ref_path),
            "kernel_path": str(kernel_path),
            "has_ref": ref_path.is_file(),
            "has_kernel": kernel_path.is_file(),
        })
    return ops


def cmd_c6(args):
    ops = load_c6_corpus(args.corpus_root)
    labels = Counter(op["third_party_label"] for op in ops)
    print(f"[{_now()}] C6: {len(ops)} ops, labels={dict(labels)}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "c6_task_table.json").write_text(json.dumps({
        "created_at": _now(),
        "corpus_root": str(args.corpus_root),
        "ops": [{k: v for k, v in op.items() if k != "meta"} for op in ops],
        "label_counts": dict(labels),
        "label_note": (
            "third_party_label derives from the corpus's own *_buggy naming; "
            "reconciled 16 correct controls + 10 seeded bugs must match the "
            "gpuemu paper's published split before Table 9 is filled"),
    }, indent=2), encoding="utf-8")

    if args.dry_run:
        missing = [op["op_name"] for op in ops if not (op["has_ref"] and op["has_kernel"])]
        print(f"dry-run OK: {len(ops)} ops, incomplete={missing}")
        return

    # GPU execution path.
    # TODO(E3-C6): per-op runner —
    #   1. load ref_fp64.py (fp64 CPU/GPU reference) and kernel.py (Triton
    #      candidate) via the op_schema in meta.json (input shapes/dtypes);
    #   2. sample inputs per the op schema (seeded, replayable);
    #   3. judge with src.validation.compare_outputs under meta.json's
    #      per-(op,dtype) calibrated tolerances (unified oracle);
    #   4. one observation per (validator_config, op, case) with replay
    #      bundle for every non-PASS — reuse the E1 driver's worker pattern;
    #   5. emit Table 9 rows: bugs detected /10, controls falsely flagged /16,
    #      Clopper-Pearson exact intervals, per-bug itemization.
    raise NotImplementedError(
        "C6 GPU runner pending: the corpus loader and task table are ready; "
        "implement the per-op execution + unified-oracle judging (TODO list "
        "in source).  Do not run while E0 run5 owns the GPU.")


# ---------------------------------------------------------------------------
# B7 / B8 / B9 protocol ports (skeletons; fail loudly until aligned)
# ---------------------------------------------------------------------------

B7_ALIGNMENT_CHECKLIST = [
    "multi-init: N constructor re-initializations per subject (upstream run_kernel.py)",
    "multi-input: M input draws per init; forward AND backward comparison",
    "statistical output filters (upstream run_filter.py) - port or mark unsupported",
    "native anchors: atol/rtol=1e-5, 5 trials, 2 candidate calls per forward trial "
    "(configs/external_baselines.json robust-kbench-native-078f5bab)",
    "budget-matched port: exactly 32 candidate invocations per subject",
    "unified oracle: replace upstream allclose with src.validation compare_outputs",
    "native-dataset reproduction: rerun on external/B7_robust_kbench tasks/ and "
    "diff against highlighted/results.csv (port-fidelity delta for §5.1.2)",
]

B8_ALIGNMENT_CHECKLIST = [
    "standard/outlier/boundary input families (outlier prob 0.001, magnitude 50; "
    "configs/external_baselines.json kernelbenchx-native-fd419229)",
    "dtype-aware oracles as shipped upstream (native) vs unified oracle (port)",
    "task list: data/kernelbenchx_v1.json (176 Triton tasks) - never glob dirs",
    "budget-matched port: 32 candidate invocations",
    "native-dataset reproduction against upstream metrics/",
]

B9_ALIGNMENT_CHECKLIST = [
    "op-schema-aware input sampling (gpuemu daemon protocol, external/C6_gpuemu/gpuemu)",
    "fp64 CPU reference execution",
    "per-(op,dtype) calibrated tolerances from corpus meta.json",
    "port onto our frozen C2-C5 subjects: map subject task contracts onto op schemas; "
    "mark unsupported subjects explicitly (never silently drop)",
    "budget-matched: 32 candidate invocations",
]


def _skeleton(name: str, checklist):
    def command(args):
        print(f"{name} port checklist ({len(checklist)} clauses):")
        for index, clause in enumerate(checklist, 1):
            print(f"  {index}. {clause}")
        raise NotImplementedError(
            f"{name} protocol port not yet implemented; complete the clause "
            "checklist above, then register the port in "
            "configs/external_baselines.json before any primary run.")
    return command


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    c6 = sub.add_parser("c6", help="gpuemu seeded-bug corpus (RQ3)")
    c6.add_argument("--corpus-root", required=True, type=Path)
    c6.add_argument("--out-dir", required=True, type=Path)
    c6.add_argument("--dry-run", action="store_true")
    c6.set_defaults(func=cmd_c6)

    for name, checklist in (("b7", B7_ALIGNMENT_CHECKLIST),
                            ("b8", B8_ALIGNMENT_CHECKLIST),
                            ("b9", B9_ALIGNMENT_CHECKLIST)):
        parser = sub.add_parser(name, help=f"{name.upper()} protocol port (skeleton)")
        parser.set_defaults(func=_skeleton(name.upper(), checklist))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
