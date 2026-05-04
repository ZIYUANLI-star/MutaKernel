#!/usr/bin/env python3
"""Run baseline-only differential test on every Correct=True Sakana kernel.

For each of the 229 kernels in the registry, runs 3 seeds with the
``__identity__`` policy (i.e. unmodified, default-shaped inputs) and
records ``(passed, failed, errors)``.

Resumable via a checkpoint JSON. Crashes / OOMs on individual kernels
do NOT abort the run; the kernel is recorded as ERR and we move on.

Output (live + on disk):
  - <result_dir>/baseline_checkpoint.json   incremental, one entry per kernel
  - <result_dir>/baseline_summary.json      final report
  - stdout / log                            human-readable progress

Usage:
  python3 scripts/run_sakana_baseline_all.py [--limit N]
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

from scripts.run_fullscale_diff_test import run_quick_baseline  # noqa

REG = ROOT / "external_benchmarks/ai_cuda_engineer/registry.json"
RESULT_DIR = ROOT / "第三次实验汇总" / "results" / "ai_cuda_engineer"
CKPT = RESULT_DIR / "baseline_checkpoint.json"
SUMMARY = RESULT_DIR / "baseline_summary.json"


def classify(passed: int, failed: int, errors: int) -> str:
    if errors >= 2:
        return "ERR"
    if passed >= 2 and errors == 0:
        return "OK"
    return "BAD"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Limit kernels (0=all)")
    parser.add_argument("--n-seeds", type=int, default=3, help="Seeds per kernel for baseline")
    args = parser.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REG, encoding="utf-8") as f:
        registry = json.load(f)

    if args.limit:
        registry = registry[:args.limit]

    completed: dict[str, Any] = {}
    if CKPT.exists():
        try:
            with open(CKPT, encoding="utf-8") as f:
                completed = json.load(f)
            print(f"Resuming from checkpoint: {len(completed)} kernels already done")
        except Exception:
            completed = {}

    n_total = len(registry)
    n_done = 0
    t_run_start = time.time()

    print("=" * 78)
    print(f"  Sakana baseline-only test on {n_total} kernels")
    print(f"  Result dir: {RESULT_DIR}")
    print(f"  Checkpoint: {CKPT}")
    print("=" * 78)
    print()

    for idx, entry in enumerate(registry, 1):
        kid = entry["id"]
        if kid in completed:
            n_done += 1
            continue

        problem_file = str(ROOT / entry["reference_file"])
        kernel_code = entry["kernel_source"]
        kernel_name = entry.get("kernel_name", kid)

        if not os.path.exists(problem_file):
            rec = {
                "id": kid,
                "name": kernel_name,
                "verdict": "ERR",
                "reason": "ref_file_missing",
                "baseline_pass": 0, "baseline_fail": 0, "baseline_err": 0,
                "elapsed_s": 0,
            }
            completed[kid] = rec
            print(f"[{idx:3d}/{n_total}] {kid:<24}  SKIP  ref file missing")
            with open(CKPT, "w", encoding="utf-8") as f:
                json.dump(completed, f, indent=2, ensure_ascii=False)
            continue

        t0 = time.time()
        try:
            passed, failed, errors = run_quick_baseline(
                problem_file, kernel_code, n_seeds=args.n_seeds,
            )
        except Exception as ex:
            elapsed = time.time() - t0
            verdict = "ERR"
            rec = {
                "id": kid,
                "name": kernel_name,
                "verdict": "ERR",
                "reason": f"exception: {type(ex).__name__}: {str(ex)[:200]}",
                "baseline_pass": 0, "baseline_fail": 0, "baseline_err": args.n_seeds,
                "elapsed_s": round(elapsed, 1),
                "speedup": entry.get("speedup", 0),
                "level_id": entry.get("level_id", 0),
            }
        else:
            elapsed = time.time() - t0
            verdict = classify(passed, failed, errors)
            rec = {
                "id": kid,
                "name": kernel_name,
                "verdict": verdict,
                "baseline_pass": passed,
                "baseline_fail": failed,
                "baseline_err": errors,
                "elapsed_s": round(elapsed, 1),
                "speedup": entry.get("speedup", 0),
                "level_id": entry.get("level_id", 0),
            }
        completed[kid] = rec
        n_done += 1

        elapsed_total = time.time() - t_run_start
        eta_s = (elapsed_total / max(1, n_done - len(completed) + n_done)) * (n_total - n_done) \
                if n_done > 0 else 0

        print(f"[{idx:3d}/{n_total}] {kid:<24}  "
              f"p/f/e={rec['baseline_pass']}/{rec['baseline_fail']}/{rec['baseline_err']}  "
              f"{rec['elapsed_s']:5.1f}s  {verdict:3s}  "
              f"L{rec.get('level_id','?')}  "
              f"{kernel_name[:40]}",
              flush=True)

        # Save checkpoint after every kernel
        with open(CKPT, "w", encoding="utf-8") as f:
            json.dump(completed, f, indent=2, ensure_ascii=False)

        gc.collect()

    # Build final summary
    total_elapsed = time.time() - t_run_start
    by_verdict = {"OK": 0, "BAD": 0, "ERR": 0}
    by_verdict_per_level = {1: {"OK": 0, "BAD": 0, "ERR": 0},
                            2: {"OK": 0, "BAD": 0, "ERR": 0},
                            3: {"OK": 0, "BAD": 0, "ERR": 0}}
    ok_kernels = []
    bad_kernels = []
    err_kernels = []
    for kid, rec in completed.items():
        v = rec["verdict"]
        by_verdict[v] = by_verdict.get(v, 0) + 1
        lid = rec.get("level_id", 0)
        if lid in by_verdict_per_level:
            by_verdict_per_level[lid][v] = by_verdict_per_level[lid].get(v, 0) + 1
        if v == "OK":
            ok_kernels.append({
                "id": rec["id"],
                "name": rec["name"],
                "speedup": rec.get("speedup", 0),
                "level_id": rec.get("level_id", 0),
            })
        elif v == "BAD":
            bad_kernels.append({
                "id": rec["id"],
                "name": rec["name"],
                "p_f_e": (rec["baseline_pass"], rec["baseline_fail"], rec["baseline_err"]),
            })
        else:
            err_kernels.append({
                "id": rec["id"],
                "name": rec["name"],
                "reason": rec.get("reason", "")[:150],
            })

    summary = {
        "dataset": "ai_cuda_engineer (Correct=True only)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_kernels": n_total,
        "n_completed": len(completed),
        "n_seeds": args.n_seeds,
        "elapsed_total_s": round(total_elapsed, 1),
        "verdict_counts": by_verdict,
        "verdict_counts_per_level": by_verdict_per_level,
        "ok_kernels": sorted(ok_kernels, key=lambda x: -x["speedup"]),
        "bad_kernels": bad_kernels,
        "err_kernels": err_kernels,
    }
    with open(SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    n = sum(by_verdict.values())
    print()
    print("=" * 78)
    print(f"  FINAL ({total_elapsed:.0f}s = {total_elapsed/60:.1f} min)")
    print("=" * 78)
    print(f"  Kernels completed:  {len(completed)}/{n_total}")
    print(f"    OK  (>={args.n_seeds-1}/{args.n_seeds} pass): {by_verdict['OK']:3d}  ({by_verdict['OK']/max(1,n)*100:.1f}%)")
    print(f"    BAD (semantic mismatch): {by_verdict['BAD']:3d}  ({by_verdict['BAD']/max(1,n)*100:.1f}%)")
    print(f"    ERR (worker error/OOM):  {by_verdict['ERR']:3d}  ({by_verdict['ERR']/max(1,n)*100:.1f}%)")
    print()
    print(f"  Per level:")
    for lid in [1, 2, 3]:
        d = by_verdict_per_level[lid]
        tot = sum(d.values())
        if tot:
            print(f"    Level {lid}: OK={d['OK']:2d}  BAD={d['BAD']:2d}  ERR={d['ERR']:2d}  (total {tot})")
    print()
    print(f"  Summary: {SUMMARY}")
    print(f"  Checkpoint: {CKPT}")


if __name__ == "__main__":
    main()
