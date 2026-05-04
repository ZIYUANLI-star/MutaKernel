#!/usr/bin/env python3
"""Phase 2: Run full 5-dimension enhanced test on 5 selected baseline-OK kernels.

Reads ``第三次实验汇总/results/ai_cuda_engineer/baseline_summary.json``,
picks 5 baseline-OK kernels (diverse: different levels & speedups), and
runs the full 5-dimension MutaKernel test suite on each. Then summarizes
which dimensions found extra bugs that baseline missed.

Usage:
  python3 scripts/run_sakana_5dim_select.py [--ids id1,id2,...]

If --ids is not given, picks 5 automatically: top-1 by speedup per level
plus 2 fillers (preferring under-represented levels).
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

from scripts.run_fullscale_diff_test import run_kernel_5dim  # noqa

REG = ROOT / "external_benchmarks/ai_cuda_engineer/registry.json"
RESULT_DIR = ROOT / "第三次实验汇总" / "results" / "ai_cuda_engineer"
BASELINE_SUMMARY = RESULT_DIR / "baseline_summary.json"
PHASE2_DIR = RESULT_DIR / "phase2_5dim"


def auto_select_5(ok_kernels: list[dict]) -> list[str]:
    """Pick 5 kernels covering the 3 levels with diverse speedups."""
    by_level: dict[int, list[dict]] = {1: [], 2: [], 3: []}
    for k in ok_kernels:
        lid = k.get("level_id", 1)
        if lid in by_level:
            by_level[lid].append(k)

    for lid in by_level:
        by_level[lid].sort(key=lambda x: -x.get("speedup", 0))

    picked: list[dict] = []
    if by_level[1]:
        picked.append(by_level[1][0])
    if by_level[2]:
        picked.append(by_level[2][0])
    if by_level[3]:
        picked.append(by_level[3][0])

    extras: list[dict] = []
    for lid in [1, 2, 3]:
        for k in by_level[lid][1:]:
            extras.append(k)
    extras.sort(key=lambda x: -x.get("speedup", 0))
    while len(picked) < 5 and extras:
        picked.append(extras.pop(0))

    return [k["id"] for k in picked]


def has_5dim_finding(result: dict) -> tuple[bool, list[str]]:
    """Check whether 5-dim test found any discrepancies. Returns (found, reasons)."""
    findings = []
    for dim in ["value_stress", "dtype_stress", "training_stress",
                "repeated_run", "config_stress"]:
        d = result.get(dim, {})
        n_disc = d.get("discrepancies", 0)
        if n_disc > 0:
            details = d.get("details", {})
            if details:
                tags = list(details.keys())[:3]
                findings.append(f"{dim}: {n_disc} disc ({','.join(map(str, tags))})")
            else:
                findings.append(f"{dim}: {n_disc} disc")
    return (len(findings) > 0, findings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="", help="Comma-separated kernel ids (override auto-select)")
    args = parser.parse_args()

    PHASE2_DIR.mkdir(parents=True, exist_ok=True)

    if not BASELINE_SUMMARY.exists():
        print(f"[ERROR] baseline_summary.json not found at {BASELINE_SUMMARY}")
        print("Run scripts/run_sakana_baseline_all.py first.")
        sys.exit(1)

    with open(BASELINE_SUMMARY, encoding="utf-8") as f:
        bsummary = json.load(f)
    ok_kernels = bsummary.get("ok_kernels", [])
    print(f"[INFO] baseline OK kernels available: {len(ok_kernels)}")

    if args.ids:
        selected_ids = [s.strip() for s in args.ids.split(",") if s.strip()]
    else:
        selected_ids = auto_select_5(ok_kernels)

    print(f"[INFO] Selected for phase-2: {selected_ids}")

    with open(REG, encoding="utf-8") as f:
        registry_list = json.load(f)
    registry = {e["id"]: e for e in registry_list}

    phase2: list[dict[str, Any]] = []
    t_start = time.time()

    for idx, kid in enumerate(selected_ids, 1):
        if kid not in registry:
            print(f"[WARN] {kid} not in registry, skipping")
            continue
        entry = registry[kid]
        problem_file = str(ROOT / entry["reference_file"])
        kernel_code = entry["kernel_source"]
        kernel_name = entry.get("kernel_name", kid)

        print()
        print("=" * 78)
        print(f"  [{idx}/{len(selected_ids)}]  {kid}  ({kernel_name})")
        print(f"  level={entry.get('level_id')}  speedup={entry.get('speedup'):.3f}x")
        print("=" * 78)

        t0 = time.time()
        try:
            result = run_kernel_5dim(problem_file, kernel_code, kid)
        except Exception as ex:
            elapsed = time.time() - t0
            rec = {
                "id": kid, "name": kernel_name,
                "level_id": entry.get("level_id"),
                "speedup": entry.get("speedup", 0),
                "elapsed_s": round(elapsed, 1),
                "error": f"{type(ex).__name__}: {str(ex)[:300]}",
                "found_extra_bug": False,
                "findings": [],
                "result": None,
            }
        else:
            elapsed = time.time() - t0
            found, findings = has_5dim_finding(result)
            rec = {
                "id": kid, "name": kernel_name,
                "level_id": entry.get("level_id"),
                "speedup": entry.get("speedup", 0),
                "elapsed_s": round(elapsed, 1),
                "found_extra_bug": found,
                "findings": findings,
                "summary_per_dim": {
                    dim: {
                        "passes": result.get(dim, {}).get("passes", 0),
                        "discrepancies": result.get(dim, {}).get("discrepancies", 0),
                    }
                    for dim in ["value_stress", "dtype_stress", "training_stress",
                                "repeated_run", "config_stress"]
                },
                "result": result,
            }
        phase2.append(rec)

        per_kernel_path = PHASE2_DIR / f"{kid}.json"
        with open(per_kernel_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False, default=str)

        print()
        print(f"  Elapsed: {rec['elapsed_s']:.1f}s")
        if rec.get("error"):
            print(f"  [CRASH] {rec['error']}")
        else:
            print(f"  Per-dimension summary:")
            for dim, sd in rec["summary_per_dim"].items():
                pas = sd["passes"]
                disc = sd["discrepancies"]
                tag = "[!]" if disc else "   "
                print(f"     {tag} {dim:<18}  pass={pas:2d}  disc={disc:2d}")
            if rec["found_extra_bug"]:
                print(f"  >>> 5-dim found extra bug(s) NOT caught by baseline:")
                for fnd in rec["findings"]:
                    print(f"      - {fnd}")
            else:
                print(f"  >>> 5-dim found NOTHING beyond baseline (kernel is robust)")

        gc.collect()

    summary_path = PHASE2_DIR / "phase2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_selected": len(selected_ids),
            "elapsed_total_s": round(time.time() - t_start, 1),
            "kernels": phase2,
        }, f, indent=2, ensure_ascii=False, default=str)

    n_extra = sum(1 for k in phase2 if k.get("found_extra_bug"))
    print()
    print("=" * 78)
    print(f"  PHASE-2 FINAL: {len(phase2)} kernels tested")
    print(f"    Found extra bugs (5-dim > baseline): {n_extra}/{len(phase2)}")
    print(f"    Robust (no new finding):              {len(phase2) - n_extra}/{len(phase2)}")
    print(f"  Total elapsed: {(time.time() - t_start)/60:.1f} min")
    print(f"  Summary: {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
