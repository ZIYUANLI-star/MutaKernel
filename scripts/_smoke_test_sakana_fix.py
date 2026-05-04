#!/usr/bin/env python3
"""Smoke test for the fixed Sakana wrapper.

For a curated set of diverse kernels (activation, parameterized layer,
multi-input op, reduction-with-dim, conv-style), we:
  1. Run a baseline test (3 seeds with __identity__ policy)
  2. Run a small value_stress sweep (3 policies x 1 seed)
  3. Report pass/fail counts

This is a quick pre-flight; full 5-dim testing comes later.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

from scripts.run_fullscale_diff_test import (  # noqa
    DEVICE, DEFAULT_ATOL, DEFAULT_RTOL, STRESS_TIMEOUT,
    _run_stress_worker, _classify, run_quick_baseline,
)

REG = ROOT / "external_benchmarks/ai_cuda_engineer/registry.json"
with open(REG, encoding="utf-8") as f:
    full = json.load(f)
by_id = {e["id"]: e for e in full}

# Curated diverse smoke set
SMOKE_IDS = [
    "sakana__L1_T25",   # Swish (element-wise, no params) — known to pass before
    "sakana__L1_T34",   # InstanceNorm (parameterized: weight/bias/eps)
    "sakana__L1_T40",   # LayerNorm (multiple params)
    "sakana__L1_T36",   # RMSNorm (eps param)
    "sakana__L1_T50",   # Product reduction (dim arg)
    "sakana__L1_T98",   # KLDivLoss (multi-input, no params)
    "sakana__L1_T15",   # Lower triangular MM (multi-input, no params)
    "sakana__L1_T88",   # MinGPT GELU (element-wise)
]

POLICIES_FOR_VALUE_STRESS = ["large_magnitude", "extreme_magnitude", "all_negative"]


def run_value_stress_one(problem_file, kernel_code, policy, seed):
    cfg = {
        "mode": "value_stress",
        "problem_file": problem_file,
        "kernel_code": kernel_code,
        "mutated_code": kernel_code,
        "policy_name": policy,
        "seed": seed,
        "atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL,
        "device": DEVICE,
        "sync_weights": True,
    }
    return _run_stress_worker(cfg, timeout=STRESS_TIMEOUT)


def main():
    rows = []
    t_total_start = time.time()
    for kid in SMOKE_IDS:
        if kid not in by_id:
            print(f"[SKIP] {kid} not in registry")
            continue
        e = by_id[kid]
        problem_file = str(ROOT / e["reference_file"])
        kernel_code = e["kernel_source"]
        kernel_name = e.get("kernel_name", kid)

        print(f"\n{'='*72}")
        print(f"[{kid}] {kernel_name}")
        print(f"{'='*72}")

        t0 = time.time()
        passed, failed, errors = run_quick_baseline(problem_file, kernel_code, n_seeds=3)
        t_base = time.time() - t0
        print(f"  Baseline: {passed} pass, {failed} fail, {errors} errors  (took {t_base:.1f}s)")

        # Skip value_stress if baseline all errored (compile failure)
        if errors == 3:
            rows.append((kid, kernel_name, "ALL_ERROR", passed, failed, errors, 0, 0, 0))
            continue

        vs_pass = vs_disc = vs_err = 0
        seed_base = 50000
        for i, policy in enumerate(POLICIES_FOR_VALUE_STRESS):
            data = run_value_stress_one(problem_file, kernel_code, policy, seed_base + i)
            status = _classify(data)
            err_str = (data or {}).get("error", "")[:80]
            print(f"  value_stress[{policy}]: {status}  err={err_str!r}")
            if status == "pass":
                vs_pass += 1
            elif status == "discrepancy":
                vs_disc += 1
            elif status == "ref_fail":
                # Ref crashed under stress input — this is expected for some
                # extreme inputs; we don't penalize the kernel.
                vs_err += 1
            else:
                vs_err += 1

        verdict = "BASELINE_OK" if (passed >= 2 and errors == 0) else "BASELINE_BAD"
        rows.append((kid, kernel_name, verdict, passed, failed, errors,
                     vs_pass, vs_disc, vs_err))

    total_elapsed = time.time() - t_total_start

    print(f"\n{'='*72}")
    print(f"SUMMARY (total {total_elapsed:.1f}s)")
    print(f"{'='*72}")
    print(f"{'kid':<25} {'verdict':<14} BL:p/f/e  VS:p/d/e   kernel_name")
    for r in rows:
        kid, kn, v, p, f, e, vp, vd, ve = r
        print(f"{kid:<25} {v:<14} {p}/{f}/{e}     {vp}/{vd}/{ve}     {kn}")

    n = len(rows)
    bl_ok = sum(1 for r in rows if r[2] == "BASELINE_OK")
    print(f"\nBaseline OK rate: {bl_ok}/{n} = {bl_ok/max(1,n)*100:.1f}%")


if __name__ == "__main__":
    main()
