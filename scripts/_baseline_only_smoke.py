#!/usr/bin/env python3
"""Baseline-only smoke test on 30 sakana kernels (sampled across tasks).

We only run the 3-seed baseline (no stress dimensions) to get a quick
estimate of the wrapper-fix's impact on baseline pass rate. This is the
key metric: how many kernels survive the basic differential test.
"""
from __future__ import annotations
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

from scripts.run_fullscale_diff_test import run_quick_baseline  # noqa

REG = ROOT / "external_benchmarks/ai_cuda_engineer/registry.json"
with open(REG, encoding="utf-8") as f:
    full = json.load(f)

# Sample 30 kernels with deterministic seed across L1/L2/L3, biased toward
# diverse op types (skip duplicates of same Op_Name prefix).
random.seed(42)
sample = []
seen_op_prefixes = set()
random.shuffle(full)
for e in full:
    if len(sample) >= 30:
        break
    op = e.get("kernel_name", "")
    prefix = op.split("_")[0] if "_" in op else op
    if prefix in seen_op_prefixes:
        continue
    seen_op_prefixes.add(prefix)
    sample.append(e)

# Resort by id for stable presentation
sample.sort(key=lambda e: e["id"])

print(f"Testing {len(sample)} kernels (baseline only, 3 seeds each)")
print(f"{'='*72}\n")

results = []
t_total = time.time()
for i, e in enumerate(sample, 1):
    kid = e["id"]
    problem_file = str(ROOT / e["reference_file"])
    kernel_code = e["kernel_source"]
    name = e.get("kernel_name", kid)

    t0 = time.time()
    try:
        passed, failed, errors = run_quick_baseline(problem_file, kernel_code, n_seeds=3)
    except Exception as ex:
        print(f"[{i}/{len(sample)}] {kid}: EXCEPTION {ex}")
        results.append((kid, name, "EXCEPTION", 0, 0, 0, 0))
        continue
    elapsed = time.time() - t0
    if passed >= 2 and errors == 0:
        verdict = "OK"
    elif errors >= 2:
        verdict = "ERR"
    else:
        verdict = "BAD"
    print(f"[{i:2d}/{len(sample)}] {kid:<26}  p/f/e={passed}/{failed}/{errors}  "
          f"({elapsed:5.1f}s)  {verdict}  {name[:48]}")
    results.append((kid, name, verdict, passed, failed, errors, round(elapsed, 1)))

total_elapsed = time.time() - t_total
n = len(results)
ok = sum(1 for r in results if r[2] == "OK")
err = sum(1 for r in results if r[2] == "ERR")
bad = sum(1 for r in results if r[2] == "BAD")

print(f"\n{'='*72}")
print(f"SUMMARY  ({total_elapsed:.1f}s total)")
print(f"{'='*72}")
print(f"Total kernels: {n}")
print(f"  OK  (>=2/3 baseline pass, 0 errors):     {ok}  ({ok/n*100:.1f}%)")
print(f"  BAD (>=2/3 baseline pass discrepancy):   {bad} ({bad/n*100:.1f}%)")
print(f"  ERR (>=2/3 worker errors / compile):     {err} ({err/n*100:.1f}%)")

# Save JSON
out = ROOT / "第三次实验汇总" / "logs" / "smoke_sakana_fixed_30.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump([
        {"id": r[0], "name": r[1], "verdict": r[2],
         "baseline_pass": r[3], "baseline_fail": r[4], "baseline_err": r[5],
         "elapsed_s": r[6] if len(r) > 6 else 0}
        for r in results
    ], f, indent=2, ensure_ascii=False)
print(f"\nResults saved: {out}")
