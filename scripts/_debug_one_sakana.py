#!/usr/bin/env python3
"""Run one sakana kernel through the worker with verbose output (diff_summary)."""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

from scripts.run_fullscale_diff_test import (  # noqa
    DEVICE, DEFAULT_ATOL, DEFAULT_RTOL, STRESS_TIMEOUT, _run_stress_worker,
)

REG = ROOT / "external_benchmarks/ai_cuda_engineer/registry.json"
with open(REG, encoding="utf-8") as f:
    full = json.load(f)
by_id = {e["id"]: e for e in full}

kid = sys.argv[1] if len(sys.argv) > 1 else "sakana__L1_T34"
e = by_id[kid]
problem_file = str(ROOT / e["reference_file"])
kernel_code = e["kernel_source"]
print(f"Testing {kid}  ({e.get('kernel_name')})")
print(f"  problem_file: {problem_file}")
print()

for seed in range(3):
    cfg = {
        "mode": "value_stress",
        "problem_file": problem_file,
        "kernel_code": kernel_code,
        "mutated_code": kernel_code,
        "policy_name": "__identity__",
        "seed": seed,
        "atol": DEFAULT_ATOL, "rtol": DEFAULT_RTOL,
        "device": DEVICE,
        "sync_weights": True,
    }
    data = _run_stress_worker(cfg, timeout=STRESS_TIMEOUT)
    if data is None:
        print(f"seed={seed}: timeout")
        continue
    print(f"seed={seed}:")
    for k in ["ref_ok", "original_ok", "mutant_ok", "ref_nan_fallback",
              "diff_summary", "error", "time_ms"]:
        if k in data:
            v = data[k]
            if isinstance(v, str) and len(v) > 200:
                v = v[:200] + "..."
            print(f"  {k}: {v}")
