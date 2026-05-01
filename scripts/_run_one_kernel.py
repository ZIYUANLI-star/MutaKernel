#!/usr/bin/env python3
"""Run full 5-dim enhanced test on a specific kernel by ID."""
from __future__ import annotations
import json, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

from scripts.run_fullscale_diff_test import run_dataset, RESULT_BASE, PROJECT_ROOT

kid = sys.argv[1]
dataset = sys.argv[2] if len(sys.argv) > 2 else "auto"

datasets_map = {
    "sakana": "ai_cuda_engineer",
    "ai_cuda_engineer": "ai_cuda_engineer",
    "tritonbench": "tritonbench_g",
    "tritonbench_g": "tritonbench_g",
    "cuda_l1": "cuda_l1",
    "cuda-l1": "cuda_l1",
}

if dataset == "auto":
    if kid.startswith("sakana"):
        dataset = "ai_cuda_engineer"
    elif kid.startswith("tritonbench"):
        dataset = "tritonbench_g"
    elif kid.startswith("cuda_l1"):
        dataset = "cuda_l1"
    else:
        print(f"Cannot auto-detect dataset for {kid}, specify as 2nd arg")
        sys.exit(1)
else:
    dataset = datasets_map.get(dataset, dataset)

reg_path = PROJECT_ROOT / "external_benchmarks" / dataset / "registry.json"
with open(reg_path, encoding="utf-8") as f:
    registry = json.load(f)

entry = [e for e in registry if e["id"] == kid]
if not entry:
    print(f"Kernel {kid} not found in {reg_path}")
    sys.exit(1)

result_dir = RESULT_BASE / dataset
run_dataset(dataset, entry, result_dir)
