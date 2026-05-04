#!/usr/bin/env python3
"""Count how many AI CUDA Engineer kernels have the PYBIND11_MODULE conflict."""
from __future__ import annotations
import json, re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
reg = PROJECT_ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"

with open(reg, encoding="utf-8") as f:
    data = json.load(f)

has_conflict = 0
has_load_inline = 0
no_load_inline = 0

for e in data:
    ks = e["kernel_source"]
    if "load_inline" not in ks:
        no_load_inline += 1
        continue

    has_load_inline += 1
    if "PYBIND11_MODULE" in ks and "functions=" in ks:
        has_conflict += 1

print(f"Total kernels: {len(data)}")
print(f"  Without load_inline: {no_load_inline}")
print(f"  With load_inline: {has_load_inline}")
print(f"  With PYBIND11_MODULE + functions= conflict: {has_conflict}")
print(f"  -> {has_conflict} kernels would have FAILED compilation without our fix")
