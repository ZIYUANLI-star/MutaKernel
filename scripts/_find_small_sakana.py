#!/usr/bin/env python3
"""Find AI CUDA Engineer kernels with small input tensors suitable for local testing."""
from __future__ import annotations
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
reg = PROJECT_ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"

with open(reg, encoding="utf-8") as f:
    data = json.load(f)

print(f"Total kernels: {len(data)}\n")
print(f"{'ID':<30} {'ref_exists':>10} {'ks_chars':>10} {'load_inline':>12} {'ref_summary'}")
print("-" * 100)

for e in data:
    kid = e["id"]
    ks = e["kernel_source"]
    ref_path = PROJECT_ROOT / e["reference_file"]

    ref_exists = ref_path.exists()
    has_li = "load_inline" in ks

    ref_summary = ""
    if ref_exists:
        with open(ref_path, encoding="utf-8") as f2:
            ref_code = f2.read()
        for line in ref_code.split("\n"):
            s = line.strip()
            if "randn" in s or "rand(" in s or "randint" in s or "zeros" in s or "ones(" in s:
                ref_summary += s + " | "

    print(f"{kid:<30} {str(ref_exists):>10} {len(ks):>10} {str(has_li):>12}   {ref_summary[:80]}")
