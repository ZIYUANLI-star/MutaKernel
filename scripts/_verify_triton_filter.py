#!/usr/bin/env python3
"""Verify TritonBench identity passthrough filter works."""
from __future__ import annotations
import json, os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
tb_reg_path = PROJECT_ROOT / "external_benchmarks" / "tritonbench_g" / "registry.json"

with open(tb_reg_path, encoding="utf-8") as f:
    tb_registry = json.load(f)

tb_before = len(tb_registry)
tb_filtered = []
for e in tb_registry:
    ref = e.get("reference_file", "")
    ref_path = Path(ref) if os.path.isabs(ref) else PROJECT_ROOT / ref
    if ref_path.exists():
        try:
            with open(ref_path, encoding="utf-8") as rf:
                if "identity passthrough" not in rf.read():
                    tb_filtered.append(e)
        except Exception:
            pass

print(f"TritonBench-G: {len(tb_filtered)}/{tb_before} kernels with valid PyTorch reference")
print(f"Filtered {tb_before - len(tb_filtered)} identity-passthrough kernels")

# Show some examples of kept kernels
print(f"\nFirst 5 valid kernels:")
for e in tb_filtered[:5]:
    ref_path = PROJECT_ROOT / e["reference_file"]
    with open(ref_path) as f:
        code = f.read()
    for line in code.split("\n"):
        if "return " in line and "def " not in line and "get_" not in line:
            print(f"  {e['id']}: {line.strip()}")
            break
