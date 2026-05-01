#!/usr/bin/env python3
"""Analyze why sakana kernels fail at baseline."""
from __future__ import annotations
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
reg = PROJECT_ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"

with open(reg, encoding="utf-8") as f:
    data = json.load(f)

for kid in ["sakana__L1_T34", "sakana__L1_T15"]:
    for e in data:
        if e["id"] == kid:
            ks = e["kernel_source"]
            print(f"=== {kid} ===")
            print(f"  Has PYBIND11_MODULE: {'PYBIND11_MODULE' in ks}")
            print(f"  Has load_inline: {'load_inline' in ks}")
            
            # Check if load_inline is at module level (not inside a function)
            lines = ks.split("\n")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if "load_inline(" in stripped or "_ext = load_inline" in stripped:
                    indent = len(line) - len(line.lstrip())
                    print(f"  load_inline at line {i+1}, indent={indent} (0=module level)")
                    
            # Check problem: the issue is that load_inline at module level
            # means CUDA compilation happens on IMPORT
            # In baseline test, kernel_code is loaded as "orig" AND "mut" = same code
            # So it compiles TWICE
            print(f"  kernel_source length: {len(ks)} chars")
            
            # Key: get_inputs produces large tensor
            ref_file = e["reference_file"]
            full_path = PROJECT_ROOT / ref_file
            if full_path.exists():
                with open(full_path) as f2:
                    ref_code = f2.read()
                # Find tensor sizes in get_inputs
                for line in ref_code.split("\n"):
                    if "randn" in line or "batch_size" in line or "dim" in line:
                        print(f"  ref: {line.strip()}")
            print()
            break
