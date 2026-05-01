#!/usr/bin/env python3
"""Inspect specific kernels from a registry and attempt to diagnose compilation issues."""
from __future__ import annotations
import json, sys, os, importlib, traceback, tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def inspect_kernel(reg_path, kid):
    with open(reg_path, encoding="utf-8") as f:
        registry = json.load(f)

    entry = None
    for e in registry:
        if e["id"] == kid:
            entry = e
            break

    if not entry:
        print(f"Kernel {kid} not found in {reg_path}")
        return

    print(f"=== {kid} ===")
    print(f"  kernel_name: {entry.get('kernel_name', '?')}")
    print(f"  reference_file: {entry.get('reference_file', '?')}")
    ks = entry.get("kernel_source", "")
    print(f"  kernel_source length: {len(ks)} chars")

    # Check reference file exists
    ref = entry.get("reference_file", "")
    if not os.path.isabs(ref):
        ref = str(PROJECT_ROOT / ref)
    if os.path.exists(ref):
        print(f"  reference_file: EXISTS")
        with open(ref, encoding="utf-8") as f:
            ref_code = f.read()
        print(f"  reference code length: {len(ref_code)} chars")
        # Check for Model class
        if "class Model" in ref_code:
            print(f"  reference has class Model: YES")
        else:
            print(f"  reference has class Model: NO <-- PROBLEM")
        if "get_inputs" in ref_code:
            print(f"  reference has get_inputs: YES")
        else:
            print(f"  reference has get_inputs: NO <-- PROBLEM")
        if "get_init_inputs" in ref_code:
            print(f"  reference has get_init_inputs: YES")
        else:
            print(f"  reference has get_init_inputs: NO <-- PROBLEM")
    else:
        print(f"  reference_file: MISSING at {ref}")

    # Check kernel_source for load_inline
    if "load_inline" in ks:
        print(f"  kernel uses load_inline: YES (CUDA JIT compilation)")
    if "triton" in ks.lower():
        print(f"  kernel uses Triton: YES")
    if "class ModelNew" in ks:
        print(f"  kernel has class ModelNew: YES")
    else:
        print(f"  kernel has class ModelNew: NO <-- PROBLEM")

    # Print first 80 lines of kernel_source
    lines = ks.split("\n")
    print(f"\n  --- kernel_source (first 30 lines) ---")
    for i, line in enumerate(lines[:30]):
        print(f"  {i+1:3d} | {line}")
    if len(lines) > 30:
        print(f"  ... ({len(lines) - 30} more lines)")

    # Print __init__ signature of ModelNew if present
    for i, line in enumerate(lines):
        if "def __init__" in line and i > 0:
            context = lines[max(0,i-1):min(len(lines),i+5)]
            print(f"\n  --- ModelNew.__init__ (around line {i+1}) ---")
            for c in context:
                print(f"    {c}")
            break

    print()


if __name__ == "__main__":
    reg = PROJECT_ROOT / "external_benchmarks" / "ai_cuda_engineer" / "registry.json"
    for kid in ["sakana__L1_T34", "sakana__L1_T15"]:
        inspect_kernel(reg, kid)
