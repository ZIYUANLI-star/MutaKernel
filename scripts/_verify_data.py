#!/usr/bin/env python3
"""Verify all pre-prepared dataset files are complete and valid."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def verify_registry(name: str, reg_path: Path) -> tuple[int, int, list[str]]:
    """Verify a registry.json: check that all reference_file entries exist.
    Returns (total, missing_count, missing_files).
    """
    if not reg_path.exists():
        return 0, 0, [f"REGISTRY NOT FOUND: {reg_path}"]

    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    missing = []

    for entry in data:
        ref = entry.get("reference_file", "")
        if os.path.isabs(ref):
            full = ref
        else:
            full = str(PROJECT_ROOT / ref)

        if not os.path.exists(full):
            missing.append(f"  {entry['id']}: {ref}")

        ks = entry.get("kernel_source", "")
        if not ks or len(ks.strip()) < 20:
            missing.append(f"  {entry['id']}: kernel_source empty/too short")

    return total, len(missing), missing[:10]


def main():
    ext = PROJECT_ROOT / "external_benchmarks"
    datasets = [
        ("CUDA-L1", ext / "cuda_l1" / "registry.json"),
        ("AI CUDA Engineer", ext / "ai_cuda_engineer" / "registry.json"),
        ("TritonBench-G", ext / "tritonbench_g" / "registry.json"),
    ]

    all_ok = True
    print("=" * 60)
    print("  Dataset Integrity Verification")
    print("=" * 60)

    for name, reg_path in datasets:
        total, miss_count, details = verify_registry(name, reg_path)
        status = "OK" if miss_count == 0 else "ISSUES"
        print(f"\n  {name}: {total} kernels, {miss_count} issues [{status}]")
        if details:
            for d in details:
                print(f"    {d}")
            all_ok = False

    # Apex check
    apex_reg = ext / "registry.py"
    if apex_reg.exists():
        print(f"\n  Apex: built-in registry.py [OK]")
    else:
        print(f"\n  Apex: registry.py NOT FOUND [FAIL]")
        all_ok = False

    print()
    if all_ok:
        print("  RESULT: All datasets verified. Ready for offline execution.")
    else:
        print("  RESULT: Some issues found. Please fix before transferring.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
