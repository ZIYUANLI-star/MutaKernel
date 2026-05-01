"""Batch verification of Category B operators on real CUDA kernel files.

Run:  python -m scripts.verify_b_operators_batch
"""
import os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mutengine.operators.gpu_parallel import (
    IndexReplace, SyncRemove, MaskBoundary, LaunchConfigMutate,
)

KERNEL_DIR = Path(__file__).resolve().parent.parent / "test_data" / "cuda_kernels"
SEP = "=" * 78


def analyze(filepath, b1, b2, b3, b4):
    source = filepath.read_text(encoding="utf-8", errors="replace")
    loc = len(source.splitlines())
    s1 = b1.find_sites(source)
    s2 = b2.find_sites(source)
    s3 = b3.find_sites(source)
    s4 = b4.find_sites(source)
    return {
        "name": filepath.stem, "loc": loc,
        "b1": len(s1), "b2": len(s2), "b3": len(s3), "b4": len(s4),
        "s1": s1, "s2": s2, "s3": s3, "s4": s4,
        "lines": source.splitlines(),
    }


def main():
    files = sorted(KERNEL_DIR.glob("*.py"))
    b1, b2, b3, b4 = IndexReplace(), SyncRemove(), MaskBoundary(), LaunchConfigMutate()

    results = []
    for f in files:
        results.append(analyze(f, b1, b2, b3, b4))

    for r in results:
        print(SEP)
        print(f"  File: {r['name']}.py  ({r['loc']} lines)")
        print(SEP)
        print(f"  B1 IndexReplace       : {r['b1']:3d} sites")
        print(f"  B2 SyncRemove         : {r['b2']:3d} sites")
        print(f"  B3 MaskBoundary       : {r['b3']:3d} sites")
        print(f"  B4 LaunchConfigMutate : {r['b4']:3d} sites")

        for label, sites in [("B1", r["s1"]), ("B2", r["s2"]),
                              ("B3", r["s3"]), ("B4", r["s4"])]:
            if sites:
                print(f"  --- {label} samples ---")
                for s in sites[:4]:
                    line = r["lines"][s.line_start - 1].rstrip()
                    print(f"    L{s.line_start:3d}  '{s.original_code[:40]}'  "
                          f"type={s.node_type}  {line[:80]}")
        print()

    # Aggregate
    print(SEP)
    print("  AGGREGATE SUMMARY")
    print(SEP)
    header = f"  {'File':<30s}  {'LOC':>4s}  {'B1':>4s}  {'B2':>4s}  {'B3':>4s}  {'B4':>4s}  {'Total':>5s}"
    print(header)
    print("  " + "-" * (len(header.strip())))
    t = {"b1": 0, "b2": 0, "b3": 0, "b4": 0}
    for r in results:
        total = r["b1"] + r["b2"] + r["b3"] + r["b4"]
        print(f"  {r['name']:<30s}  {r['loc']:>4d}  "
              f"{r['b1']:>4d}  {r['b2']:>4d}  {r['b3']:>4d}  {r['b4']:>4d}  {total:>5d}")
        t["b1"] += r["b1"]; t["b2"] += r["b2"]
        t["b3"] += r["b3"]; t["b4"] += r["b4"]
    total_all = sum(t.values())
    print("  " + "-" * (len(header.strip())))
    print(f"  {'TOTAL':<30s}  {'':>4s}  "
          f"{t['b1']:>4d}  {t['b2']:>4d}  {t['b3']:>4d}  {t['b4']:>4d}  {total_all:>5d}")


if __name__ == "__main__":
    main()
