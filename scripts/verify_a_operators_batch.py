"""Batch verification of Category A operators on real CUDA kernel files from WSL/KernelBench.

Reads all .py files from test_data/cuda_kernels/ and runs A1/A2/A3 on each.
Reports per-file and aggregate statistics.

Run:  python -m scripts.verify_a_operators_batch
"""
import os
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mutengine.operators.arithmetic import ArithReplace, RelOpReplace, ConstPerturb

KERNEL_DIR = Path(__file__).resolve().parent.parent / "test_data" / "cuda_kernels"

SEP = "=" * 78


def classify_sites(sites):
    ast_sites = [s for s in sites if not s.node_type.startswith("cuda_")]
    cuda_sites = [s for s in sites if s.node_type.startswith("cuda_")]
    return ast_sites, cuda_sites


def analyze_file(filepath: Path, a1: ArithReplace, a2: RelOpReplace, a3: ConstPerturb):
    source = filepath.read_text(encoding="utf-8", errors="replace")
    loc = len(source.splitlines())

    sites_a1 = a1.find_sites(source)
    sites_a2 = a2.find_sites(source)
    sites_a3 = a3.find_sites(source)

    ast_a1, cuda_a1 = classify_sites(sites_a1)
    ast_a2, cuda_a2 = classify_sites(sites_a2)

    # Test apply on first CUDA site of each operator
    apply_ok_a1 = False
    apply_ok_a2 = False
    if cuda_a1:
        mutated = a1.apply(source, cuda_a1[0])
        apply_ok_a1 = mutated != source
    if cuda_a2:
        mutated = a2.apply(source, cuda_a2[0])
        apply_ok_a2 = mutated != source

    return {
        "name": filepath.stem,
        "loc": loc,
        "a1_total": len(sites_a1),
        "a1_ast": len(ast_a1),
        "a1_cuda": len(cuda_a1),
        "a1_apply": apply_ok_a1,
        "a2_total": len(sites_a2),
        "a2_ast": len(ast_a2),
        "a2_cuda": len(cuda_a2),
        "a2_apply": apply_ok_a2,
        "a3_total": len(sites_a3),
        # Detailed breakdowns
        "a1_by_op": defaultdict(int),
        "a2_by_op": defaultdict(int),
        "cuda_a1_sites": cuda_a1[:5],
        "cuda_a2_sites": cuda_a2[:5],
        "source_lines": source.splitlines(),
    }


def main():
    if not KERNEL_DIR.exists():
        print(f"ERROR: {KERNEL_DIR} not found. Copy CUDA kernel files from WSL first.")
        sys.exit(1)

    files = sorted(KERNEL_DIR.glob("*.py"))
    if not files:
        print(f"ERROR: No .py files found in {KERNEL_DIR}")
        sys.exit(1)

    a1 = ArithReplace()
    a2 = RelOpReplace()
    a3 = ConstPerturb()

    results = []
    for f in files:
        r = analyze_file(f, a1, a2, a3)
        results.append(r)

    # ======================= Per-file report =======================
    for r in results:
        print(SEP)
        print(f"  File: {r['name']}.py  ({r['loc']} lines)")
        print(SEP)
        print(f"  A1 ArithReplace : {r['a1_total']:3d} sites  "
              f"(AST={r['a1_ast']}, CUDA={r['a1_cuda']})  "
              f"apply={'OK' if r['a1_apply'] else 'N/A'}")
        print(f"  A2 RelOpReplace : {r['a2_total']:3d} sites  "
              f"(AST={r['a2_ast']}, CUDA={r['a2_cuda']})  "
              f"apply={'OK' if r['a2_apply'] else 'N/A'}")
        print(f"  A3 ConstPerturb : {r['a3_total']:3d} sites")

        # Show sample CUDA hits
        if r["cuda_a1_sites"]:
            print(f"  --- A1 CUDA samples (top {min(3, len(r['cuda_a1_sites']))}) ---")
            for s in r["cuda_a1_sites"][:3]:
                line = r["source_lines"][s.line_start - 1].rstrip()
                print(f"    L{s.line_start:3d} '{s.original_code}' ({s.node_type})  {line}")

        if r["cuda_a2_sites"]:
            print(f"  --- A2 CUDA samples (top {min(3, len(r['cuda_a2_sites']))}) ---")
            for s in r["cuda_a2_sites"][:3]:
                line = r["source_lines"][s.line_start - 1].rstrip()
                print(f"    L{s.line_start:3d} '{s.original_code}' ({s.node_type})  {line}")
        print()

    # ======================= Aggregate summary =======================
    print(SEP)
    print("  AGGREGATE SUMMARY")
    print(SEP)
    print()
    header = f"  {'File':<30s}  {'LOC':>4s}  {'A1':>4s}  {'(AST':>5s}  {'CUDA)':>5s}  {'A2':>4s}  {'(AST':>5s}  {'CUDA)':>5s}  {'A3':>4s}"
    print(header)
    print("  " + "-" * len(header.strip()))

    totals = {"a1": 0, "a1_ast": 0, "a1_cuda": 0,
              "a2": 0, "a2_ast": 0, "a2_cuda": 0,
              "a3": 0, "loc": 0, "files": 0}

    for r in results:
        print(f"  {r['name']:<30s}  {r['loc']:>4d}  "
              f"{r['a1_total']:>4d}  {r['a1_ast']:>5d}  {r['a1_cuda']:>5d}  "
              f"{r['a2_total']:>4d}  {r['a2_ast']:>5d}  {r['a2_cuda']:>5d}  "
              f"{r['a3_total']:>4d}")
        totals["a1"] += r["a1_total"]
        totals["a1_ast"] += r["a1_ast"]
        totals["a1_cuda"] += r["a1_cuda"]
        totals["a2"] += r["a2_total"]
        totals["a2_ast"] += r["a2_ast"]
        totals["a2_cuda"] += r["a2_cuda"]
        totals["a3"] += r["a3_total"]
        totals["loc"] += r["loc"]
        totals["files"] += 1

    print("  " + "-" * len(header.strip()))
    print(f"  {'TOTAL':<30s}  {totals['loc']:>4d}  "
          f"{totals['a1']:>4d}  {totals['a1_ast']:>5d}  {totals['a1_cuda']:>5d}  "
          f"{totals['a2']:>4d}  {totals['a2_ast']:>5d}  {totals['a2_cuda']:>5d}  "
          f"{totals['a3']:>4d}")

    print()
    print(f"  Files tested: {totals['files']}")
    print(f"  A1 CUDA coverage: {totals['a1_cuda']}/{totals['a1']} sites "
          f"({100*totals['a1_cuda']/max(totals['a1'],1):.1f}%)")
    print(f"  A2 CUDA coverage: {totals['a2_cuda']}/{totals['a2']} sites "
          f"({100*totals['a2_cuda']/max(totals['a2'],1):.1f}%)")

    # Verify apply works for each file with CUDA sites
    apply_a1_ok = sum(1 for r in results if r["a1_apply"])
    apply_a2_ok = sum(1 for r in results if r["a2_apply"])
    files_with_cuda_a1 = sum(1 for r in results if r["a1_cuda"] > 0)
    files_with_cuda_a2 = sum(1 for r in results if r["a2_cuda"] > 0)
    print(f"  A1 apply() verified: {apply_a1_ok}/{files_with_cuda_a1} files")
    print(f"  A2 apply() verified: {apply_a2_ok}/{files_with_cuda_a2} files")


if __name__ == "__main__":
    main()
