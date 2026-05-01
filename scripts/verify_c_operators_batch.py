"""Batch verification of Category C operators on real CUDA kernel files.

Run:  python -m scripts.verify_c_operators_batch
"""
import os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mutengine.operators.ml_semantic import (
    StabRemove, AccDowngrade, EpsilonModify,
    ScaleModify, CastRemove, ReductionReorder, InitModify,
)

KERNEL_DIR = Path(__file__).resolve().parent.parent / "test_data" / "cuda_kernels"
SEP = "=" * 78

OPS = [
    ("C1 StabRemove", StabRemove()),
    ("C2 AccDowngrade", AccDowngrade()),
    ("C3 EpsilonModify", EpsilonModify()),
    ("C4 ScaleModify", ScaleModify()),
    ("C5 CastRemove", CastRemove()),
    ("C6 ReductionReorder", ReductionReorder()),
    ("C7 InitModify", InitModify()),
]


def main():
    files = sorted(KERNEL_DIR.glob("*.py"))
    results = []

    for f in files:
        source = f.read_text(encoding="utf-8", errors="replace")
        lines = source.splitlines()
        loc = len(lines)
        row = {"name": f.stem, "loc": loc, "lines": lines}
        for label, op in OPS:
            sites = op.find_sites(source)
            row[label] = sites
            # Test apply on first site
            if sites:
                mutated = op.apply(source, sites[0])
                row[f"{label}_apply"] = mutated != source
            else:
                row[f"{label}_apply"] = None
        results.append(row)

    # Per-file detail
    for r in results:
        print(SEP)
        print(f"  File: {r['name']}.py  ({r['loc']} lines)")
        print(SEP)
        for label, _ in OPS:
            sites = r[label]
            apply_status = r.get(f"{label}_apply")
            astr = "OK" if apply_status is True else ("FAIL" if apply_status is False else "N/A")
            print(f"  {label:<22s}: {len(sites):3d} sites  apply={astr}")
            for s in sites[:3]:
                line = r["lines"][s.line_start - 1].rstrip()
                print(f"    L{s.line_start:3d} '{s.original_code[:50]}'  "
                      f"nt={s.node_type}")
                print(f"         {line[:90]}")
        print()

    # Aggregate summary
    print(SEP)
    print("  AGGREGATE SUMMARY — Category C on 7 real CUDA kernel files")
    print(SEP)
    labels = [l for l, _ in OPS]
    header = f"  {'File':<28s} " + " ".join(f"{l.split()[0]:>4s}" for l in labels) + "  Total"
    print(header)
    print("  " + "-" * (len(header) - 2))

    totals = {l: 0 for l in labels}
    for r in results:
        counts = [len(r[l]) for l in labels]
        total = sum(counts)
        print(f"  {r['name']:<28s} " + " ".join(f"{c:>4d}" for c in counts) + f"  {total:>5d}")
        for l, c in zip(labels, counts):
            totals[l] += c
    grand = sum(totals.values())
    print("  " + "-" * (len(header) - 2))
    print(f"  {'TOTAL':<28s} " + " ".join(f"{totals[l]:>4d}" for l in labels) + f"  {grand:>5d}")

    # Apply verification
    print()
    for label, _ in OPS:
        ok = sum(1 for r in results if r.get(f"{label}_apply") is True)
        has = sum(1 for r in results if r.get(f"{label}_apply") is not None)
        print(f"  {label} apply(): {ok}/{has} files verified")


if __name__ == "__main__":
    main()
