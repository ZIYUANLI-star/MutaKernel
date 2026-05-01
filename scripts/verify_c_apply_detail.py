"""Detailed apply() verification for C-category operators."""
import os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mutengine.operators.ml_semantic import (
    StabRemove, AccDowngrade, EpsilonModify,
    ScaleModify, CastRemove, ReductionReorder, InitModify,
)

KERNEL = (Path(__file__).resolve().parent.parent
          / "test_data" / "cuda_kernels" / "synthetic_softmax_layernorm.py")

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
    source = KERNEL.read_text(encoding="utf-8")
    lines = source.splitlines()

    for label, op in OPS:
        sites = op.find_sites(source)
        if not sites:
            continue
        print(f"\n{'=' * 70}")
        print(f"  {label}  ({len(sites)} sites)")
        print(f"{'=' * 70}")
        for i, s in enumerate(sites):
            mutated = op.apply(source, s)
            if mutated == source:
                print(f"  [{i}] NO CHANGE (potential bug!)  nt={s.node_type}")
                continue
            orig_line = lines[s.line_start - 1].rstrip()
            mut_lines = mutated.splitlines()
            mut_line = mut_lines[s.line_start - 1].rstrip() if s.line_start <= len(mut_lines) else "???"
            print(f"  [{i}] L{s.line_start} nt={s.node_type}")
            print(f"       ORIG: {orig_line.strip()}")
            print(f"       MUT:  {mut_line.strip()}")
            ok = mut_line != orig_line
            print(f"       {'OK' if ok else 'FAIL: lines identical'}")


if __name__ == "__main__":
    main()
