"""Detailed apply() verification for D-category operators."""
import os, sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mutengine.operators.llm_pattern import BroadcastUnsafe, LayoutAssume

KERNEL = (Path(__file__).resolve().parent.parent
          / "test_data" / "cuda_kernels" / "synthetic_d_patterns.py")

OPS = [
    ("D1 BroadcastUnsafe", BroadcastUnsafe()),
    ("D2 LayoutAssume", LayoutAssume()),
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
            mut_line = mut_lines[s.line_start - 1].rstrip()
            print(f"  [{i}] L{s.line_start} nt={s.node_type}")
            print(f"       ORIG: {orig_line.strip()}")
            print(f"       MUT:  {mut_line.strip()}")
            ok = mut_line != orig_line
            print(f"       {'OK' if ok else 'FAIL: lines identical'}")


if __name__ == "__main__":
    main()
