#!/usr/bin/env python3
"""Scan best_kernels.json to find kernels with C/D operator mutation sites."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mutengine.operators.ml_semantic import (
    StabRemove, AccDowngrade, EpsilonModify, ScaleModify,
    CastRemove, ReductionReorder, InitModify,
)
from src.mutengine.operators.llm_pattern import BroadcastUnsafe, LayoutAssume

with open(Path(__file__).resolve().parent.parent / "best_kernels.json") as f:
    bk = json.load(f)

cd_ops = [
    ("C1", StabRemove()), ("C2", AccDowngrade()), ("C3", EpsilonModify()),
    ("C4", ScaleModify()), ("C5", CastRemove()), ("C6", ReductionReorder()),
    ("C7", InitModify()), ("D1", BroadcastUnsafe()), ("D2", LayoutAssume()),
]

results = []
for key in sorted(bk.keys()):
    kp = Path(bk[key]["kernel_path"])
    if not kp.exists():
        continue
    src = kp.read_text(errors="replace")
    hits = {}
    for label, op in cd_ops:
        try:
            sites = op.find_sites(src)
            if sites:
                hits[label] = len(sites)
        except Exception:
            pass
    if hits:
        results.append((key, hits, bk[key].get("speedup", 0)))

results.sort(key=lambda x: -len(x[1]))
print(f"Found {len(results)} kernels with C/D sites:\n")
for key, hits, spd in results[:20]:
    h = ", ".join(f"{k}={v}" for k, v in sorted(hits.items()))
    print(f"  {key:12s}  spd={spd:.3f}  {h}")
