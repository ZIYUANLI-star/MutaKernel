#!/usr/bin/env python3
"""Check L2 data from Block 1-2 source results for survived mutants."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B12 = ROOT / "第二次实验汇总" / "full_block12_results" / "details"

targets = {
    "L1_P100__arith_replace__7",
    "L1_P16__arith_replace__23",
    "L1_P40__const_perturb__2",
    "L1_P89__arith_replace__10",
    "L1_P28__arith_replace__9",
}

found = 0
for jf in sorted(B12.glob("*.json")):
    data = json.loads(jf.read_text())
    for m in data.get("mutants", []):
        mid = m.get("id", "")
        if mid in targets:
            found += 1
            ed = m.get("equiv_detail", {})
            l2 = ed.get("layer2", {})
            print(f"\n{'='*60}")
            print(f"Source: {jf.name}")
            print(f"Mutant: {mid}")
            print(f"Status: {m.get('status')}")
            print(f"Layer 2 data:")
            print(json.dumps(l2, indent=2, default=str))
            if found >= 5:
                break
    if found >= 5:
        break

if found == 0:
    print("No target mutants found in block12 results")
