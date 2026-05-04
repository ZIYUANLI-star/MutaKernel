#!/usr/bin/env python3
import json, sys
from pathlib import Path

cp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("第三次实验汇总/results/cuda_l1/checkpoint.json")
if not cp.exists():
    print(f"Not found: {cp}")
    sys.exit(1)

with open(cp, encoding="utf-8") as f:
    data = json.load(f)

print(f"Completed: {len(data)} kernels")
for kid, v in data.items():
    st = v.get("status", "?")
    disc = v.get("total_discrepancies", "?")
    elapsed = v.get("elapsed_s", "?")
    dims = v.get("discrepant_dimensions", [])
    print(f"  {kid}: {st} | disc={disc} | {elapsed}s | dims={dims}")
