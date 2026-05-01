#!/usr/bin/env python3
import json, sys
from pathlib import Path

d = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/full_block12_results/details")
if not d.exists():
    print("No results yet"); sys.exit()

for f in sorted(d.glob("*.json")):
    data = json.load(open(f))
    s = data["summary"]
    k = f.stem
    print(f"{k:12s}  total={s['total']:2d}  K={s['killed']:2d}  "
          f"S={s['survived']:2d}  SB={s['stillborn']:2d}  "
          f"EQ={s['equivalent']:2d}  Score={s['mutation_score']:.2%}")
