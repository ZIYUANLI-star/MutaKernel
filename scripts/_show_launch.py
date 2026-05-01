#!/usr/bin/env python3
"""Show kernel launch config and output tensor init for P100, P19, P23."""
import json
from pathlib import Path

BK = json.loads(Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/best_kernels.json").read_text())

for kn in ["L1_P100", "L1_P19", "L1_P23", "L1_P27", "L1_P28"]:
    bki = BK.get(kn, {})
    kpath = Path(bki.get("kernel_path", ""))
    if not kpath.exists():
        print(f"\n=== {kn}: FILE NOT FOUND {kpath}")
        continue
    code = kpath.read_text()
    print(f"\n{'='*60}")
    print(f"  {kn}: {kpath.name}")
    print(f"{'='*60}")
    print(code)
    print(f"\n{'='*60}")
