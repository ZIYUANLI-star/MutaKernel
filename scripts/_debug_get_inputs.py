"""Debug: check why get_inputs extraction fails for most kernels."""
import json
from pathlib import Path

PROJECT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
bk = json.loads((PROJECT / "best_kernels.json").read_text())

for kn in ["L1_P1", "L1_P2", "L1_P100", "L2_P1", "L2_P9"]:
    entry = bk.get(kn, {})
    kpath = entry.get("kernel_path", "")
    p = Path(kpath)
    if p.exists():
        code = p.read_text()
        lines = code.split("\n")
        gi_lines = [i for i, l in enumerate(lines) if "def get_inputs" in l]
        print(f"{kn}: exists=True, lines={len(lines)}, get_inputs at lines={gi_lines}")
        if gi_lines:
            start = gi_lines[0]
            for j in range(max(0, start - 3), min(start + 10, len(lines))):
                print(f"  {j}: {lines[j]}")
        print()
    else:
        print(f"{kn}: path={kpath}, exists=False")
        print()
