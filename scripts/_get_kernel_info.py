import json
from pathlib import Path

with open("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/best_kernels.json") as f:
    d = json.load(f)

for k in ["L1_P23", "L1_P24"]:
    info = d.get(k, {})
    kp = info.get("kernel_path", "?")
    # Extract problem name from the problem dir
    pid = info.get("problem_id", "?")
    level = info.get("level", "?")
    # Find problem file to get the problem name
    pdir = Path(f"/home/kbuser/projects/KernelBench-0/KernelBench/level1")
    pname = "?"
    for f2 in pdir.iterdir():
        if f2.name.startswith(f"{pid}_"):
            pname = f2.stem
            break
    print(f"{k}: problem={pname}, speedup={info.get('speedup','?')}")
    # Show first few lines of kernel
    kpath = Path(kp)
    if kpath.exists():
        lines = kpath.read_text()[:500]
        print(f"  Kernel preview:\n{lines}\n")
