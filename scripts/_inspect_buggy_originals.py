"""Inspect what is inside buggy_kernel of TRIVIAL_SHELL cases.
Print first 50 lines of each."""
import json
from pathlib import Path
BEST = json.loads(Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/best_kernels.json").read_text())

TARGETS = ["L1_P15", "L1_P16", "L1_P18", "L1_P48", "L1_P89", "L1_P91", "L1_P98", "L2_P9", "L2_P58"]

for name in TARGETS:
    if name not in BEST: continue
    p = Path(BEST[name]["kernel_path"])
    code = p.read_text(encoding="utf-8")
    n_global = code.count("__global__")
    has_triton = "@triton" in code or "tl.load" in code
    has_load_inline = "load_inline" in code
    has_cublas = "cublas" in code.lower() or "torch::mm(" in code or "torch::matmul(" in code
    print(f"\n{'='*70}")
    print(f"=== {name}  (KB speedup={BEST[name]['speedup']}) ===")
    print(f"  __global__: {n_global}, triton: {has_triton}, load_inline: {has_load_inline}, cublas/torch::mm: {has_cublas}")
    print(f"  path: {p.name}")
    print(f"{'='*70}")
    print("\n".join(code.splitlines()[:35]))
    print("...\n")
