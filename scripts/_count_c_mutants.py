#!/usr/bin/env python3
"""计算采样 1/op 策略下，实际会产生多少个 C/D 变异体。"""
import json, sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
sys.path.insert(0, str(PROJECT_ROOT))

from src.mutengine.operators.ml_semantic import (
    StabRemove, AccDowngrade, EpsilonModify, ScaleModify,
    CastRemove, ReductionReorder, InitModify,
)
from src.mutengine.operators.llm_pattern import BroadcastUnsafe, LayoutAssume

C_OPS = [
    ("C1", "stab_remove", StabRemove()),
    ("C2", "acc_downgrade", AccDowngrade()),
    ("C3", "epsilon_modify", EpsilonModify()),
    ("C4", "scale_modify", ScaleModify()),
    ("C5", "cast_remove", CastRemove()),
    ("C6", "reduction_reorder", ReductionReorder()),
    ("C7", "init_modify", InitModify()),
]
D_OPS = [
    ("D1", "broadcast_unsafe", BroadcastUnsafe()),
    ("D2", "layout_assume", LayoutAssume()),
]

with open(PROJECT_ROOT / "best_kernels.json") as f:
    best_kernels = json.load(f)

# Per kernel: count how many C/D operators have >=1 site (= actual mutants sampled)
c_mutants_total = 0
d_mutants_total = 0
c_per_op = defaultdict(int)
d_per_op = defaultdict(int)

kernel_c_detail = {}

for key in sorted(best_kernels.keys()):
    kpath = Path(best_kernels[key]["kernel_path"])
    if not kpath.exists():
        continue
    source = kpath.read_text(encoding="utf-8", errors="replace")

    c_ops_with_sites = []
    for label, name, op in C_OPS:
        try:
            sites = op.find_sites(source)
        except Exception:
            sites = []
        if len(sites) > 0:
            c_mutants_total += 1  # sample 1 per op
            c_per_op[label] += 1
            c_ops_with_sites.append(f"{label}({len(sites)}sites)")

    d_ops_with_sites = []
    for label, name, op in D_OPS:
        try:
            sites = op.find_sites(source)
        except Exception:
            sites = []
        if len(sites) > 0:
            d_mutants_total += 1
            d_per_op[label] += 1
            d_ops_with_sites.append(f"{label}({len(sites)}sites)")

    if c_ops_with_sites or d_ops_with_sites:
        kernel_c_detail[key] = (c_ops_with_sites, d_ops_with_sites)

print(f"采样策略: 1 mutant / operator / kernel")
print(f"")
print(f"C 类实际变异体数: {c_mutants_total}")
print(f"D 类实际变异体数: {d_mutants_total}")
print(f"C+D 合计: {c_mutants_total + d_mutants_total}")
print()

print(f"C 类按算子分解:")
for label, name, _ in C_OPS:
    print(f"  {label} {name:<22} → {c_per_op[label]:>3} 个变异体 (在 {c_per_op[label]} 个 kernel 上)")

print(f"\nD 类按算子分解:")
for label, name, _ in D_OPS:
    print(f"  {label} {name:<22} → {d_per_op[label]:>3} 个变异体 (在 {d_per_op[label]} 个 kernel 上)")

print(f"\n各 kernel 的 C/D 算子分布:")
for key, (c_list, d_list) in sorted(kernel_c_detail.items()):
    parts = ", ".join(c_list + d_list)
    print(f"  {key:<12} → {parts}")
