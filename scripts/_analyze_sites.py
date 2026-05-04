#!/usr/bin/env python3
"""分析所有 90 个 best kernel 上各算子能找到的变异位点数量。"""
import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
sys.path.insert(0, str(PROJECT_ROOT))

from src.mutengine.operators.arithmetic import ArithReplace, RelOpReplace, ConstPerturb
from src.mutengine.operators.gpu_parallel import IndexReplace, SyncRemove, MaskBoundary, LaunchConfigMutate
from src.mutengine.operators.ml_semantic import (
    StabRemove, AccDowngrade, EpsilonModify, ScaleModify,
    CastRemove, ReductionReorder, InitModify,
)
from src.mutengine.operators.llm_pattern import BroadcastUnsafe, LayoutAssume

ALL_OPS = [
    ("A1", "arith_replace", ArithReplace()),
    ("A2", "relop_replace", RelOpReplace()),
    ("A3", "const_perturb", ConstPerturb()),
    ("B1", "index_replace", IndexReplace()),
    ("B2", "sync_remove", SyncRemove()),
    ("B3", "mask_boundary", MaskBoundary()),
    ("B4", "launch_config_mutate", LaunchConfigMutate()),
    ("C1", "stab_remove", StabRemove()),
    ("C2", "acc_downgrade", AccDowngrade()),
    ("C3", "epsilon_modify", EpsilonModify()),
    ("C4", "scale_modify", ScaleModify()),
    ("C5", "cast_remove", CastRemove()),
    ("C6", "reduction_reorder", ReductionReorder()),
    ("C7", "init_modify", InitModify()),
    ("D1", "broadcast_unsafe", BroadcastUnsafe()),
    ("D2", "layout_assume", LayoutAssume()),
]

BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
with open(BEST_KERNELS_FILE) as f:
    best_kernels = json.load(f)

# Count sites per operator across all kernels
op_kernel_count = defaultdict(int)  # op -> how many kernels have >= 1 site
op_total_sites = defaultdict(int)   # op -> total sites across all kernels
kernel_cat_sites = {}  # kernel -> {cat -> site_count}

cat_kernel_count = defaultdict(int)
cat_total_sites = defaultdict(int)

for key in sorted(best_kernels.keys()):
    info = best_kernels[key]
    kpath = Path(info["kernel_path"])
    if not kpath.exists():
        continue
    source = kpath.read_text(encoding="utf-8", errors="replace")

    cat_sites = defaultdict(int)

    for label, name, op in ALL_OPS:
        cat = label[0]
        try:
            sites = op.find_sites(source)
        except Exception:
            sites = []
        n = len(sites)
        op_total_sites[name] += n
        cat_sites[cat] += n
        cat_total_sites[cat] += n
        if n > 0:
            op_kernel_count[name] += 1

    for cat in "ABCD":
        if cat_sites[cat] > 0:
            cat_kernel_count[cat] += 1

    kernel_cat_sites[key] = dict(cat_sites)

total_kernels = len(kernel_cat_sites)

print(f"{'='*70}")
print(f"  变异位点分析 (全部 {total_kernels} 个 kernel)")
print(f"{'='*70}")

print(f"\n  按类别:")
print(f"  {'Cat':<5} {'有位点的kernel数':>18} {'总位点数':>10}")
print(f"  {'-'*35}")
for cat in "ABCD":
    print(f"  {cat:<5} {cat_kernel_count[cat]:>10}/{total_kernels:<6} {cat_total_sites[cat]:>10}")

print(f"\n  按算子:")
print(f"  {'Label':<5} {'Operator':<26} {'有位点的kernel数':>18} {'总位点数':>10}")
print(f"  {'-'*62}")
for label, name, op in ALL_OPS:
    print(f"  {label:<5} {name:<26} {op_kernel_count[name]:>10}/{total_kernels:<6} {op_total_sites[name]:>10}")

# Show which kernels have C/D sites
print(f"\n{'='*70}")
print(f"  有 C 类位点的 kernel:")
print(f"{'='*70}")
for key in sorted(kernel_cat_sites.keys()):
    cs = kernel_cat_sites[key]
    if cs.get("C", 0) > 0:
        print(f"  {key}: C={cs['C']}")

print(f"\n{'='*70}")
print(f"  有 D 类位点的 kernel:")
print(f"{'='*70}")
for key in sorted(kernel_cat_sites.keys()):
    cs = kernel_cat_sites[key]
    if cs.get("D", 0) > 0:
        print(f"  {key}: D={cs['D']}")

# Check why: sample a few kernels and show what patterns C/D look for
print(f"\n{'='*70}")
print(f"  L1 vs L2 对比")
print(f"{'='*70}")
for level in ["L1", "L2"]:
    lk = {k: v for k, v in kernel_cat_sites.items() if k.startswith(level)}
    n = len(lk)
    c_has = sum(1 for v in lk.values() if v.get("C", 0) > 0)
    d_has = sum(1 for v in lk.values() if v.get("D", 0) > 0)
    c_total = sum(v.get("C", 0) for v in lk.values())
    d_total = sum(v.get("D", 0) for v in lk.values())
    print(f"  {level}: {n} kernels, C有位点={c_has} ({c_total}个), D有位点={d_has} ({d_total}个)")

# Language distribution
print(f"\n{'='*70}")
print(f"  语言分布")
print(f"{'='*70}")
cuda_count = 0
triton_count = 0
for key in sorted(best_kernels.keys()):
    kpath = Path(best_kernels[key]["kernel_path"])
    if not kpath.exists():
        continue
    source = kpath.read_text(encoding="utf-8", errors="replace")
    indicators = ["__global__", "__device__", "load_inline", "cuda_source"]
    is_cuda = sum(1 for ind in indicators if ind in source) >= 2
    if is_cuda:
        cuda_count += 1
    else:
        triton_count += 1
print(f"  CUDA: {cuda_count}, Triton: {triton_count}")
