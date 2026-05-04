"""验证修改后的 C 类算子能否命中 CUDA kernel 中的数值语义模式。

读取 KernelBench 中实际的 LLM 生成 kernel，运行所有 C 类算子，
报告每个算子的命中数量和命中位置。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mutengine.operators.ml_semantic import (
    StabRemove, AccDowngrade, EpsilonModify, ScaleModify,
    CastRemove, ReductionReorder, InitModify,
)

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
RUN_DIR = KB_ROOT / "runs" / "iter_full_l1_caesar_paper_v2"

# Interesting L1 kernels that should have ML numerical patterns
INTERESTING = {
    1: "Square_matmul (tiled, syncthreads)",
    23: "Softmax (expf, -INFINITY, fmaxf, static_cast)",
    24: "LogSoftmax",
    33: "BatchNorm (rsqrt, static_cast)",
    34: "InstanceNorm (rsqrt)",
    35: "GroupNorm (rsqrt)",
    37: "FrobeniusNorm",
    49: "Max reduction (-FLT_MAX, fmaxf)",
}

operators = [
    StabRemove(), AccDowngrade(), EpsilonModify(), ScaleModify(),
    CastRemove(), ReductionReorder(), InitModify(),
]

import json
with open(RUN_DIR / "eval_results.json") as f:
    results = json.load(f)
correct_ids = {int(k) for k, v in results.items() if v.get("correctness", False)}

grand_total = {op.name: 0 for op in operators}
tested = 0

for pid in sorted(INTERESTING.keys()):
    kernel_file = RUN_DIR / f"level_1_problem_{pid}_sample_0_kernel.py"
    if not kernel_file.exists():
        print(f"P{pid}: SKIP (file missing)")
        continue
    if pid not in correct_ids:
        print(f"P{pid}: SKIP (not correct)")
        continue

    code = kernel_file.read_text()
    tested += 1
    print(f"\n{'='*60}")
    print(f"P{pid}: {INTERESTING[pid]}  ({len(code.splitlines())} lines)")
    print(f"{'='*60}")

    total_sites = 0
    for op in operators:
        sites = op.find_sites(code)
        if sites:
            grand_total[op.name] += len(sites)
            total_sites += len(sites)
            print(f"  {op.name}: {len(sites)} sites")
            for s in sites[:5]:
                snippet = s.original_code[:70].replace("\n", " ")
                print(f"    L{s.line_start} [{s.node_type}] `{snippet}`")
            if len(sites) > 5:
                print(f"    ... and {len(sites)-5} more")
    if total_sites == 0:
        print(f"  ** NO C-category sites found **")

# Also run on ALL 63 correct kernels for summary stats
print(f"\n\n{'#'*60}")
print(f"FULL SCAN: all correct L1 kernels")
print(f"{'#'*60}")

all_stats = {op.name: {"total": 0, "kernels": 0} for op in operators}
for pid in sorted(correct_ids):
    kernel_file = RUN_DIR / f"level_1_problem_{pid}_sample_0_kernel.py"
    if not kernel_file.exists():
        continue
    code = kernel_file.read_text()
    for op in operators:
        sites = op.find_sites(code)
        if sites:
            all_stats[op.name]["total"] += len(sites)
            all_stats[op.name]["kernels"] += 1

print(f"\n{'Operator':<22} {'Sites':>8} {'Kernels':>10}")
print("-" * 42)
total_all = 0
for op in operators:
    s = all_stats[op.name]
    total_all += s["total"]
    k_pct = f"({s['kernels']}/63)" if s['kernels'] > 0 else ""
    print(f"  {op.name:<20} {s['total']:>6}   {k_pct:>10}")
print("-" * 42)
print(f"  {'TOTAL':<20} {total_all:>6}")
