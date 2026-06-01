#!/usr/bin/env python3
"""正确的统计：仅看 Phase II main_track 的 value_stress / training_stress
下原 kernel `o` 失败的 policy 列表，按 kernel 去重。

排除来源：
- Task A/C（这些 LLM 对抗输入大多 out-of-contract，不算 o 的真 bug）
- Phase II 的 llm_iterative_analysis（也是 LLM 对抗输入）

只信确定性 stress（hardcoded policy + 标准 input 形状）下的 o 失败。
"""
import json
from pathlib import Path
from collections import defaultdict, Counter

P2_DIR = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/stress_enhance_results/details')

# kernel -> set of failed policies (from value_stress) + set of failed policies (training_stress)
kernel_value_fails = defaultdict(set)
kernel_train_fails = defaultdict(set)
# 也记录原 kernel 整体在哪些 mutant 文件上看到了 failure
kernel_seen = set()

for f in sorted(P2_DIR.glob('*.json')):
    try:
        d = json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        continue
    kernel = d.get('kernel_name')
    if not kernel:
        continue
    kernel_seen.add(kernel)
    mt = d.get('main_track', {}) or {}
    vs = mt.get('value_stress', {}) or {}
    ts = mt.get('training_stress', {}) or {}
    for p in (vs.get('original_failures') or []):
        kernel_value_fails[kernel].add(p)
    for p in (ts.get('original_failures') or []):
        kernel_train_fails[kernel].add(p)

# 输出
all_buggy_kernels = sorted(set(kernel_value_fails.keys()) | set(kernel_train_fails.keys()))
print(f"=== 统计基础 ===")
print(f"Phase II 覆盖的 unique kernel 数: {len(kernel_seen)}")
print(f"原 kernel 在 stress 下有 failures 的 kernel 数: {len(all_buggy_kernels)}")
print()
print(f"=== 全部 buggy kernel（按 value_stress 失败 policy 数排序）===")
print(f"{'kernel':12s} | value_stress failed policies     | training_stress failed policies")
print('-' * 90)
sorted_kernels = sorted(all_buggy_kernels,
                       key=lambda k: -(len(kernel_value_fails.get(k, set())) + len(kernel_train_fails.get(k, set()))))
for k in sorted_kernels:
    vs_set = sorted(kernel_value_fails.get(k, set()))
    ts_set = sorted(kernel_train_fails.get(k, set()))
    vs_str = ', '.join(vs_set) if vs_set else '-'
    ts_str = ', '.join(ts_set) if ts_set else '-'
    print(f"{k:12s} | {vs_str[:50]:50s} | {ts_str}")

print()
print(f"=== 所有出现过的失败 policy 名（汇总）===")
all_vs_pol = Counter()
all_ts_pol = Counter()
for k, s in kernel_value_fails.items():
    for p in s:
        all_vs_pol[p] += 1
for k, s in kernel_train_fails.items():
    for p in s:
        all_ts_pol[p] += 1

print("\n  value_stress policies (kernel 数):")
for p, n in all_vs_pol.most_common():
    print(f"    {n:3d}  {p}")
print("\n  training_stress policies (kernel 数):")
for p, n in all_ts_pol.most_common():
    print(f"    {n:3d}  {p}")
