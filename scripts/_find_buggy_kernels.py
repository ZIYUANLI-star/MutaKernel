#!/usr/bin/env python3
"""调研：找出在 Phase II / Task A / Task C 中，哪些 kernel 出现了 'original_ok=False'
即"原 kernel 在某个 LLM 对抗输入下自己也算不对" → 这就是导师说的"正确性有误"的算子。

输出：
1. 受影响的 kernel 列表（unique kernels）
2. 各 kernel 失败次数3. 几个典型 case 的详细信息（输入代码、ref/orig/mut diff）"""
import json
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel')
SUMMARY = ROOT / '第二次实验汇总'
SUPPL = SUMMARY / '第二次实验汇总_补充'

# 收集所有 original_ok=False 的 case
findings = []  # (source, mutant_id, kernel, round, exec_result)

# --- Phase II: stress_enhance_results/details/ ---
for f in sorted((SUMMARY / 'stress_enhance_results/details').glob('*.json')):
    try:
        d = json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        continue
    mid = d.get('mutant_id', f.stem)
    kernel = d.get('kernel_name', mid.split('__')[0])
    
    # Phase II 有几种结构，我先扫所有可能字段
    # llm_iterative_analysis.rounds[*].execution_result
    llm_iter = d.get('llm_iterative_analysis', {}) or {}
    for r in (llm_iter.get('rounds') or []):
        er = r.get('execution_result') or {}
        if isinstance(er, dict) and er.get('original_ok') is False:
            findings.append({
                'source': 'PhaseII_LLM_iter',
                'mutant_id': mid,
                'kernel': kernel,
                'round': r.get('round'),
                'er': er,
                'sc': r.get('suggested_code', '')[:300],
            })
    
    # main_track / config_track 里的 stress 测试也可能记 original_failures
    for track_name in ('main_track', 'config_track'):
        track = d.get(track_name, {}) or {}
        # 各 policy 下的 original_failures
        for policy_name, policy_data in (track.items() if isinstance(track, dict) else []):
            if not isinstance(policy_data, dict):
                continue
            of = policy_data.get('original_failures') or policy_data.get('original_failed_seeds')
            if of:
                findings.append({
                    'source': f'PhaseII_{track_name}/{policy_name}',
                    'mutant_id': mid,
                    'kernel': kernel,
                    'round': None,
                    'er': {'original_failures': of},
                    'sc': '',
                })

# --- Task A ---
for f in sorted((SUPPL / 'task_a_phase2_rerun/details').glob('*.json')):
    try:
        d = json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        continue
    mid = d.get('mutant_id', f.stem)
    kernel = d.get('kernel_name') or mid.split('__')[0]
    for r in d.get('rounds', []):
        er = r.get('execution_result') or {}
        if isinstance(er, dict) and er.get('original_ok') is False:
            findings.append({
                'source': 'TaskA',
                'mutant_id': mid,
                'kernel': kernel,
                'round': r.get('round'),
                'er': er,
                'sc': (r.get('suggested_code') or '')[:300],
            })

# --- Task C ---
for f in sorted((SUPPL / 'task_c_phase1_direct/details').glob('*.json')):
    try:
        d = json.loads(f.read_text(encoding='utf-8'))
    except Exception:
        continue
    mid = d.get('mutant_id', f.stem)
    kernel = d.get('kernel_name') or mid.split('__')[0]
    for r in d.get('rounds', []):
        er = r.get('execution_result') or {}
        if isinstance(er, dict) and er.get('original_ok') is False:
            findings.append({
                'source': 'TaskC',
                'mutant_id': mid,
                'kernel': kernel,
                'round': r.get('round'),
                'er': er,
                'sc': (r.get('suggested_code') or '')[:300],
            })

print(f"=" * 80)
print(f"Total events with original_ok=False: {len(findings)}")
print(f"=" * 80)

# 按来源分组
by_source = Counter(x['source'] for x in findings)
print("\n## 来源分布")
for src, n in by_source.most_common():
    print(f"  {n:4d}  {src}")

# 按 kernel 分组（unique kernel 数）
by_kernel = defaultdict(list)
for x in findings:
    by_kernel[x['kernel']].append(x)

print(f"\n## 受影响的 unique kernel 数: {len(by_kernel)}")
print("\n## Top 20 kernel by failure count")
for k, lst in sorted(by_kernel.items(), key=lambda x: -len(x[1]))[:20]:
    src_cnt = Counter(x['source'] for x in lst)
    print(f"  {k:15s} : {len(lst):3d} events  ({dict(src_cnt)})")

# 看几个具体 case
print("\n" + "=" * 80)
print("3 个具体 case 的详细信息")
print("=" * 80)
for x in findings[:3]:
    print(f"\n--- {x['source']} | {x['mutant_id']} | round {x['round']} ---")
    er = x['er']
    print(f"  ref_ok      = {er.get('ref_ok')}")
    print(f"  original_ok = {er.get('original_ok')}")
    print(f"  mutant_ok   = {er.get('mutant_ok')}")
    print(f"  killed      = {er.get('killed')}")
    print(f"  diff_summary= {(er.get('diff_summary') or '')[:200]}")
    print(f"  error       = {(er.get('error') or '')[:120]}")
    if x['sc']:
        print(f"  对抗输入 (前 280 char):")
        for line in x['sc'].split('\n')[:8]:
            print(f"    | {line}")

# 全部受影响 kernel 排序
print("\n" + "=" * 80)
print(f"全部 {len(by_kernel)} 个受影响的 kernel（按失败次数降序）")
print("=" * 80)
all_affected = sorted(by_kernel.items(), key=lambda x: -len(x[1]))
for k, lst in all_affected:
    print(f"  {k:15s} : {len(lst):3d} events")
