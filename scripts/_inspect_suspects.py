#!/usr/bin/env python3
"""深查 3 个可疑 mutant + 验证 killing_round 字段语义"""
import json
from pathlib import Path

ROOT = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充')

# 1. 验证 killing_round 字段的默认值
print("="*80)
print("Q1: killing_round 字段在未杀死时是什么值？")
print("="*80)
sample = ROOT / 'task_a_phase2_rerun/details/L1_P100__arith_replace__7.json'
d = json.loads(sample.read_text(encoding='utf-8'))
print(f"  killed={d.get('killed')}  killing_round={d.get('killing_round')!r}  type={type(d.get('killing_round')).__name__}")
print()

# 2. 深查异常案例
suspects = [
    ('task_a_phase2_rerun', 'L1_P24__init_modify__0'),
    ('task_c_phase1_direct', 'L1_P19__mask_boundary__0'),
    ('task_c_phase1_direct', 'L1_P23__const_perturb__3'),
    ('task_c_phase1_direct', 'L1_P24__init_modify__0'),
]

for sub, mid in suspects:
    f = ROOT / sub / 'details' / f'{mid}.json'
    if not f.exists():
        print(f"!! {sub}/{mid} 不存在")
        continue
    d = json.loads(f.read_text(encoding='utf-8'))
    print("="*80)
    print(f"### {sub} / {mid}")
    print("="*80)
    print(f"  killed = {d.get('killed')}   killing_round = {d.get('killing_round')!r}")
    print(f"  operator = {d.get('operator_name')}")
    print(f"  kernel   = {d.get('kernel_name')}")
    print(f"  elapsed  = {d.get('elapsed_sec'):.1f}s")
    print(f"  trigger  = {d.get('trigger')}")
    print(f"  total rounds = {len(d.get('rounds', []))}")
    for r in d.get('rounds', []):
        print(f"\n  --- Round {r.get('round')} ---")
        print(f"    killable        = {r.get('killable')}")
        print(f"    reason_category = {r.get('reason_category')}")
        print(f"    killed          = {r.get('killed')}")
        print(f"    error           = {(r.get('error') or '')[:100]}")
        er = r.get('execution_result')
        if er is None:
            print(f"    execution_result = None")
        elif isinstance(er, dict):
            print(f"    execution_result:")
            for k, v in er.items():
                if k == 'error':
                    print(f"      {k}: {str(v)[:150]}")
                elif k == 'diff_summary':
                    print(f"      {k}: {str(v)[:150]}")
                else:
                    print(f"      {k}: {v}")
        else:
            print(f"    execution_result = {er!r}")
        sc = (r.get('suggested_code') or '').strip()
        if sc:
            print(f"    suggested_code (前 280 char):")
            for line in sc[:280].split('\n'):
                print(f"      | {line}")
    print()
