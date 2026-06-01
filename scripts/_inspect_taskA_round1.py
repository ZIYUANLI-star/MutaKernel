#!/usr/bin/env python3
"""看 Task A 第一个 mutant 的完整 round 1 内容（LLM 说了什么）"""
import json
import glob

ROOT = '/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun'
files = sorted(glob.glob(f'{ROOT}/details/*.json'))

# 抽 3 个不同 reason_category 的样本
samples_seen = {}
for f in files:
    d = json.load(open(f, encoding='utf-8'))
    r = d['rounds'][0]
    cat = r.get('reason_category')
    if cat and cat not in samples_seen:
        samples_seen[cat] = (d['mutant_id'], f, d, r)
    if len(samples_seen) >= 5:
        break

for cat, (mid, f, d, r) in samples_seen.items():
    print('='*80)
    print(f"category = {cat}   mutant = {mid}")
    print('='*80)
    print(f"killable                = {r.get('killable')}")
    print(f"prompt_type             = {r.get('prompt_type')}")
    print(f"execution_result        = {r.get('execution_result')}")
    print(f"killed                  = {r.get('killed')}")
    print()
    print('--- survival_reason ---')
    print(r.get('survival_reason'))
    print()
    print('--- kill_strategy ---')
    print(r.get('kill_strategy'))
    print()
    print('--- recommendations ---')
    recs = r.get('recommendations')
    if isinstance(recs, list):
        for rec in recs[:3]:
            print(f"  {rec}")
    else:
        print(recs)
    print()
    print('--- suggested_code (前 500 char) ---')
    sc = r.get('suggested_code') or ''
    print(sc[:500])
    print()
    print()
