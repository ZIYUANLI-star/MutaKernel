#!/usr/bin/env python3
"""分析 Task A 已完成 mutant 的 LLM 行为：早停 vs 跑满 5 轮"""
import json
import glob
from collections import Counter

ROOT = '/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun'
files = sorted(glob.glob(f'{ROOT}/details/*.json'))

early_stop = []
full_5_rounds = []
reason_categories = Counter()
total_inputs_tried = 0
inputs_per_mutant = []

for f in files:
    d = json.load(open(f, encoding='utf-8'))
    mid = d['mutant_id']
    rounds = d.get('rounds', [])
    n_rounds = len(rounds)
    
    inputs_in_this_mutant = 0
    last_reason = None
    last_verdict = None
    for r in rounds:
        inputs = r.get('candidate_inputs') or r.get('inputs') or []
        inputs_in_this_mutant += len(inputs) if isinstance(inputs, list) else 0
        cat = r.get('reason_category') or r.get('category')
        if cat:
            reason_categories[cat] += 1
            last_reason = cat
        if 'verdict' in r:
            last_verdict = r.get('verdict')
    
    total_inputs_tried += inputs_in_this_mutant
    inputs_per_mutant.append(inputs_in_this_mutant)
    
    if n_rounds < 5:
        early_stop.append((mid, n_rounds, last_reason, last_verdict))
    else:
        full_5_rounds.append((mid, last_reason, last_verdict, inputs_in_this_mutant))

print(f"=== Task A: {len(files)} mutants done, all killed=False ===")
print(f"早停 (rounds < 5): {len(early_stop)}")
print(f"跑满 5 轮:        {len(full_5_rounds)}")
print(f"总尝试输入数:     {total_inputs_tried}")
print(f"平均每 mutant:    {sum(inputs_per_mutant)/max(1,len(inputs_per_mutant)):.1f} 个候选输入")
print()
print("--- reason_category 分布 ---")
for cat, n in reason_categories.most_common():
    print(f"  {n:3d}  {cat}")

print()
print("--- 早停明细（前 10） ---")
for it in early_stop[:10]:
    print(f"  {it}")

print()
print("--- 跑满 5 轮明细（前 10） ---")
for it in full_5_rounds[:10]:
    print(f"  {it}")

print()
print("--- 单个样本结构示例（第 1 个 mutant 的 rounds 字段） ---")
if files:
    d = json.load(open(files[0], encoding='utf-8'))
    print(f"  mutant_id = {d['mutant_id']}")
    print(f"  killed    = {d.get('killed')}")
    print(f"  顶层 keys = {list(d.keys())}")
    for i, r in enumerate(d.get('rounds', [])):
        print(f"  round {r.get('round', i)} keys: {list(r.keys())}")
        for k in ['reason_category', 'category', 'verdict', 'recommendation', 'should_stop', 'stop', 'final']:
            if k in r:
                v = r[k]
                vs = str(v)[:200]
                print(f"     {k} = {vs}")
