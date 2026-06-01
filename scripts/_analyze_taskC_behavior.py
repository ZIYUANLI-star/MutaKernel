#!/usr/bin/env python3
"""分析 Task C 已完成 mutant 的 LLM 行为"""
import json
import glob
from collections import Counter

ROOT = '/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct'
files = sorted(glob.glob(f'{ROOT}/details/*.json'))

killed_mutants = []
unkilled = []
reason_categories = Counter()
total_rounds_used = 0
rounds_dist = Counter()
killable_true_unkilled = []
killable_false = []

for f in files:
    d = json.load(open(f, encoding='utf-8'))
    mid = d['mutant_id']
    rounds = d.get('rounds', [])
    n_rounds = len(rounds)
    total_rounds_used += n_rounds
    rounds_dist[n_rounds] += 1
    killed = d.get('killed', False)
    killing_round = d.get('killing_round')
    
    if killed:
        killed_mutants.append((mid, killing_round, n_rounds))
    else:
        unkilled.append((mid, n_rounds))
    
    for r in rounds:
        cat = r.get('reason_category') or r.get('category')
        if cat:
            reason_categories[cat] += 1
    
    if rounds:
        last = rounds[-1]
        if not killed:
            if last.get('killable') is False:
                killable_false.append((mid, last.get('reason_category')))
            elif last.get('killable') is True:
                killable_true_unkilled.append((mid, n_rounds, last.get('reason_category')))

print(f"=== Task C: {len(files)} mutants done ===")
print(f"已杀死:      {len(killed_mutants)}")
print(f"未杀死:      {len(unkilled)}")
print(f"  其中 LLM 判 killable=False (早停): {len(killable_false)}")
print(f"  其中 LLM 判 killable=True 但 5 轮没杀: {len(killable_true_unkilled)}")
print(f"总 round 数: {total_rounds_used}")
print(f"平均 round/mutant: {total_rounds_used/max(1,len(files)):.2f}")
print()
print("--- rounds 数分布 ---")
for r, n in sorted(rounds_dist.items()):
    print(f"  跑了 {r} 轮: {n} 个 mutant")
print()
print("--- reason_category 分布 ---")
for cat, n in reason_categories.most_common():
    print(f"  {n:3d}  {cat}")
print()
print("--- 已杀死的 mutant ---")
for it in killed_mutants:
    print(f"  {it}  (mid, killing_round, total_rounds)")
print()
print("--- LLM 判 killable=True 但没杀掉的 ---")
for it in killable_true_unkilled[:10]:
    print(f"  {it}")
