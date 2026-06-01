#!/usr/bin/env python3
"""深查 ThrottlingException 的影响范围"""
import json
import glob
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充')

def analyze(name):
    sub = ROOT / name
    files = sorted(sub.glob('details/*.json'))
    print('='*90)
    print(f"### {name}  ({len(files)} files)")
    print('='*90)

    # 按 mutant 级别统计
    mutants_with_throttle = set()
    mutants_all_throttled = []     # 所有 round 全部 throttled
    mutants_partial_throttled = []  # 部分 round throttled
    rounds_throttled = 0
    rounds_total = 0
    rounds_with_real_llm = 0
    
    # 时间分布：throttle 集中在哪批 mutants
    throttle_by_kernel = Counter()
    throttle_by_operator = Counter()
    
    no_throttle_kill = 0    # 没有 throttle 且 killed
    has_throttle_kill = 0   # 有 throttle 但 killed
    no_throttle_unkilled = 0
    has_throttle_unkilled = 0
    
    for f in files:
        d = json.loads(f.read_text(encoding='utf-8'))
        mid = d['mutant_id']
        kernel = d.get('kernel_name', '?')
        operator = d.get('operator_name', '?')
        rounds = d.get('rounds', [])
        killed = d.get('killed', False)
        
        n_throttle = 0
        n_total = len(rounds)
        for r in rounds:
            rounds_total += 1
            err = r.get('error') or ''
            if 'ThrottlingException' in err:
                n_throttle += 1
                rounds_throttled += 1
                throttle_by_kernel[kernel] += 1
                throttle_by_operator[operator] += 1
            else:
                rounds_with_real_llm += 1
        
        if n_throttle > 0:
            mutants_with_throttle.add(mid)
            if n_throttle == n_total:
                mutants_all_throttled.append(mid)
            else:
                mutants_partial_throttled.append((mid, n_throttle, n_total))
            if killed:
                has_throttle_kill += 1
            else:
                has_throttle_unkilled += 1
        else:
            if killed:
                no_throttle_kill += 1
            else:
                no_throttle_unkilled += 1
    
    print(f"\n📊 Round 级别:")
    print(f"  总 round 数:                      {rounds_total}")
    print(f"  被 ThrottlingException 拦截 round: {rounds_throttled}  ({rounds_throttled*100/max(1,rounds_total):.1f}%)")
    print(f"  有效 LLM 响应 round:              {rounds_with_real_llm}")
    
    print(f"\n📊 Mutant 级别 ({len(files)} 个):")
    print(f"  至少 1 round 被 throttle:    {len(mutants_with_throttle)}  ({len(mutants_with_throttle)*100/len(files):.1f}%)")
    print(f"  所有 round 都被 throttle:     {len(mutants_all_throttled)}  ⚠️ 这些 mutant 完全没拿到 LLM 任何响应")
    print(f"  部分 round 被 throttle:       {len(mutants_partial_throttled)}")
    
    print(f"\n📊 与 killed 交叉:")
    print(f"  无 throttle 且 killed:    {no_throttle_kill}")
    print(f"  无 throttle 且 未杀死:    {no_throttle_unkilled}")
    print(f"  有 throttle 且 killed:    {has_throttle_kill}")
    print(f"  有 throttle 且 未杀死:    {has_throttle_unkilled}  WARN: 这部分未杀死的可信度受限")
    
    print(f"\n📊 Throttle 集中度 (top kernel):")
    for k, n in throttle_by_kernel.most_common(10):
        print(f"  {n:3d}  {k}")
    
    print(f"\n📊 全部 round 都被 throttle 的 mutants (前 20):")
    for mid in mutants_all_throttled[:20]:
        print(f"  - {mid}")
    if len(mutants_all_throttled) > 20:
        print(f"  ... 还有 {len(mutants_all_throttled)-20} 个")
    
    print()
    return {
        'total_mutants': len(files),
        'all_throttled': len(mutants_all_throttled),
        'partial_throttled': len(mutants_partial_throttled),
        'rounds_throttled': rounds_throttled,
        'rounds_total': rounds_total,
    }

stats = {}
for name in ['task_a_phase2_rerun', 'task_c_phase1_direct']:
    stats[name] = analyze(name)

print('='*90)
print("汇总")
print('='*90)
total_clean = 0
total_dirty = 0
for k, v in stats.items():
    clean = v['total_mutants'] - v['all_throttled'] - v['partial_throttled']
    total_clean += clean
    total_dirty += v['all_throttled'] + v['partial_throttled']
    print(f"  {k}: 干净 {clean}, 污染 {v['all_throttled']+v['partial_throttled']}")
print(f"  合计: 干净 {total_clean}, 污染 {total_dirty}")
