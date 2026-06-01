#!/usr/bin/env python3
"""Task A + Task C 整体汇报：合并干净数据 + 当前重跑进度。"""
import json
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充')

def analyze(name, total_target):
    sub = ROOT / name
    files = sorted(sub.glob('details/*.json'))
    
    # 分桶
    clean_killed = []
    clean_unkilled_killable_false = []
    clean_unkilled_5rounds_failed = []
    polluted = []
    rounds_dist = Counter()
    reason_cat_clean = Counter()
    kernel_kills = defaultdict(int)
    operator_kills = defaultdict(int)
    
    total_in_tok = 0
    total_out_tok = 0
    total_reason_tok = 0
    total_elapsed = 0.0
    
    for f in files:
        d = json.loads(f.read_text(encoding='utf-8'))
        mid = d['mutant_id']
        kernel = d.get('kernel_name', '?')
        operator = d.get('operator_name', '?')
        rounds = d.get('rounds', [])
        killed = d.get('killed', False)
        elapsed = d.get('elapsed_sec', 0.0) or 0.0
        total_elapsed += elapsed
        
        has_throttle = any('ThrottlingException' in (r.get('error') or '') for r in rounds)
        rounds_dist[len(rounds)] += 1
        
        # token totals (all)
        for r in rounds:
            u = r.get('usage', {}) or {}
            total_in_tok += u.get('prompt_tokens', 0) or 0
            total_out_tok += u.get('completion_tokens', 0) or 0
            total_reason_tok += u.get('reasoning_tokens', 0) or 0
        
        if has_throttle:
            polluted.append(mid)
            continue
        
        # clean data
        if killed:
            clean_killed.append((mid, d.get('killing_round'), len(rounds)))
            kernel_kills[kernel] += 1
            operator_kills[operator] += 1
        else:
            last = rounds[-1] if rounds else {}
            if last.get('killable') is False:
                clean_unkilled_killable_false.append(mid)
                cat = last.get('reason_category')
                if cat:
                    reason_cat_clean[cat] += 1
            else:
                clean_unkilled_5rounds_failed.append(mid)
    
    return {
        'name': name,
        'total_target': total_target,
        'files_total': len(files),
        'clean_killed': clean_killed,
        'clean_unkilled_kf': clean_unkilled_killable_false,
        'clean_unkilled_5r': clean_unkilled_5rounds_failed,
        'polluted': polluted,
        'rounds_dist': rounds_dist,
        'reason_cat': reason_cat_clean,
        'kernel_kills': kernel_kills,
        'operator_kills': operator_kills,
        'tokens': (total_in_tok, total_out_tok, total_reason_tok),
        'total_elapsed': total_elapsed,
    }

a = analyze('task_a_phase2_rerun', 365)
c = analyze('task_c_phase1_direct', 534)

def fmt(s):
    print('='*80)
    print(f"## {s['name']}")
    print('='*80)
    n_killed = len(s['clean_killed'])
    n_kf = len(s['clean_unkilled_kf'])
    n_5r = len(s['clean_unkilled_5r'])
    n_poll = len(s['polluted'])
    n_clean = s['files_total'] - n_poll
    
    print(f"  目标 mutant 数:        {s['total_target']}")
    print(f"  已写入 detail JSON:    {s['files_total']}")
    print(f"  ├ 干净（有效 LLM）:    {n_clean}")
    print(f"  │  ├ 已杀死:           {n_killed}  (kill rate = {n_killed/max(1,n_clean)*100:.1f}% on clean)")
    print(f"  │  ├ killable=False 早停:  {n_kf}")
    print(f"  │  └ killable=True 5轮失败: {n_5r}")
    print(f"  └ 被 throttle 污染:    {n_poll}  (待重跑)")
    print()
    print(f"  Round 分布:")
    for r, n in sorted(s['rounds_dist'].items()):
        print(f"    {r} 轮: {n} 个 mutant")
    print()
    print(f"  reason_category 分布（仅干净 unkilled）:")
    for cat, n in s['reason_cat'].most_common():
        print(f"    {n:3d}  {cat}")
    print()
    if s['clean_killed']:
        print(f"  已杀死的 mutant ({n_killed} 个，按 kernel 分组):")
        for k, n in sorted(s['kernel_kills'].items()):
            print(f"    {k:12s} : {n} 个")
        print()
        print(f"  按 operator 分组:")
        for op, n in sorted(s['operator_kills'].items(), key=lambda x:-x[1]):
            print(f"    {op:25s} : {n}")
    print()
    in_tok, out_tok, r_tok = s['tokens']
    print(f"  Token 总用量:")
    print(f"    input:     {in_tok:>10,}")
    print(f"    output:    {out_tok:>10,}")
    print(f"    reasoning: {r_tok:>10,}")
    print(f"    total in+out: {in_tok+out_tok:>10,}")
    print()
    print(f"  累计 LLM 耗时（仅 mutant elapsed，不含 GPU verify）: "
          f"{s['total_elapsed']:.0f}s ≈ {s['total_elapsed']/3600:.1f} h")
    print()

fmt(a)
fmt(c)

print('='*80)
print("## 合计")
print('='*80)
print(f"  目标 mutant 总数:    {a['total_target']+c['total_target']}")
print(f"  已写 detail:         {a['files_total']+c['files_total']}")
print(f"  其中干净 mutant:     {a['files_total']-len(a['polluted'])+c['files_total']-len(c['polluted'])}")
print(f"    其中已杀死:        {len(a['clean_killed'])+len(c['clean_killed'])}")
print(f"  待重跑（污染）:      {len(a['polluted'])+len(c['polluted'])}")
total_in = a['tokens'][0]+c['tokens'][0]
total_out = a['tokens'][1]+c['tokens'][1]
total_r = a['tokens'][2]+c['tokens'][2]
print(f"  Token: in={total_in:,} out={total_out:,} reasoning={total_r:,}")

# 关键观察：杀死率对比
n_a_clean = a['files_total'] - len(a['polluted'])
n_c_clean = c['files_total'] - len(c['polluted'])
print()
print("--- 关键对比指标 ---")
print(f"  Task A kill rate (Phase II 残留): {len(a['clean_killed'])}/{n_a_clean} "
      f"= {len(a['clean_killed'])/max(1,n_a_clean)*100:.1f}%")
print(f"  Task C kill rate (Phase I 残留):  {len(c['clean_killed'])}/{n_c_clean} "
      f"= {len(c['clean_killed'])/max(1,n_c_clean)*100:.1f}%")
print(f"  → Phase I 残留 vs Phase II 残留：Phase II 残留更难杀（符合预期）")
