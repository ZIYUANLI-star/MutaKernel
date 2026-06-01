#!/usr/bin/env python3
"""Task A 中跑了 2/3/4 轮的 mutant：他们到底为啥停？"""
import json
from pathlib import Path

ROOT = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun')

multi_round = []
for f in sorted(ROOT.glob('details/*.json')):
    d = json.loads(f.read_text(encoding='utf-8'))
    rounds = d.get('rounds', [])
    has_throttle = any('ThrottlingException' in (r.get('error') or '') for r in rounds)
    if has_throttle:
        continue
    if len(rounds) >= 2:
        multi_round.append((d['mutant_id'], rounds))

print(f"Task A 中跑了 >=2 轮的 clean mutant: {len(multi_round)}")
print()

# 分类：最后一轮的状态
final_states = {}
for mid, rounds in multi_round:
    last = rounds[-1]
    killable = last.get('killable')
    has_error = bool(last.get('error'))
    sc = (last.get('suggested_code') or '').strip()
    er = last.get('execution_result')
    er_killed = er.get('killed') if isinstance(er, dict) else None
    er_error = er.get('error') if isinstance(er, dict) else None
    
    if has_error:
        key = f"ERROR ({last.get('error')[:50]!r})"
    elif killable is False:
        key = "killable=False 早停"
    elif killable is True and not sc:
        key = "killable=True 但 LLM 没给代码（continue 进下一轮但 break了？）"
    elif er and not er_killed and er_error:
        key = f"exec error: {(er_error or '')[:60]}"
    elif er and not er_killed:
        key = "exec ran, not killed (within atol/rtol)"
    else:
        key = "other"
    final_states.setdefault(key, []).append((mid, len(rounds)))

print("最后一轮停止原因分布：")
for k, lst in sorted(final_states.items(), key=lambda x:-len(x[1])):
    print(f"  [{len(lst)} 个] {k}")
    for mid, n in lst[:3]:
        print(f"    - {mid} ({n} 轮)")
    if len(lst) > 3:
        print(f"    ... 还有 {len(lst)-3} 个")
print()

# 详细查 3 个有代表性的样本
print("="*80)
print("详细样本（3 个）")
print("="*80)
shown = 0
for mid, rounds in multi_round:
    if shown >= 3:
        break
    print(f"\n--- {mid} ---  ({len(rounds)} 轮)")
    for r in rounds:
        rnum = r.get('round')
        killable = r.get('killable')
        cat = r.get('reason_category')
        err = r.get('error') or ''
        sc_len = len((r.get('suggested_code') or '').strip())
        er = r.get('execution_result')
        er_str = ''
        if isinstance(er, dict):
            er_str = f"exec: killed={er.get('killed')} ref_ok={er.get('ref_ok')} orig_ok={er.get('original_ok')} mut_ok={er.get('mutant_ok')} diff={(er.get('diff_summary') or '')[:60]}"
        elif er is None:
            er_str = "exec: None"
        print(f"  R{rnum}: killable={killable} cat={cat}  err={err[:40]!r}  code_len={sc_len}  {er_str}")
    shown += 1
