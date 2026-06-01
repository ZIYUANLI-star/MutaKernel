#!/usr/bin/env python3
"""导出被 ThrottlingException 完全/部分污染的 mutant 清单"""
import json
from pathlib import Path

ROOT = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充')

for name in ['task_a_phase2_rerun', 'task_c_phase1_direct']:
    sub = ROOT / name
    files = sorted(sub.glob('details/*.json'))
    polluted = []
    for f in files:
        d = json.loads(f.read_text(encoding='utf-8'))
        rounds = d.get('rounds', [])
        # 任何一个 round 含 ThrottlingException 都算污染
        has_throttle = any('ThrottlingException' in (r.get('error') or '') for r in rounds)
        if has_throttle:
            polluted.append(d['mutant_id'])
    out = sub / 'rerun_throttled.txt'
    out.write_text('\n'.join(polluted), encoding='utf-8')
    print(f"{name}: {len(polluted)} mutants -> {out}")
