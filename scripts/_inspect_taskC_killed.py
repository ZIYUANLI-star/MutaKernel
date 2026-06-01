#!/usr/bin/env python3
"""看 Task C 那个被杀死的 mutant 的细节"""
import json

f = '/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/details/L1_P100__mask_boundary__0.json'
d = json.load(open(f, encoding='utf-8'))

print("=" * 80)
print(f"mutant_id   = {d['mutant_id']}")
print(f"operator    = {d.get('operator_name')}")
print(f"kernel      = {d.get('kernel_name')}")
print(f"killed      = {d.get('killed')}")
print(f"killing_round = {d.get('killing_round')}")
print(f"elapsed_sec = {d.get('elapsed_sec'):.1f}s")
print(f"phase1_baseline = {d.get('phase1_baseline')}")
print()

for r in d.get('rounds', []):
    print(f"--- Round {r.get('round')} ---")
    print(f"  killable         = {r.get('killable')}")
    print(f"  reason_category  = {r.get('reason_category')}")
    print(f"  killed           = {r.get('killed')}")
    er = r.get('execution_result')
    if er:
        print(f"  execution_result keys: {list(er.keys()) if isinstance(er, dict) else er}")
        if isinstance(er, dict):
            print(f"    killed   = {er.get('killed')}")
            print(f"    max_abs_diff = {er.get('max_abs_diff')}")
            print(f"    max_rel_diff = {er.get('max_rel_diff')}")
            print(f"    diff_summary = {er.get('diff_summary')}")
    sc = r.get('suggested_code') or ''
    print(f"  suggested_code (前 300 char):")
    print('    ' + sc[:300].replace('\n', '\n    '))
    print()
