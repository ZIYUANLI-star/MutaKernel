#!/usr/bin/env python3
import json, glob
ROOT = '/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充'
for sub in ['task_a_phase2_rerun', 'task_c_phase1_direct']:
    files = sorted(glob.glob(f'{ROOT}/{sub}/details/*.json'))
    killed = 0
    timeouts = 0
    errs = 0
    err_list = []
    for f in files:
        d = json.load(open(f, encoding='utf-8'))
        if d.get('killed'):
            killed += 1
        for r in d.get('rounds', []):
            e = r.get('error') or ''
            if e:
                if 'timeout' in e.lower() or 'timed out' in e.lower():
                    timeouts += 1
                    err_list.append((d['mutant_id'], r.get('round'), 'TIMEOUT: ' + e[:60]))
                else:
                    errs += 1
                    err_list.append((d['mutant_id'], r.get('round'), 'ERR: ' + e[:60]))
    print(f'[{sub}] done={len(files)}, killed={killed}, timeouts={timeouts}, other_errs={errs}')
    for it in err_list[:15]:
        print('  ', it)
