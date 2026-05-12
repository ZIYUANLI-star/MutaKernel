#!/usr/bin/env bash
SUPPL=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充

echo "=== Timeout / error cases across both tasks ==="
python3 - <<'PY'
import json
from pathlib import Path
ROOT = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充')

for sub in ['task_a_phase2_rerun', 'task_c_phase1_direct']:
    print(f"\n--- {sub} ---")
    timeouts = []
    other_errs = []
    no_category = []
    for f in (ROOT / sub / 'details').glob('*.json'):
        d = json.loads(f.read_text(encoding='utf-8'))
        for r in d.get('rounds', []):
            err = r.get('error') or ''
            if 'timeout' in err.lower() or 'timed out' in err.lower():
                timeouts.append((d['mutant_id'], r.get('round'), err[:80]))
            elif err:
                other_errs.append((d['mutant_id'], r.get('round'), err[:80]))
        first_round = (d.get('rounds') or [{}])[0]
        if d.get('rounds') and not first_round.get('reason_category') and not first_round.get('error'):
            no_category.append(d['mutant_id'])

    print(f"  Bedrock timeouts: {len(timeouts)}")
    for mid, r, e in timeouts:
        print(f"    {mid} round {r}: {e}")
    print(f"  Other errors: {len(other_errs)}")
    for mid, r, e in other_errs:
        print(f"    {mid} round {r}: {e}")
    print(f"  No reason_category (suspicious): {len(no_category)}")
    for mid in no_category:
        print(f"    {mid}")
PY
