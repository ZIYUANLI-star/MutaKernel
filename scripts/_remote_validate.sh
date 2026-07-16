#!/bin/bash
set -e
cd /root/autodl-tmp
rm -rf mk_stress && mkdir mk_stress && cd mk_stress
unzip -q ../mk_stress.zip
PY=/root/miniconda3/bin/python
echo "=== validation: 30 mutants (mixed status), 4 workers ==="
$PY scripts/kgb_stress_orchestrator.py --limit 30 --workers 4 --timeout 150
echo "=== kill-mode tally ==="
$PY - <<'EOF'
import json,collections
rows=[json.loads(l) for l in open("stress/results.jsonl",encoding="utf-8") if l.strip()]
print("n",len(rows),"killed",sum(r['any_killed'] for r in rows))
print("first_kill_mode",dict(collections.Counter(r.get('first_kill_mode') for r in rows)))
print("by_status",dict(collections.Counter((r['final_emd_status'],r['any_killed']) for r in rows)))
EOF
