#!/bin/bash
set -e
cd /root/autodl-tmp
rm -rf mk_stress && mkdir mk_stress && cd mk_stress
unzip -q ../mk_stress.zip
PY=/root/miniconda3/bin/python
echo "=== worker smoke: 5 survived mutants, 1 worker ==="
$PY scripts/kgb_stress_orchestrator.py --limit 5 --status survived --workers 1 --timeout 150
echo "=== results ==="
cat stress/results.jsonl
