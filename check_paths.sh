#!/bin/bash
source /home/kbuser/projects/KernelBench-0/.venv/bin/activate
KB="/home/kbuser/projects/KernelBench-0"

echo "=== KernelBench tree (depth 3) ==="
find "$KB/KernelBench" -maxdepth 3 -type d 2>/dev/null | head -20

echo ""
echo "=== Problem files ==="
find "$KB" -maxdepth 5 -name "1_*.py" -type f 2>/dev/null | head -10

echo ""
echo "=== Runs L1 structure ==="
ls "$KB/runs/iter_full_l1_caesar_paper_v2/" | head -10

echo ""
echo "=== Runs L1 subdirs ==="
ls "$KB/runs/iter_full_l1_caesar_paper_v2/" -d */ 2>/dev/null | head -10
find "$KB/runs/iter_full_l1_caesar_paper_v2" -maxdepth 1 -type d | head -10

echo ""
echo "=== Find generated py files ==="
find "$KB/runs/iter_full_l1_caesar_paper_v2" -maxdepth 3 -name "*.py" -type f 2>/dev/null | head -10

echo ""
echo "=== Find any generated_ files ==="
find "$KB/runs/iter_full_l1_caesar_paper_v2" -maxdepth 3 -name "generated*" 2>/dev/null | head -10

echo ""
echo "=== Eval results keys sample ==="
python -c "
import json
with open('$KB/runs/iter_full_l1_caesar_paper_v2/eval_results.json') as f:
    data = json.load(f)
for k, v in list(data.items())[:3]:
    print(f'key={k!r}, correctness={v.get(\"correctness\")}, compiled={v.get(\"compiled\")}')
    if 'metadata' in v:
        print(f'  metadata keys: {list(v[\"metadata\"].keys())[:5]}')
"
