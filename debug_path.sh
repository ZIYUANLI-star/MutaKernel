#!/bin/bash
source /home/kbuser/projects/KernelBench-0/.venv/bin/activate
KB="/home/kbuser/projects/KernelBench-0"
MK="/home/kbuser/projects/MutaKernel"

echo "=== Check exact file names for problem 1 ==="
ls "$KB/runs/iter_full_l1_caesar_paper_v2/level_1_problem_1_"* 2>/dev/null || echo "No level_1_problem_1_* files"

echo ""
echo "=== List ALL kernel files (count) ==="
ls "$KB/runs/iter_full_l1_caesar_paper_v2/"level_1_problem_*.py 2>/dev/null | wc -l

echo ""
echo "=== First 5 kernel files ==="
ls "$KB/runs/iter_full_l1_caesar_paper_v2/"level_1_problem_*.py 2>/dev/null | sort -t_ -k4 -n | head -5

echo ""
echo "=== Check __pycache__ ==="
find "$MK" -name "__pycache__" -type d

echo ""
echo "=== Clean pycache ==="
find "$MK" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "Cleaned"

echo ""
echo "=== Verify bridge code has new pattern ==="
grep "level_" "$MK/src/bridge/eval_bridge.py" | head -5
