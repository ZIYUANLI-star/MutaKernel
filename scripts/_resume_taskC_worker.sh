#!/usr/bin/env bash
# Task C 补跑 worker：仅跑剩余的 232 个污染 mutant。
# 每个 mutant 间 sleep 8s，主动节流以避免再次触发日 token 配额。
set -uo pipefail

PROJ=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
VENV=/home/kbuser/projects/KernelBench-0/.venv
LIST_C="$PROJ/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/rerun_throttled.txt"

cd "$PROJ"

echo
echo "=============================================================="
echo "[$(date -Iseconds)] TASK C RESUME (after daily quota reset)"
echo "=============================================================="
echo "  Targets: $(wc -l < "$LIST_C") polluted mutants"
echo "  Inter-mutant sleep: 8s (token-budget aware throttle)"
echo

"$VENV/bin/python" -u scripts/run_taskC_phase1_direct.py \
    --only-mutants "$LIST_C" --rounds 5 \
    --sleep-between-mutants 8
RC=$?
echo
echo "[$(date -Iseconds)] Task C resume finished with rc=$RC"
echo "=============================================================="
