#!/usr/bin/env bash
# 串行重跑 worker：被 _launch_rerun_throttled.sh 启动。
# 不能直接由用户调用（会阻塞 wsl.exe）。
set -uo pipefail

PROJ=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
VENV=/home/kbuser/projects/KernelBench-0/.venv
LIST_A="$PROJ/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/rerun_throttled.txt"
LIST_C="$PROJ/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/rerun_throttled.txt"

cd "$PROJ"

echo
echo "=============================================================="
echo "[$(date -Iseconds)] STARTING SERIAL RERUN (new API key)"
echo "=============================================================="
echo

echo "[1/2] Task A rerun ($(wc -l < "$LIST_A") throttled mutants) starting..."
"$VENV/bin/python" -u scripts/run_taskA_phase2_rerun.py \
    --only-mutants "$LIST_A" --rounds 5
RC_A=$?
echo
echo "[$(date -Iseconds)] Task A rerun finished with rc=$RC_A"
echo

echo "[2/2] Task C rerun ($(wc -l < "$LIST_C") throttled mutants) starting..."
"$VENV/bin/python" -u scripts/run_taskC_phase1_direct.py \
    --only-mutants "$LIST_C" --rounds 5
RC_C=$?
echo
echo "[$(date -Iseconds)] Task C rerun finished with rc=$RC_C"
echo
echo "=============================================================="
echo "SERIAL RERUN ALL DONE"
echo "=============================================================="
