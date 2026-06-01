#!/usr/bin/env bash
# Wrapper: run Task A extra-3 supplement under the KernelBench-0 venv.
# Usage:
#   bash scripts/_run_taskA_extra3_dryrun.sh         # dry-run
#   bash scripts/_run_taskA_extra3_dryrun.sh --real  # full run
set -euo pipefail

ROOT="/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel"
VENV_PY="/home/kbuser/projects/KernelBench-0/.venv/bin/python"
LOG_DIR="$ROOT/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun"
mkdir -p "$LOG_DIR"

cd "$ROOT"

if [[ "${1:-}" == "--real" ]]; then
    LOG="$LOG_DIR/run_extra3.log"
    echo "[wrapper] real run -> $LOG" >&2
    "$VENV_PY" -u scripts/run_taskA_3_extra.py --rounds 5 2>&1 | tee "$LOG"
else
    LOG="$LOG_DIR/run_extra3_dryrun.log"
    echo "[wrapper] dry-run -> $LOG" >&2
    "$VENV_PY" -u scripts/run_taskA_3_extra.py --dry-run 2>&1 | tee "$LOG"
fi
