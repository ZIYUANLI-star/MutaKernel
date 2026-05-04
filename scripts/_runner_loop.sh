#!/bin/bash
set -o pipefail

VENV=/home/kbuser/projects/KernelBench-0/.venv
PYTHON=$VENV/bin/python3
PROJECT=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
MAX_RESTARTS=90
RESTART_COUNT=0

cd "$PROJECT"

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo ""
    echo "========================================"
    echo "  Run #$((RESTART_COUNT+1)) — $(date)"
    echo "========================================"

    find /home/kbuser/.cache/torch_extensions -name "lock" -delete 2>/dev/null || true

    $PYTHON scripts/full_block12.py 2>&1 | tee -a full_block12_console.log
    EXIT_CODE=${PIPESTATUS[0]}

    echo "[RUNNER] Python exited with code $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "[RUNNER] Experiment completed successfully (exit 0)."
        break
    elif [ $EXIT_CODE -eq 77 ]; then
        RESTART_COUNT=$((RESTART_COUNT+1))
        echo ""
        echo "[RUNNER] CUDA crash (exit 77). Auto-restarting in 5s... (restart #$RESTART_COUNT)"
        sleep 5
    elif [ $EXIT_CODE -eq 78 ]; then
        RESTART_COUNT=$((RESTART_COUNT+1))
        echo ""
        echo "[RUNNER] Watchdog timeout (exit 78). Auto-restarting in 5s... (restart #$RESTART_COUNT)"
        sleep 5
    else
        RESTART_COUNT=$((RESTART_COUNT+1))
        echo ""
        echo "[RUNNER] Unexpected exit ($EXIT_CODE). Auto-restarting in 10s... (restart #$RESTART_COUNT)"
        sleep 10
    fi
done

if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
    echo "[RUNNER] Max restarts reached ($MAX_RESTARTS). Stopping."
fi
echo "[RUNNER] Done."
