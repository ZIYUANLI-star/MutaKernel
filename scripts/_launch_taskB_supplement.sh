#!/usr/bin/env bash
# Run train mode ref_ok supplement in background.
set -u
cd /home/kbuser/projects/KernelBench-0
source .venv/bin/activate
cd /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel

LOG_DIR=/home/kbuser/mutakernel_logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/taskB_supplement.log"
PIDF="$LOG_DIR/taskB_supplement.pid"

echo "=== Task B supplement train ref_ok started at $(date) ===" > "$LOG"

# nohup + setsid for true detachment
setsid python -u scripts/_supplement_train_refok.py --resume \
    >> "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PIDF"
disown $PID 2>/dev/null || true

echo "Started PID=$PID"
echo "Log:  $LOG"
echo "PID:  $PIDF"
