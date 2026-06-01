#!/usr/bin/env bash
# Launch Task B main loop (3-round Opus 4.5 regeneration on 18 buggy kernels).
# Run AFTER _supplement_train_refok.py finishes.
set -u
cd /home/kbuser/projects/KernelBench-0
source .venv/bin/activate
cd /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel

LOG_DIR=/home/kbuser/mutakernel_logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/taskB_main.log"
PIDF="$LOG_DIR/taskB_main.pid"

# Optional: pass through extra args (e.g. --only-kernel L1_P39 for smoke)
ARGS="--resume $@"

echo "=== Task B regenerate started at $(date) ===" > "$LOG"
echo "ARGS: $ARGS" >> "$LOG"
echo "" >> "$LOG"

# nohup + setsid for full session detachment (resists WSL idle reaping when
# combined with vmIdleTimeout=-1 / a parallel keepalive).
nohup setsid python -u scripts/run_taskB_regenerate.py $ARGS \
    </dev/null >> "$LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDF"
disown $PID 2>/dev/null || true

# Keepalive: a tiny daemon that touches the log every 30s so WSL never
# considers itself idle. It dies if the main proc dies.
KEEPALIVE_LOG=/home/kbuser/mutakernel_logs/taskB_keepalive.log
nohup setsid bash -c "
while kill -0 $PID 2>/dev/null; do
    echo \"[\$(date +%H:%M:%S)] keepalive: main PID=$PID alive\" >> $KEEPALIVE_LOG
    sleep 30
done
echo \"[\$(date +%H:%M:%S)] keepalive: main PID=$PID gone, exit\" >> $KEEPALIVE_LOG
" </dev/null >> "$KEEPALIVE_LOG" 2>&1 &
KEEPALIVE_PID=$!
disown $KEEPALIVE_PID 2>/dev/null || true

echo "Started main PID=$PID, keepalive PID=$KEEPALIVE_PID"
echo "Log:        $LOG"
echo "Keepalive:  $KEEPALIVE_LOG"
echo "PID:        $PIDF"

# Block this launch shell for 10s so WSL stays alive while the bg proc starts up
sleep 10
echo "Launch shell exiting; main proc continues in background."
