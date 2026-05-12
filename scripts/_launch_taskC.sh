#!/usr/bin/env bash
# Launch Task C truly detached so it survives wsl.exe session exit.
set -euo pipefail

PROJ=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
VENV=/home/kbuser/projects/KernelBench-0/.venv
LOGDIR=/home/kbuser/mutakernel_logs
mkdir -p "$LOGDIR"
PIDFILE="$LOGDIR/task_c.pid"

cd "$PROJ"
setsid bash -c "
  exec </dev/null
  exec >>\"$LOGDIR/task_c.log\" 2>&1
  \"$VENV/bin/python\" -u scripts/run_taskC_phase1_direct.py --rounds 5
" &
echo $! > "$PIDFILE"
disown -a
echo "Task C launched. PID=$(cat $PIDFILE)"
echo "Log: $LOGDIR/task_c.log"
echo "Pidfile: $PIDFILE"
