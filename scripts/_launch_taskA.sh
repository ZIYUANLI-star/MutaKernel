#!/usr/bin/env bash
# Launch Task A truly detached so it survives wsl.exe session exit.
# Uses setsid to start a new session + new process group, fully detached from tty.
set -euo pipefail

PROJ=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
VENV=/home/kbuser/projects/KernelBench-0/.venv
LOGDIR=/home/kbuser/mutakernel_logs
mkdir -p "$LOGDIR"
PIDFILE="$LOGDIR/task_a.pid"

cd "$PROJ"
setsid bash -c "
  exec </dev/null
  exec >>\"$LOGDIR/task_a.log\" 2>&1
  \"$VENV/bin/python\" -u scripts/run_taskA_phase2_rerun.py --rounds 5
" &
echo $! > "$PIDFILE"
disown -a
echo "Task A launched. PID=$(cat $PIDFILE)"
echo "Log: $LOGDIR/task_a.log"
echo "Pidfile: $PIDFILE"
