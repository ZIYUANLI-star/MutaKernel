#!/usr/bin/env bash
# 启动 Task C 补跑（仅 Task C，232 个污染）。
set -euo pipefail

PROJ=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
LOGDIR=/home/kbuser/mutakernel_logs
mkdir -p "$LOGDIR"

WORKER_SRC="$PROJ/scripts/_resume_taskC_worker.sh"
WORKER_DST="$LOGDIR/_resume_taskC_worker.sh"
cp "$WORKER_SRC" "$WORKER_DST"
chmod +x "$WORKER_DST"

PIDFILE="$LOGDIR/resume_taskC.pid"
LOGFILE="$LOGDIR/resume_taskC.log"

# 旧 log 归档
if [ -f "$LOGFILE" ]; then
  mv "$LOGFILE" "$LOGFILE.$(date +%Y%m%d_%H%M%S)"
fi

setsid bash -c "exec </dev/null >>\"$LOGFILE\" 2>&1; \"$WORKER_DST\"" &
echo $! > "$PIDFILE"
disown -a

sleep 2

WORKER_PID=$(pgrep -f "_resume_taskC_worker.sh" | head -1 || true)
if [ -n "$WORKER_PID" ]; then
  echo $WORKER_PID > "$PIDFILE"
  echo "Task C resume launched."
  echo "  Worker PID: $WORKER_PID"
  echo "  Log: $LOGFILE"
  echo "  List: $PROJ/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/rerun_throttled.txt"
else
  echo "WARNING: could not locate worker process; check log."
fi
