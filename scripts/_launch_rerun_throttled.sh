#!/usr/bin/env bash
# 启动串行重跑：先 Task A 43 个，再 Task C 306 个。
# 全部用新 Bedrock API key + 加强后的限流退避。
set -euo pipefail

PROJ=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
LOGDIR=/home/kbuser/mutakernel_logs
mkdir -p "$LOGDIR"

# 把 worker 脚本拷贝到 WSL 本地（避免 /mnt/d 上行权限问题）
WORKER_SRC="$PROJ/scripts/_rerun_worker.sh"
WORKER_DST="$LOGDIR/_rerun_worker.sh"
cp "$WORKER_SRC" "$WORKER_DST"
chmod +x "$WORKER_DST"

PIDFILE="$LOGDIR/rerun_throttled.pid"
LOGFILE="$LOGDIR/rerun_throttled.log"

# 清掉旧 log（保留 archive 用）
if [ -f "$LOGFILE" ]; then
  mv "$LOGFILE" "$LOGFILE.$(date +%Y%m%d_%H%M%S)"
fi

setsid bash -c "exec </dev/null >>\"$LOGFILE\" 2>&1; \"$WORKER_DST\"" &
LAUNCH_PID=$!
echo $LAUNCH_PID > "$PIDFILE"
disown -a

sleep 2

# 找真正的 worker pid（setsid 的孙进程）
WORKER_PID=$(pgrep -f "_rerun_worker.sh" | head -1 || true)
if [ -n "$WORKER_PID" ]; then
  echo $WORKER_PID > "$PIDFILE"
  echo "Serial rerun launched."
  echo "  Worker PID: $WORKER_PID"
  echo "  Log: $LOGFILE"
else
  echo "WARNING: could not locate worker process; check log."
fi
