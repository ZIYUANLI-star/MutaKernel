#!/usr/bin/env bash
# Quick status report for Task A and Task C background jobs.
LOGDIR=/home/kbuser/mutakernel_logs

show_task() {
  local name=$1 total=$2 pidfile=$3 logfile=$4
  echo "========== $name (target=$total) =========="
  if [[ -f "$pidfile" ]]; then
    local pid
    pid=$(cat "$pidfile")
    if ps -p "$pid" > /dev/null 2>&1; then
      echo "STATUS: RUNNING (pid=$pid)"
      ps -p "$pid" -o etime= 2>/dev/null | awk '{print "ELAPSED: " $1}'
    else
      echo "STATUS: NOT_RUNNING (pid=$pid)"
    fi
  else
    echo "STATUS: NOT_LAUNCHED"
  fi
  if [[ -f "$logfile" ]]; then
    local done killed
    done=$(grep -c 'rate=' "$logfile" 2>/dev/null || echo 0)
    killed=$(grep -oE 'rate=[0-9]+/[0-9]+' "$logfile" | tail -1)
    echo "DONE: $done / $total    (latest: $killed)"
    echo "--- tail ---"
    tail -5 "$logfile"
  fi
  echo
}

show_task "Task A (Phase II rerun)" 365 "$LOGDIR/task_a.pid" "$LOGDIR/task_a.log"
show_task "Task C (Phase I direct)" 534 "$LOGDIR/task_c.pid" "$LOGDIR/task_c.log"
