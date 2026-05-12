#!/usr/bin/env bash
# Kill Task A and Task C background processes.
LOGDIR=/home/kbuser/mutakernel_logs

for name in task_a task_c; do
  pidfile="$LOGDIR/$name.pid"
  if [[ -f "$pidfile" ]]; then
    pid=$(cat "$pidfile")
    if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
      echo "Killing $name pid=$pid"
      kill -TERM "$pid" 2>/dev/null
      sleep 2
      if ps -p "$pid" > /dev/null 2>&1; then
        kill -KILL "$pid" 2>/dev/null
      fi
    fi
    rm -f "$pidfile"
  fi
done

sleep 1
remaining=$(ps -ef | grep run_task | grep -v grep)
if [[ -z "$remaining" ]]; then
  echo "All Task A/C processes stopped."
else
  echo "WARNING: still running:"
  echo "$remaining"
fi
