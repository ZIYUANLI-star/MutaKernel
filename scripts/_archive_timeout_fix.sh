#!/usr/bin/env bash
# Archive current Task A/C runs (correct allclose protocol, but pre-timeout-fix)
# and start fresh with new 600s Bedrock timeout config.
set -euo pipefail

ROOT=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
SUPPL="$ROOT/第二次实验汇总/第二次实验汇总_补充"
TAG=$(date +%Y%m%d_%H%M%S)_pre_timeout_fix

mkdir -p "$SUPPL/_archive"

for task in task_a_phase2_rerun task_c_phase1_direct; do
  src="$SUPPL/$task"
  if [[ -d "$src" ]] && [[ -n "$(ls -A "$src" 2>/dev/null)" ]]; then
    dst="$SUPPL/_archive/${task}__${TAG}"
    echo "Archiving $task -> _archive/${task}__${TAG}"
    mv "$src" "$dst"
    mkdir -p "$src"
  fi
done

LOGDIR=/home/kbuser/mutakernel_logs
mkdir -p "$LOGDIR/_archive"
for log in task_a.log task_c.log; do
  if [[ -s "$LOGDIR/$log" ]]; then
    mv "$LOGDIR/$log" "$LOGDIR/_archive/${log%.log}__${TAG}.log"
  fi
done

echo
echo "Now launching fresh Task A and Task C with new timeout config..."
bash "$ROOT/scripts/_launch_taskA.sh"
bash "$ROOT/scripts/_launch_taskC.sh"
