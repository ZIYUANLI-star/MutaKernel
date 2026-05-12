#!/usr/bin/env bash
# Sanity checks for ongoing Task A/C runs.
ROOT=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
LOGDIR=/home/kbuser/mutakernel_logs

echo "============================================="
echo "  1. Processes alive?"
echo "============================================="
for name in task_a task_c; do
  pid=$(cat "$LOGDIR/$name.pid" 2>/dev/null)
  if [[ -n "$pid" ]] && ps -p "$pid" > /dev/null 2>&1; then
    echo "  $name  pid=$pid  RUNNING  $(ps -p $pid -o etime= | tr -d ' ')"
  else
    echo "  $name  NOT RUNNING (pid=$pid)"
  fi
done
echo

echo "============================================="
echo "  2. GPU health"
echo "============================================="
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader 2>&1 | head -1
echo

echo "============================================="
echo "  3. Disk space (D drive via /mnt/d)"
echo "============================================="
df -h /mnt/d 2>&1 | tail -1
echo

echo "============================================="
echo "  4. Token usage so far (Task A + C combined)"
echo "============================================="
python3 - <<'PY'
import json
from pathlib import Path
ROOT = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充')
total_in = total_out = total_reasoning = total_rounds = 0
for sub in ['task_a_phase2_rerun', 'task_c_phase1_direct']:
    for f in (ROOT / sub / 'details').glob('*.json'):
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
        except Exception:
            continue
        for r in d.get('rounds', []) or []:
            u = r.get('usage') or {}
            total_in += u.get('prompt_tokens', 0)
            total_out += u.get('completion_tokens', 0)
            total_reasoning += u.get('reasoning_tokens', 0)
            total_rounds += 1
print(f"  Rounds executed   : {total_rounds}")
print(f"  Input tokens      : {total_in:,}")
print(f"  Output tokens     : {total_out:,}")
print(f"  Reasoning tokens  : {total_reasoning:,}  (subset of output)")
# Claude Opus 4.5 on Bedrock: input $15/M, output $75/M
cost_in  = total_in  * 15 / 1_000_000
cost_out = total_out * 75 / 1_000_000
print(f"  Estimated cost    : ${cost_in + cost_out:.4f}  (in ${cost_in:.4f} + out ${cost_out:.4f})")
PY
echo

echo "============================================="
echo "  5. Recent errors / warnings in logs"
echo "============================================="
for name in task_a task_c; do
  log="$LOGDIR/$name.log"
  if [[ -f "$log" ]]; then
    errs=$(grep -iE 'error|exception|traceback|fail|warning' "$log" \
           | grep -v 'pass\|fail count\|all_dimensions_survived' | tail -5)
    if [[ -z "$errs" ]]; then
      echo "  $name: (no errors)"
    else
      echo "  $name: latest 5 error/warning lines"
      echo "$errs" | sed 's/^/    /'
    fi
  fi
done
echo

echo "============================================="
echo "  6. Output file growth rate"
echo "============================================="
for sub in task_a_phase2_rerun task_c_phase1_direct; do
  dir="$ROOT/第二次实验汇总/第二次实验汇总_补充/$sub/details"
  if [[ -d "$dir" ]]; then
    n=$(ls "$dir" 2>/dev/null | wc -l)
    sz=$(du -sh "$dir" 2>/dev/null | awk '{print $1}')
    echo "  $sub: $n mutant details  ($sz)"
  fi
done
echo

echo "============================================="
echo "  7. Bedrock latency (last 5 rounds)"
echo "============================================="
python3 - <<'PY'
import json
from pathlib import Path
ROOT = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充')
all_lat = []
for sub in ['task_a_phase2_rerun', 'task_c_phase1_direct']:
    rdir = ROOT / sub / 'llm_responses'
    if not rdir.exists(): continue
    for f in sorted(rdir.glob('*.json'), key=lambda p: p.stat().st_mtime)[-3:]:
        try:
            d = json.loads(f.read_text(encoding='utf-8'))
        except Exception:
            continue
        lat = d.get('latency_ms')
        if lat is not None:
            all_lat.append((f.stat().st_mtime, sub, f.stem, lat))
all_lat.sort()
for _, sub, name, lat in all_lat[-5:]:
    print(f"  {sub[:6]:6s}  {name:50s}  {lat/1000:.1f}s")
PY
