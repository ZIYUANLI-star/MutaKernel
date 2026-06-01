#!/usr/bin/env bash
set -u
LOG=/home/kbuser/mutakernel_logs/resume_taskC.log
DET=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/details
RERUN=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/rerun_throttled.txt

echo "=== Rerun 总目标数 ==="
wc -l "$RERUN"

echo ""
echo "=== Resume rerun 进程信息 ==="
ps -ef | grep -E 'run_taskC|_resume_taskC' | grep -v grep
P=$(pgrep -f 'run_taskC_phase1_direct' | head -1)
if [ -n "$P" ]; then
    echo "  PID $P state: $(grep '^State' /proc/$P/status)"
    echo "  start_time: $(ps -o lstart= -p $P)"
    echo "  cpu_time: $(ps -o etime= -p $P) elapsed"
fi

echo ""
echo "=== 日志中当前进度（最近一行 [X/232]） ==="
grep -oP '\[\d+/\d+\] L\d_P\d+__\w+__\d+' "$LOG" | tail -1
echo ""
echo "=== 最近 5 个 mutant 结果（killed / not killed） ==="
grep -E "running kill rate=" "$LOG" | tail -10

echo ""
echo "=== Rerun 全程 throttle/error 检查 ==="
echo "ThrottlingException: $(grep -c 'ThrottlingException' $LOG)"
echo "API error:          $(grep -c 'LLM API error' $LOG)"
echo "Exception:          $(grep -cE '^Traceback|Exception:' $LOG)"

echo ""
echo "=== Rerun 累计 kill / 完成数 ==="
COMPLETED=$(grep -c "running kill rate=" "$LOG")
LAST_KR=$(grep -oP 'running kill rate=\d+/\d+' "$LOG" | tail -1)
echo "  completed:        $COMPLETED"
echo "  last kill rate:   $LAST_KR"

echo ""
echo "=== Manifest 当前状态 ==="
MAN=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/run_manifest.json
python3 -c "
import json
m = json.load(open('$MAN'))
for k in ('started_at','finished_at','completed_count','killed_count','input_count','total_tokens'):
    print(f'  {k:18s}: {m.get(k)}')
"

echo ""
echo "=== 最近 30 分钟 detail 落盘速率 ==="
find "$DET" -name '*.json' -mmin -30 2>/dev/null | wc -l
echo "  → 30 分钟内落盘 = $(find "$DET" -name '*.json' -mmin -30 2>/dev/null | wc -l) 个"
echo "  → 平均速率约 $(awk -v n=$(find "$DET" -name '*.json' -mmin -30 2>/dev/null | wc -l) 'BEGIN{printf "%.1f", n/30}') 个/分钟"

echo ""
echo "=== 进度估算 ==="
TOTAL=232
DONE=$COMPLETED
REMAIN=$((TOTAL - DONE))
echo "  目标:    $TOTAL"
echo "  已完成:  $DONE"
echo "  剩余:    $REMAIN"
RATE_PER_MIN=$(awk -v n=$(find "$DET" -name '*.json' -mmin -30 2>/dev/null | wc -l) 'BEGIN{printf "%.4f", n/30}')
if [ "$REMAIN" -gt 0 ]; then
    ETA_MIN=$(awk -v r=$REMAIN -v rate=$RATE_PER_MIN 'BEGIN{printf "%.0f", r/rate}')
    echo "  按当前速率，剩余 $ETA_MIN 分钟（≈ $((ETA_MIN/60))h$((ETA_MIN%60))m）"
fi

echo ""
echo "=== 当前正在处理的 mutant（日志末尾 20 行） ==="
tail -20 "$LOG"
