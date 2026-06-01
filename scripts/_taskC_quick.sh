#!/usr/bin/env bash
LOG=/home/kbuser/mutakernel_logs/resume_taskC.log
LAST=$(grep -oE '\[[0-9]+/[0-9]+\] L' "$LOG" | tail -1)
KR=$(grep -oE 'running kill rate=[0-9]+/[0-9]+' "$LOG" | tail -1)
CUR=$(echo "$LAST" | grep -oE '[0-9]+' | head -1)
TOTAL=$(echo "$LAST" | grep -oE '[0-9]+' | sed -n '2p')
REM=$((TOTAL - CUR))
echo "当前正在处理: $LAST"
echo "已开始/总计: $CUR / $TOTAL"
echo "剩余:        $REM"
echo "$KR"
echo ""
echo "=== 最近 5 个结果 ==="
grep "running kill rate=" "$LOG" | tail -5
