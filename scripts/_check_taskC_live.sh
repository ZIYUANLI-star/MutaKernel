#!/usr/bin/env bash
set -u
echo "=== Task C 进程 ==="
ps -ef | grep -E 'run_taskC|_resume_taskC|_rerun_worker' | grep -v grep | head -20

echo ""
echo "=== 日志目录 ==="
LOG_DIR=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/logs
ls -lt "$LOG_DIR" 2>/dev/null | head -10

echo ""
echo "=== Task C details 总数 ==="
DET=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/details
ls "$DET" | wc -l

echo ""
echo "=== 最新一个 detail ==="
ls -lt "$DET" | head -3

echo ""
echo "=== 最近 5 分钟有更新的 detail 数（说明还在跑） ==="
find "$DET" -name '*.json' -mmin -5 2>/dev/null | wc -l

echo ""
echo "=== 最近 30 分钟有更新的 detail 数 ==="
find "$DET" -name '*.json' -mmin -30 2>/dev/null | wc -l

echo ""
echo "=== Task C 最新日志末尾 ==="
LATEST=$(ls -t "$LOG_DIR"/taskC*.log 2>/dev/null | head -1)
echo "tail of: $LATEST"
tail -25 "$LATEST" 2>/dev/null
