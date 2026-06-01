#!/bin/bash
LOG=/home/kbuser/mutakernel_logs/resume_taskC.log

echo "=== 进程 ==="
ps -p 367,379 -o pid,etime,pcpu,pmem,cmd 2>/dev/null

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv 2>/dev/null

echo
echo "=== 最近 15 行 stdout ==="
tail -n 15 "$LOG"

echo
echo "=== 整体进度 ==="
DONE=$(grep -cE 'running kill rate=' "$LOG")
echo "  已完成 mutant 数: $DONE / 232"
KILLED=$(grep -cE 'KILLED in round' "$LOG")
echo "  本次累计 killed:  $KILLED"
echo "  最后一行进度:"
grep -E 'running kill rate=' "$LOG" | tail -1

echo
echo "=== 错误信号扫描 ==="
ERR=$(grep -cE 'ThrottlingException|TooManyRequests|Traceback|RuntimeError|Too many tokens' "$LOG")
echo "  累计错误条数: $ERR"
if [ "$ERR" -gt 0 ]; then
    echo "  最近 5 条错误："
    grep -E 'ThrottlingException|TooManyRequests|Traceback|RuntimeError|Too many tokens' "$LOG" | tail -5
fi

echo
echo "=== 输出目录 最新 details ==="
ls -lt /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/details/ 2>/dev/null | head -6
