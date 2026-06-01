#!/usr/bin/env bash
set -u

echo "=== [1] Task C 进程树 + 运行时长 ==="
ps -ef | grep -E 'run_taskC|_resume_taskC|_rerun_worker' | grep -v grep

echo ""
echo "=== [2] 进程当前在做什么 (proc状态/读写) ==="
for PID in $(pgrep -f 'run_taskC_phase1_direct'); do
    echo "--- PID $PID ---"
    cat /proc/$PID/status 2>/dev/null | grep -E '^(State|Name|Threads|VmRSS):'
    echo "  cmdline: $(tr '\0' ' ' </proc/$PID/cmdline 2>/dev/null)"
    echo "  cwd:     $(readlink /proc/$PID/cwd 2>/dev/null)"
    echo "  open files (top 20):"
    ls -l /proc/$PID/fd 2>/dev/null | awk 'NR>1 {print "    "$NF}' | head -20
    echo "  stack (网络/系统调用):"
    cat /proc/$PID/wchan 2>/dev/null; echo ""
done

echo ""
echo "=== [3] 找到 Task C stdout 日志文件 ==="
# 看哪个文件描述符是日志输出
for PID in $(pgrep -f 'run_taskC_phase1_direct'); do
    echo "--- PID $PID fd 1/2 ---"
    ls -l /proc/$PID/fd/1 /proc/$PID/fd/2 2>/dev/null
done

echo ""
echo "=== [4] 跟踪一下当前系统调用 (5 秒抓样) ==="
PID=$(pgrep -f 'run_taskC_phase1_direct' | head -1)
if [ -n "$PID" ]; then
    timeout 5 strace -p "$PID" -e trace=network,read,write,poll,epoll_wait -s 80 -c 2>&1 | tail -40
fi

echo ""
echo "=== [5] 最近 60 分钟内 detail 写入时间分布 ==="
DET=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/details
find "$DET" -name '*.json' -mmin -60 2>/dev/null | xargs -I{} stat -c '%y  %n' {} 2>/dev/null | sort | tail -25

echo ""
echo "=== [6] Task C 输出日志 ==="
LOG_DIR=/home/kbuser/mutakernel_logs
ls -lt "$LOG_DIR" 2>/dev/null | head -10
LATEST=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
echo "tail -80 of: $LATEST"
tail -80 "$LATEST" 2>/dev/null
