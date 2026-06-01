#!/usr/bin/env bash
set -u
echo "=== [1] 进程检查 ==="
echo "-- _supplement_train_refok.py --"
pgrep -af _supplement_train_refok | head -5 || echo "  (none)"
echo "-- run_taskB_regenerate.py --"
pgrep -af run_taskB_regenerate | head -5 || echo "  (none)"

echo ""
echo "=== [2] 补跑状态 ==="
SUPP_LOG=/home/kbuser/mutakernel_logs/taskB_supplement.log
echo "log: $SUPP_LOG (size=$(stat -c %s $SUPP_LOG 2>/dev/null) bytes)"
echo "last 15 lines:"
tail -15 "$SUPP_LOG" 2>/dev/null

echo ""
echo "=== [3] 补跑产物 ==="
SUPP_FILE=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_b_regenerate/_train_refok_supplement.json
if [ -f "$SUPP_FILE" ]; then
    echo "存在: $SUPP_FILE"
    python3 -c "
import json
d = json.load(open('$SUPP_FILE'))
s = d.get('stats', {})
sup = d.get('supplemented', {})
print(f'  kernels 已写入: {len(sup)}')
print(f'  stats: {s}')
for k, evs in sup.items():
    n_buggy = sum(1 for e in evs if e.get('ref_ok') and not e.get('original_ok'))
    n_total = len(evs)
    print(f'    {k}: {n_buggy}/{n_total} buggy')
"
else
    echo "(不存在)"
fi

echo ""
echo "=== [4] Task B 主流程状态 ==="
MAIN_LOG=/home/kbuser/mutakernel_logs/taskB_main.log
if [ -f "$MAIN_LOG" ]; then
    echo "log: $MAIN_LOG (size=$(stat -c %s $MAIN_LOG 2>/dev/null) bytes)"
    echo "last 30 lines:"
    tail -30 "$MAIN_LOG"
else
    echo "(主流程未启动 — log 不存在)"
fi

echo ""
echo "=== [5] Task B details 目录 ==="
TASKB_DIR=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_b_regenerate
if [ -d "$TASKB_DIR/details" ]; then
    ls -la "$TASKB_DIR/details" 2>/dev/null | head -25
else
    echo "(details 目录不存在)"
fi
