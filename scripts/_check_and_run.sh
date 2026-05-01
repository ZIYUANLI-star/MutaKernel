#!/bin/bash
cd /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel

echo "=== Checking syntax ==="
/home/kbuser/projects/KernelBench-0/.venv/bin/python -m py_compile scripts/run_fullscale_diff_test.py
if [ $? -ne 0 ]; then
    echo "SYNTAX ERROR! Aborting."
    exit 1
fi
echo "SYNTAX OK"

echo "=== Starting full-scale diff test ==="
nohup /home/kbuser/projects/KernelBench-0/.venv/bin/python scripts/run_fullscale_diff_test.py --dataset cuda-l1 > /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/fullscale_v2.log 2>&1 &
echo "PID: $!"
sleep 3
echo "=== First lines of log ==="
head -20 /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/fullscale_v2.log 2>/dev/null || echo "(no output yet)"
