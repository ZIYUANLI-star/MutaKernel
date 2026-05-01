#!/bin/bash
cd /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel

echo "=== Checking syntax ==="
/home/kbuser/projects/KernelBench-0/.venv/bin/python -m py_compile scripts/run_fullscale_diff_test.py
if [ $? -ne 0 ]; then
    echo "SYNTAX ERROR! Aborting."
    exit 1
fi
echo "SYNTAX OK"

echo "=== GPU status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "(nvidia-smi failed)"

echo "=== Starting CUDA-L1 test (resuming from checkpoint, 52 done) ==="
nohup /home/kbuser/projects/KernelBench-0/.venv/bin/python scripts/run_fullscale_diff_test.py --dataset cuda-l1 > /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/fullscale_cuda_l1_v3.log 2>&1 &
echo "PID: $!"
disown

sleep 5
echo "=== First lines of log ==="
head -30 /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/fullscale_cuda_l1_v3.log 2>/dev/null || echo "(no output yet)"
echo "=== Process check ==="
ps aux | grep fullscale | grep -v grep
