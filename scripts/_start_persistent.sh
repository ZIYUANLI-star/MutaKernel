#!/bin/bash
cd /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
LOG=/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/fullscale_cuda_l1_v3.log
PYTHON=/home/kbuser/projects/KernelBench-0/.venv/bin/python

echo "Starting at $(date)" > "$LOG"
exec $PYTHON scripts/run_fullscale_diff_test.py --dataset cuda-l1 >> "$LOG" 2>&1
