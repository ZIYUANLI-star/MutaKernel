#!/bin/bash
cd /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
echo "Starting at $(date)"
exec /home/kbuser/projects/KernelBench-0/.venv/bin/python scripts/run_fullscale_diff_test.py --dataset cuda-l1
