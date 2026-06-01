#!/usr/bin/env bash
set -uo pipefail
cd /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel
source /home/kbuser/projects/KernelBench-0/.venv/bin/activate
exec python -u scripts/_benchmark_taskB_speedup.py
