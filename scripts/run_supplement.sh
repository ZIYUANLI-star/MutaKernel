#!/bin/bash
set -e

source /home/kbuser/projects/KernelBench-0/.venv/bin/activate
cd /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel

echo "=== Environment Check ==="
python --version
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>/dev/null || echo "No GPU info"

echo ""
echo "=== Starting Supplementary Enhancement Testing ==="
echo "This will pick up the 35 missing Tier 3 mutants (arith_replace, cast_remove, init_modify, scale_modify)"
echo "Already completed mutants will be skipped via completed.json"
echo ""

python scripts/run_stress_enhance.py 2>&1
