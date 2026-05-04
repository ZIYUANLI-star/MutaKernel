#!/bin/bash
source /home/kbuser/projects/KernelBench-0/.venv/bin/activate
export TORCH_CUDA_ARCH_LIST="8.9"
export PYTHONUNBUFFERED=1

cd /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel

# Clean stale lock files from previous interrupted runs
find /home/kbuser/.cache/torch_extensions/ -name 'lock' -delete 2>/dev/null
echo "Cleaned stale lock files"

echo "=== Environment ==="
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "=== Running full kill rate test (L1+L2, rules v2) ==="
echo "=== Tolerance: atol=1e-2, rtol=1e-2 (KernelBench native) ==="

# Use nohup to survive terminal disconnects
nohup python3 scripts/kill_rate_full.py > /dev/null 2>&1 &
PID=$!
echo "Started kill_rate_full.py as PID=$PID (nohup background)"
echo "Monitor with: tail -f /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/kill_rate_full_output.txt"
echo "Kill with: kill $PID"
