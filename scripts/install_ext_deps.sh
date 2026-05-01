#!/bin/bash
set -e
source /home/kbuser/projects/KernelBench-0/.venv/bin/activate

echo "=== Installing NVIDIA Apex (with CUDA extensions) ==="
cd /tmp
if [ ! -d apex ]; then
    git clone https://github.com/NVIDIA/apex
fi
cd apex
git checkout master && git pull
MAX_JOBS=4 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" ./
echo "=== Apex installed ==="

echo ""
echo "=== Installing FlashAttention ==="
pip install flash-attn --no-build-isolation
echo "=== FlashAttention installed ==="

echo ""
echo "=== Verification ==="
python /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/scripts/check_env.py
