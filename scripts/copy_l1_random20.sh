#!/bin/bash
KB="/home/kbuser/projects/KernelBench-0"
RUN="$KB/runs/iter_full_l1_caesar_paper_v2"
OUT="/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/test_data/l1_smoke20"

echo "=== Check paths ==="
echo "KB exists: $([ -d $KB ] && echo YES || echo NO)"
echo "RUN exists: $([ -d $RUN ] && echo YES || echo NO)"

# Count available kernels
count=$(ls "$RUN"/level_1_problem_*_kernel.py 2>/dev/null | wc -l)
echo "Total L1 kernels: $count"

if [ "$count" -eq 0 ]; then
    echo "ERROR: No kernel files found"
    exit 1
fi

# Create output directory
mkdir -p "$OUT"

# Random pick 20 (or all if < 20)
ls "$RUN"/level_1_problem_*_kernel.py 2>/dev/null | shuf -n 20 | while read f; do
    cp "$f" "$OUT/"
    echo "Copied: $(basename $f)"
done

echo "=== Done: $(ls $OUT/*.py 2>/dev/null | wc -l) files copied ==="
