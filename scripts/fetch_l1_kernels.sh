#!/bin/bash
BASE="/home/dog/KernelBench/KernelBench/level1"
# Try both possible locations
for dir in "$BASE/generated_kernels" "$BASE"; do
    count=$(ls "$dir"/*.py 2>/dev/null | wc -l)
    if [ "$count" -gt 0 ]; then
        echo "DIR=$dir"
        echo "TOTAL=$count"
        ls "$dir"/*.py 2>/dev/null | shuf -n 20
        exit 0
    fi
done

# Try level1 problems structure
for dir in "$BASE"/*/; do
    if ls "$dir"*.py 2>/dev/null | head -1 > /dev/null 2>&1; then
        echo "DIR=$dir"
        find "$BASE" -name "*.py" -maxdepth 2 | head -5
        echo "..."
        break
    fi
done

# Fallback: show directory structure
echo "=== L1 structure ==="
find "$BASE" -maxdepth 3 -type f -name "*.py" | head -30
