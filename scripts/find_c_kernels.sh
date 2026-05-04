#!/bin/bash
# Find CUDA kernel files that contain C-category patterns
BASE="/home/dog/KernelBench/KernelBench"
for lvl in level1 level2; do
    dir="$BASE/$lvl/generated_kernels"
    if [ ! -d "$dir" ]; then continue; fi
    for f in "$dir"/*.py; do
        [ -f "$f" ] || continue
        if grep -ql -E 'softmax|Softmax|layer_norm|LayerNorm|batch_norm|BatchNorm|rsqrt|epsilon|1e-5|1e-6|INFINITY|FLT_MAX|expf.*-|static_cast.*float|half2float|float2half' "$f" 2>/dev/null; then
            echo "$f"
        fi
    done
done
