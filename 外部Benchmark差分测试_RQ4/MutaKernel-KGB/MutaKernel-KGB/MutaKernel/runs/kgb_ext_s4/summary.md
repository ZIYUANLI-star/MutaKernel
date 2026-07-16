# MutaKernel Mutation Testing Report

## Overall

- Total kernels: 9
- Total mutants: 484
- Killed: 242
- Survived: 233
- Stillborn: 9
- Strict Equivalent: 0
- Candidate Equivalent: 0
- **Conservative Score: 50.95%** (excl. strict equiv)
- **Optimistic Score: 50.95%** (excl. strict + candidate)

## By Category

| Category | Name | Total | Killed | Survived | Stillborn | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|------|-------|--------|----------|---------- |-----------|---------|-------|-------------|
| A | Arithmetic (baseline) | 378 | 193 | 176 | 9 | 0 | 0 | 52.30% | 52.30% |
| B | GPU Parallel Semantics | 27 | 25 | 2 | 0 | 0 | 0 | 92.59% | 92.59% |
| C | ML Numerical Semantics | 75 | 24 | 51 | 0 | 0 | 0 | 32.00% | 32.00% |
| D | LLM Error Patterns | 4 | 0 | 4 | 0 | 0 | 0 | 0.00% | 0.00% |

## By Operator

| Operator | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|-------|--------|----------|-----------|---------|-------|-------------|
| acc_downgrade | 23 | 10 | 13 | 0 | 0 | 43.48% | 43.48% |
| arith_replace | 216 | 147 | 69 | 0 | 0 | 68.06% | 68.06% |
| broadcast_unsafe | 2 | 0 | 2 | 0 | 0 | 0.00% | 0.00% |
| cast_remove | 18 | 0 | 18 | 0 | 0 | 0.00% | 0.00% |
| const_perturb | 108 | 35 | 73 | 0 | 0 | 32.41% | 32.41% |
| epsilon_modify | 6 | 2 | 4 | 0 | 0 | 33.33% | 33.33% |
| index_replace | 22 | 22 | 0 | 0 | 0 | 100.00% | 100.00% |
| init_modify | 10 | 1 | 9 | 0 | 0 | 10.00% | 10.00% |
| launch_config_mutate | 5 | 3 | 2 | 0 | 0 | 60.00% | 60.00% |
| layout_assume | 2 | 0 | 2 | 0 | 0 | 0.00% | 0.00% |
| reduction_reorder | 6 | 6 | 0 | 0 | 0 | 100.00% | 100.00% |
| relop_replace | 54 | 11 | 34 | 0 | 0 | 24.44% | 24.44% |
| scale_modify | 3 | 2 | 1 | 0 | 0 | 66.67% | 66.67% |
| stab_remove | 9 | 3 | 6 | 0 | 0 | 33.33% | 33.33% |

## By Kernel

| Kernel | Name | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|--------|------|-------|--------|----------|-----------|---------|-------|-------------|
| L0_P4 | softmax__float32__1024x1024 | 31 | 9 | 21 | 0 | 0 | 30.00% | 30.00% |
| L0_P10 | rmsnorm__float32__128x512 | 30 | 17 | 12 | 0 | 0 | 58.62% | 58.62% |
| L0_P16 | rmsnorm__float32__2048x256 | 30 | 17 | 12 | 0 | 0 | 58.62% | 58.62% |
| L0_P22 | layernorm__float32__512x1024 | 49 | 24 | 24 | 0 | 0 | 50.00% | 50.00% |
| L0_P28 | reduce__float32__128x1024 | 26 | 14 | 11 | 0 | 0 | 56.00% | 56.00% |
| L0_P34 | reduce__float32__1024x2048 | 26 | 14 | 11 | 0 | 0 | 56.00% | 56.00% |
| L0_P46 | cross_entropy__float32__1024x1 | 48 | 15 | 32 | 0 | 0 | 31.91% | 31.91% |
| L0_P52 | rotary_embedding__float32__256 | 120 | 47 | 72 | 0 | 0 | 39.50% | 39.50% |
| L0_P64 | flash_attention__float32__2x4x | 124 | 85 | 38 | 0 | 0 | 69.11% | 69.11% |
