# MutaKernel Mutation Testing Report

## Overall

- Total kernels: 9
- Total mutants: 430
- Killed: 237
- Survived: 184
- Stillborn: 9
- Strict Equivalent: 0
- Candidate Equivalent: 0
- **Conservative Score: 56.29%** (excl. strict equiv)
- **Optimistic Score: 56.29%** (excl. strict + candidate)

## By Category

| Category | Name | Total | Killed | Survived | Stillborn | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|------|-------|--------|----------|---------- |-----------|---------|-------|-------------|
| A | Arithmetic (baseline) | 323 | 182 | 132 | 9 | 0 | 0 | 57.96% | 57.96% |
| B | GPU Parallel Semantics | 30 | 30 | 0 | 0 | 0 | 0 | 100.00% | 100.00% |
| C | ML Numerical Semantics | 72 | 25 | 47 | 0 | 0 | 0 | 34.72% | 34.72% |
| D | LLM Error Patterns | 5 | 0 | 5 | 0 | 0 | 0 | 0.00% | 0.00% |

## By Operator

| Operator | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|-------|--------|----------|-----------|---------|-------|-------------|
| acc_downgrade | 19 | 6 | 13 | 0 | 0 | 31.58% | 31.58% |
| arith_replace | 186 | 140 | 46 | 0 | 0 | 75.27% | 75.27% |
| broadcast_unsafe | 4 | 0 | 4 | 0 | 0 | 0.00% | 0.00% |
| cast_remove | 14 | 2 | 12 | 0 | 0 | 14.29% | 14.29% |
| const_perturb | 78 | 30 | 48 | 0 | 0 | 38.46% | 38.46% |
| epsilon_modify | 6 | 1 | 5 | 0 | 0 | 16.67% | 16.67% |
| index_replace | 24 | 24 | 0 | 0 | 0 | 100.00% | 100.00% |
| init_modify | 12 | 1 | 11 | 0 | 0 | 8.33% | 8.33% |
| launch_config_mutate | 6 | 6 | 0 | 0 | 0 | 100.00% | 100.00% |
| layout_assume | 1 | 0 | 1 | 0 | 0 | 0.00% | 0.00% |
| reduction_reorder | 7 | 7 | 0 | 0 | 0 | 100.00% | 100.00% |
| relop_replace | 59 | 12 | 38 | 0 | 0 | 24.00% | 24.00% |
| scale_modify | 4 | 3 | 1 | 0 | 0 | 75.00% | 75.00% |
| stab_remove | 10 | 5 | 5 | 0 | 0 | 50.00% | 50.00% |

## By Kernel

| Kernel | Name | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|--------|------|-------|--------|----------|-----------|---------|-------|-------------|
| L0_P0 | softmax__float16__128x512 | 31 | 12 | 18 | 0 | 0 | 40.00% | 40.00% |
| L0_P6 | softmax__float16__2048x256 | 31 | 12 | 18 | 0 | 0 | 40.00% | 40.00% |
| L0_P12 | rmsnorm__float16__512x1024 | 30 | 16 | 13 | 0 | 0 | 55.17% | 55.17% |
| L0_P18 | layernorm__float16__128x512 | 49 | 24 | 24 | 0 | 0 | 50.00% | 50.00% |
| L0_P24 | layernorm__float16__2048x256 | 49 | 25 | 23 | 0 | 0 | 52.08% | 52.08% |
| L0_P30 | reduce__float16__256x4096 | 26 | 14 | 11 | 0 | 0 | 56.00% | 56.00% |
| L0_P36 | matmul__float16__256x256x256 | 42 | 34 | 7 | 0 | 0 | 82.93% | 82.93% |
| L0_P42 | cross_entropy__float16__256x51 | 48 | 17 | 30 | 0 | 0 | 36.17% | 36.17% |
| L0_P60 | flash_attention__float16__1x2x | 124 | 83 | 40 | 0 | 0 | 67.48% | 67.48% |
