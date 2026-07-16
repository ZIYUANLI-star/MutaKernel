# MutaKernel Mutation Testing Report

## Overall

- Total kernels: 14
- Total mutants: 778
- Killed: 413
- Survived: 351
- Stillborn: 14
- Strict Equivalent: 0
- Candidate Equivalent: 0
- **Conservative Score: 54.06%** (excl. strict equiv)
- **Optimistic Score: 54.06%** (excl. strict + candidate)

## By Category

| Category | Name | Total | Killed | Survived | Stillborn | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|------|-------|--------|----------|---------- |-----------|---------|-------|-------------|
| A | Arithmetic (baseline) | 600 | 323 | 263 | 14 | 0 | 0 | 55.12% | 55.12% |
| B | GPU Parallel Semantics | 49 | 47 | 2 | 0 | 0 | 0 | 95.92% | 95.92% |
| C | ML Numerical Semantics | 123 | 43 | 80 | 0 | 0 | 0 | 34.96% | 34.96% |
| D | LLM Error Patterns | 6 | 0 | 6 | 0 | 0 | 0 | 0.00% | 0.00% |

## By Operator

| Operator | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|-------|--------|----------|-----------|---------|-------|-------------|
| acc_downgrade | 35 | 15 | 20 | 0 | 0 | 42.86% | 42.86% |
| arith_replace | 342 | 240 | 102 | 0 | 0 | 70.18% | 70.18% |
| broadcast_unsafe | 4 | 0 | 4 | 0 | 0 | 0.00% | 0.00% |
| cast_remove | 26 | 2 | 24 | 0 | 0 | 7.69% | 7.69% |
| const_perturb | 170 | 65 | 105 | 0 | 0 | 38.24% | 38.24% |
| epsilon_modify | 8 | 3 | 5 | 0 | 0 | 37.50% | 37.50% |
| index_replace | 38 | 38 | 0 | 0 | 0 | 100.00% | 100.00% |
| init_modify | 20 | 2 | 18 | 0 | 0 | 10.00% | 10.00% |
| launch_config_mutate | 11 | 9 | 2 | 0 | 0 | 81.82% | 81.82% |
| layout_assume | 2 | 0 | 2 | 0 | 0 | 0.00% | 0.00% |
| reduction_reorder | 10 | 10 | 0 | 0 | 0 | 100.00% | 100.00% |
| relop_replace | 88 | 18 | 56 | 0 | 0 | 24.32% | 24.32% |
| scale_modify | 6 | 4 | 2 | 0 | 0 | 66.67% | 66.67% |
| stab_remove | 18 | 7 | 11 | 0 | 0 | 38.89% | 38.89% |

## By Kernel

| Kernel | Name | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|--------|------|-------|--------|----------|-----------|---------|-------|-------------|
| L0_P0 | softmax__float16__128x512 | 31 | 12 | 18 | 0 | 0 | 40.00% | 40.00% |
| L0_P1 | softmax__float32__128x512 | 31 | 12 | 18 | 0 | 0 | 40.00% | 40.00% |
| L0_P2 | rmsnorm__float16__128x512 | 30 | 17 | 12 | 0 | 0 | 58.62% | 58.62% |
| L0_P3 | rmsnorm__float32__128x512 | 30 | 17 | 12 | 0 | 0 | 58.62% | 58.62% |
| L0_P4 | layernorm__float16__128x512 | 49 | 24 | 24 | 0 | 0 | 50.00% | 50.00% |
| L0_P5 | layernorm__float32__128x512 | 49 | 25 | 23 | 0 | 0 | 52.08% | 52.08% |
| L0_P6 | reduce__float16__128x1024 | 26 | 14 | 11 | 0 | 0 | 56.00% | 56.00% |
| L0_P7 | reduce__float32__128x1024 | 26 | 14 | 11 | 0 | 0 | 56.00% | 56.00% |
| L0_P8 | matmul__float16__256x256x256 | 42 | 34 | 7 | 0 | 0 | 82.93% | 82.93% |
| L0_P10 | cross_entropy__float16__256x51 | 48 | 17 | 30 | 0 | 0 | 36.17% | 36.17% |
| L0_P11 | cross_entropy__float32__256x51 | 48 | 16 | 31 | 0 | 0 | 34.04% | 34.04% |
| L0_P13 | rotary_embedding__float32__102 | 120 | 47 | 72 | 0 | 0 | 39.50% | 39.50% |
| L0_P16 | flash_attention__float16__1x2x | 124 | 83 | 40 | 0 | 0 | 67.48% | 67.48% |
| L0_P17 | flash_attention__float32__1x2x | 124 | 81 | 42 | 0 | 0 | 65.85% | 65.85% |
