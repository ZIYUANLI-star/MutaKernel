# MutaKernel Mutation Testing Report

## Overall

- Total kernels: 9
- Total mutants: 508
- Killed: 247
- Survived: 252
- Stillborn: 9
- Strict Equivalent: 0
- Candidate Equivalent: 0
- **Conservative Score: 49.50%** (excl. strict equiv)
- **Optimistic Score: 49.50%** (excl. strict + candidate)

## By Category

| Category | Name | Total | Killed | Survived | Stillborn | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|------|-------|--------|----------|---------- |-----------|---------|-------|-------------|
| A | Arithmetic (baseline) | 397 | 197 | 191 | 9 | 0 | 0 | 50.77% | 50.77% |
| B | GPU Parallel Semantics | 27 | 25 | 2 | 0 | 0 | 0 | 92.59% | 92.59% |
| C | ML Numerical Semantics | 79 | 25 | 54 | 0 | 0 | 0 | 31.65% | 31.65% |
| D | LLM Error Patterns | 5 | 0 | 5 | 0 | 0 | 0 | 0.00% | 0.00% |

## By Operator

| Operator | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|-------|--------|----------|-----------|---------|-------|-------------|
| acc_downgrade | 22 | 9 | 13 | 0 | 0 | 40.91% | 40.91% |
| arith_replace | 222 | 148 | 74 | 0 | 0 | 66.67% | 66.67% |
| broadcast_unsafe | 4 | 0 | 4 | 0 | 0 | 0.00% | 0.00% |
| cast_remove | 18 | 0 | 18 | 0 | 0 | 0.00% | 0.00% |
| const_perturb | 116 | 35 | 81 | 0 | 0 | 30.17% | 30.17% |
| epsilon_modify | 6 | 2 | 4 | 0 | 0 | 33.33% | 33.33% |
| index_replace | 22 | 22 | 0 | 0 | 0 | 100.00% | 100.00% |
| init_modify | 12 | 1 | 11 | 0 | 0 | 8.33% | 8.33% |
| launch_config_mutate | 5 | 3 | 2 | 0 | 0 | 60.00% | 60.00% |
| layout_assume | 1 | 0 | 1 | 0 | 0 | 0.00% | 0.00% |
| reduction_reorder | 7 | 7 | 0 | 0 | 0 | 100.00% | 100.00% |
| relop_replace | 59 | 14 | 36 | 0 | 0 | 28.00% | 28.00% |
| scale_modify | 4 | 3 | 1 | 0 | 0 | 75.00% | 75.00% |
| stab_remove | 10 | 3 | 7 | 0 | 0 | 30.00% | 30.00% |

## By Kernel

| Kernel | Name | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|--------|------|-------|--------|----------|-----------|---------|-------|-------------|
| L0_P1 | softmax__float32__128x512 | 31 | 12 | 18 | 0 | 0 | 40.00% | 40.00% |
| L0_P7 | softmax__float32__2048x256 | 31 | 11 | 19 | 0 | 0 | 36.67% | 36.67% |
| L0_P13 | rmsnorm__float32__512x1024 | 30 | 16 | 13 | 0 | 0 | 55.17% | 55.17% |
| L0_P19 | layernorm__float32__128x512 | 49 | 25 | 23 | 0 | 0 | 52.08% | 52.08% |
| L0_P25 | layernorm__float32__2048x256 | 49 | 25 | 23 | 0 | 0 | 52.08% | 52.08% |
| L0_P31 | reduce__float32__256x4096 | 26 | 14 | 11 | 0 | 0 | 56.00% | 56.00% |
| L0_P43 | cross_entropy__float32__256x51 | 48 | 16 | 31 | 0 | 0 | 34.04% | 34.04% |
| L0_P49 | rotary_embedding__float32__102 | 120 | 47 | 72 | 0 | 0 | 39.50% | 39.50% |
| L0_P61 | flash_attention__float32__1x2x | 124 | 81 | 42 | 0 | 0 | 65.85% | 65.85% |
