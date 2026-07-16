# MutaKernel Mutation Testing Report

## Overall

- Total kernels: 9
- Total mutants: 406
- Killed: 230
- Survived: 167
- Stillborn: 9
- Strict Equivalent: 0
- Candidate Equivalent: 0
- **Conservative Score: 57.93%** (excl. strict equiv)
- **Optimistic Score: 57.93%** (excl. strict + candidate)

## By Category

| Category | Name | Total | Killed | Survived | Stillborn | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|------|-------|--------|----------|---------- |-----------|---------|-------|-------------|
| A | Arithmetic (baseline) | 304 | 178 | 117 | 9 | 0 | 0 | 60.34% | 60.34% |
| B | GPU Parallel Semantics | 30 | 30 | 0 | 0 | 0 | 0 | 100.00% | 100.00% |
| C | ML Numerical Semantics | 68 | 22 | 46 | 0 | 0 | 0 | 32.35% | 32.35% |
| D | LLM Error Patterns | 4 | 0 | 4 | 0 | 0 | 0 | 0.00% | 0.00% |

## By Operator

| Operator | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|-------|--------|----------|-----------|---------|-------|-------------|
| acc_downgrade | 20 | 7 | 13 | 0 | 0 | 35.00% | 35.00% |
| arith_replace | 180 | 139 | 41 | 0 | 0 | 77.22% | 77.22% |
| broadcast_unsafe | 2 | 0 | 2 | 0 | 0 | 0.00% | 0.00% |
| cast_remove | 14 | 2 | 12 | 0 | 0 | 14.29% | 14.29% |
| const_perturb | 70 | 30 | 40 | 0 | 0 | 42.86% | 42.86% |
| epsilon_modify | 6 | 0 | 6 | 0 | 0 | 0.00% | 0.00% |
| index_replace | 24 | 24 | 0 | 0 | 0 | 100.00% | 100.00% |
| init_modify | 10 | 1 | 9 | 0 | 0 | 10.00% | 10.00% |
| launch_config_mutate | 6 | 6 | 0 | 0 | 0 | 100.00% | 100.00% |
| layout_assume | 2 | 0 | 2 | 0 | 0 | 0.00% | 0.00% |
| reduction_reorder | 6 | 6 | 0 | 0 | 0 | 100.00% | 100.00% |
| relop_replace | 54 | 9 | 36 | 0 | 0 | 20.00% | 20.00% |
| scale_modify | 3 | 2 | 1 | 0 | 0 | 66.67% | 66.67% |
| stab_remove | 9 | 4 | 5 | 0 | 0 | 44.44% | 44.44% |

## By Kernel

| Kernel | Name | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|--------|------|-------|--------|----------|-----------|---------|-------|-------------|
| L0_P5 | softmax__bfloat16__1024x1024 | 31 | 10 | 20 | 0 | 0 | 33.33% | 33.33% |
| L0_P11 | rmsnorm__bfloat16__128x512 | 30 | 16 | 13 | 0 | 0 | 55.17% | 55.17% |
| L0_P17 | rmsnorm__bfloat16__2048x256 | 30 | 16 | 13 | 0 | 0 | 55.17% | 55.17% |
| L0_P23 | layernorm__bfloat16__512x1024 | 49 | 24 | 24 | 0 | 0 | 50.00% | 50.00% |
| L0_P29 | reduce__bfloat16__128x1024 | 26 | 14 | 11 | 0 | 0 | 56.00% | 56.00% |
| L0_P35 | reduce__bfloat16__1024x2048 | 26 | 14 | 11 | 0 | 0 | 56.00% | 56.00% |
| L0_P41 | matmul__bfloat16__512x256x512 | 42 | 34 | 7 | 0 | 0 | 82.93% | 82.93% |
| L0_P47 | cross_entropy__bfloat16__1024x | 48 | 15 | 32 | 0 | 0 | 31.91% | 31.91% |
| L0_P65 | flash_attention__bfloat16__2x4 | 124 | 87 | 36 | 0 | 0 | 70.73% | 70.73% |
