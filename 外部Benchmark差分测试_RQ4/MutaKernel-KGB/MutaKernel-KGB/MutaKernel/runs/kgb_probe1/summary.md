# MutaKernel Mutation Testing Report

## Overall

- Total kernels: 2
- Total mutants: 62
- Killed: 23
- Survived: 37
- Stillborn: 2
- Strict Equivalent: 0
- Candidate Equivalent: 0
- **Conservative Score: 38.33%** (excl. strict equiv)
- **Optimistic Score: 38.33%** (excl. strict + candidate)

## By Category

| Category | Name | Total | Killed | Survived | Stillborn | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|------|-------|--------|----------|---------- |-----------|---------|-------|-------------|
| A | Arithmetic (baseline) | 48 | 16 | 30 | 2 | 0 | 0 | 34.78% | 34.78% |
| B | GPU Parallel Semantics | 4 | 4 | 0 | 0 | 0 | 0 | 100.00% | 100.00% |
| C | ML Numerical Semantics | 8 | 3 | 5 | 0 | 0 | 0 | 37.50% | 37.50% |
| D | LLM Error Patterns | 2 | 0 | 2 | 0 | 0 | 0 | 0.00% | 0.00% |

## By Operator

| Operator | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|-------|--------|----------|-----------|---------|-------|-------------|
| arith_replace | 22 | 14 | 8 | 0 | 0 | 63.64% | 63.64% |
| broadcast_unsafe | 2 | 0 | 2 | 0 | 0 | 0.00% | 0.00% |
| const_perturb | 8 | 0 | 8 | 0 | 0 | 0.00% | 0.00% |
| index_replace | 4 | 4 | 0 | 0 | 0 | 100.00% | 100.00% |
| init_modify | 4 | 0 | 4 | 0 | 0 | 0.00% | 0.00% |
| reduction_reorder | 2 | 2 | 0 | 0 | 0 | 100.00% | 100.00% |
| relop_replace | 18 | 2 | 14 | 0 | 0 | 12.50% | 12.50% |
| stab_remove | 2 | 1 | 1 | 0 | 0 | 50.00% | 50.00% |

## By Kernel

| Kernel | Name | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|--------|------|-------|--------|----------|-----------|---------|-------|-------------|
| L0_P0 | softmax__float16__128x512 | 31 | 12 | 18 | 0 | 0 | 40.00% | 40.00% |
| L0_P1 | softmax__float32__128x512 | 31 | 11 | 19 | 0 | 0 | 36.67% | 36.67% |
