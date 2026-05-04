# MutaKernel Mutation Testing Report

## Overall

- Total kernels: 90
- Total mutants: 1646
- Killed: 939
- Survived: 270
- Stillborn: 163
- Strict Equivalent: 10
- Candidate Equivalent: 264
- **Conservative Score: 63.75%** (excl. strict equiv)
- **Optimistic Score: 77.67%** (excl. strict + candidate)

## By Category

| Category | Name | Total | Killed | Survived | Stillborn | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|------|-------|--------|----------|---------- |-----------|---------|-------|-------------|
| A | Arithmetic (baseline) | 757 | 378 | 133 | 96 | 10 | 140 | 58.06% | 73.97% |
| B | GPU Parallel Semantics | 702 | 494 | 99 | 3 | 0 | 106 | 70.67% | 83.31% |
| C | ML Numerical Semantics | 178 | 63 | 36 | 63 | 0 | 16 | 54.78% | 63.64% |
| D | LLM Error Patterns | 9 | 4 | 2 | 1 | 0 | 2 | 50.00% | 66.67% |

## By Operator

| Operator | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|----------|-------|--------|----------|-----------|---------|-------|-------------|
| acc_downgrade | 46 | 6 | 0 | 0 | 0 | 100.00% | 100.00% |
| arith_replace | 268 | 177 | 43 | 1 | 29 | 71.08% | 80.45% |
| broadcast_unsafe | 4 | 0 | 2 | 0 | 2 | 0.00% | 0.00% |
| cast_remove | 38 | 5 | 3 | 0 | 10 | 27.78% | 62.50% |
| const_perturb | 222 | 108 | 46 | 9 | 50 | 52.94% | 70.13% |
| epsilon_modify | 32 | 13 | 14 | 0 | 2 | 44.83% | 48.15% |
| index_replace | 267 | 204 | 22 | 0 | 41 | 76.40% | 90.27% |
| init_modify | 22 | 10 | 9 | 0 | 3 | 45.45% | 52.63% |
| launch_config_mutate | 112 | 90 | 15 | 0 | 7 | 80.36% | 85.71% |
| layout_assume | 5 | 4 | 0 | 0 | 0 | 100.00% | 100.00% |
| mask_boundary | 219 | 131 | 37 | 0 | 48 | 60.65% | 77.98% |
| relop_replace | 267 | 93 | 44 | 0 | 61 | 46.97% | 67.88% |
| scale_modify | 33 | 24 | 8 | 0 | 1 | 72.73% | 75.00% |
| stab_remove | 7 | 5 | 2 | 0 | 0 | 71.43% | 71.43% |
| sync_remove | 104 | 69 | 25 | 0 | 10 | 66.35% | 73.40% |

## By Kernel

| Kernel | Name | Total | Killed | Survived | Strict Eq | Cand Eq | Score | Score (Opt) |
|--------|------|-------|--------|----------|-----------|---------|-------|-------------|
| L1_P1 | L1_P1 | 19 | 14 | 1 | 2 | 1 | 87.50% | 93.33% |
| L1_P10 | L1_P10 | 18 | 13 | 0 | 0 | 5 | 72.22% | 100.00% |
| L1_P100 | L1_P100 | 16 | 5 | 7 | 0 | 2 | 35.71% | 41.67% |
| L1_P12 | L1_P12 | 17 | 5 | 8 | 0 | 4 | 29.41% | 38.46% |
| L1_P13 | L1_P13 | 20 | 14 | 0 | 0 | 6 | 70.00% | 100.00% |
| L1_P14 | L1_P14 | 19 | 10 | 1 | 2 | 6 | 58.82% | 90.91% |
| L1_P15 | L1_P15 | 13 | 11 | 0 | 0 | 2 | 84.62% | 100.00% |
| L1_P16 | L1_P16 | 17 | 12 | 3 | 0 | 2 | 70.59% | 80.00% |
| L1_P17 | L1_P17 | 17 | 12 | 1 | 0 | 4 | 70.59% | 92.31% |
| L1_P18 | L1_P18 | 12 | 8 | 1 | 0 | 3 | 66.67% | 88.89% |
| L1_P19 | L1_P19 | 11 | 6 | 2 | 0 | 1 | 66.67% | 75.00% |
| L1_P2 | L1_P2 | 17 | 14 | 1 | 0 | 2 | 82.35% | 93.33% |
| L1_P20 | L1_P20 | 13 | 6 | 0 | 0 | 6 | 50.00% | 100.00% |
| L1_P21 | L1_P21 | 14 | 8 | 0 | 1 | 5 | 61.54% | 100.00% |
| L1_P22 | L1_P22 | 18 | 3 | 3 | 0 | 6 | 25.00% | 50.00% |
| L1_P23 | L1_P23 | 24 | 1 | 9 | 1 | 7 | 5.88% | 10.00% |
| L1_P24 | L1_P24 | 25 | 9 | 3 | 0 | 6 | 50.00% | 75.00% |
| L1_P25 | L1_P25 | 14 | 7 | 0 | 0 | 2 | 77.78% | 100.00% |
| L1_P27 | L1_P27 | 19 | 4 | 1 | 0 | 9 | 28.57% | 80.00% |
| L1_P28 | L1_P28 | 14 | 2 | 1 | 0 | 11 | 14.29% | 66.67% |
| L1_P29 | L1_P29 | 22 | 4 | 3 | 0 | 4 | 36.36% | 57.14% |
| L1_P3 | L1_P3 | 20 | 16 | 0 | 0 | 4 | 80.00% | 100.00% |
| L1_P30 | L1_P30 | 15 | 3 | 1 | 0 | 5 | 33.33% | 75.00% |
| L1_P31 | L1_P31 | 15 | 5 | 1 | 0 | 8 | 35.71% | 83.33% |
| L1_P32 | L1_P32 | 14 | 5 | 0 | 0 | 7 | 41.67% | 100.00% |
| L1_P33 | L1_P33 | 24 | 4 | 6 | 0 | 12 | 18.18% | 40.00% |
| L1_P34 | L1_P34 | 25 | 5 | 14 | 0 | 1 | 25.00% | 26.32% |
| L1_P35 | L1_P35 | 26 | 3 | 12 | 0 | 6 | 14.29% | 20.00% |
| L1_P37 | L1_P37 | 19 | 12 | 3 | 0 | 0 | 80.00% | 80.00% |
| L1_P38 | L1_P38 | 20 | 3 | 8 | 0 | 6 | 17.65% | 27.27% |
| L1_P39 | L1_P39 | 20 | 11 | 2 | 0 | 2 | 73.33% | 84.62% |
| L1_P40 | L1_P40 | 26 | 7 | 7 | 0 | 3 | 41.18% | 50.00% |
| L1_P41 | L1_P41 | 24 | 6 | 8 | 0 | 9 | 26.09% | 42.86% |
| L1_P42 | L1_P42 | 19 | 8 | 6 | 0 | 4 | 44.44% | 57.14% |
| L1_P44 | L1_P44 | 16 | 9 | 2 | 0 | 5 | 56.25% | 81.82% |
| L1_P46 | L1_P46 | 15 | 8 | 2 | 0 | 5 | 53.33% | 80.00% |
| L1_P47 | L1_P47 | 13 | 7 | 3 | 0 | 3 | 53.85% | 70.00% |
| L1_P48 | L1_P48 | 14 | 10 | 1 | 0 | 1 | 83.33% | 90.91% |
| L1_P49 | L1_P49 | 19 | 5 | 8 | 0 | 4 | 29.41% | 38.46% |
| L1_P5 | L1_P5 | 10 | 7 | 1 | 0 | 1 | 77.78% | 87.50% |
| L1_P50 | L1_P50 | 17 | 3 | 11 | 0 | 3 | 17.65% | 21.43% |
| L1_P51 | L1_P51 | 13 | 5 | 4 | 0 | 3 | 41.67% | 55.56% |
| L1_P52 | L1_P52 | 18 | 7 | 3 | 0 | 8 | 38.89% | 70.00% |
| L1_P53 | L1_P53 | 15 | 3 | 7 | 0 | 5 | 20.00% | 30.00% |
| L1_P6 | L1_P6 | 5 | 0 | 0 | 3 | 1 | 0.00% | 0.00% |
| L1_P65 | L1_P65 | 17 | 17 | 0 | 0 | 0 | 100.00% | 100.00% |
| L1_P69 | L1_P69 | 17 | 16 | 0 | 0 | 0 | 100.00% | 100.00% |
| L1_P7 | L1_P7 | 17 | 10 | 0 | 0 | 6 | 62.50% | 100.00% |
| L1_P74 | L1_P74 | 17 | 17 | 0 | 0 | 0 | 100.00% | 100.00% |
| L1_P8 | L1_P8 | 22 | 3 | 13 | 0 | 5 | 14.29% | 18.75% |
| L1_P80 | L1_P80 | 15 | 15 | 0 | 0 | 0 | 100.00% | 100.00% |
| L1_P88 | L1_P88 | 18 | 5 | 6 | 0 | 1 | 41.67% | 45.45% |
| L1_P89 | L1_P89 | 18 | 3 | 4 | 0 | 11 | 16.67% | 42.86% |
| L1_P9 | L1_P9 | 17 | 12 | 3 | 0 | 2 | 70.59% | 80.00% |
| L1_P90 | L1_P90 | 12 | 2 | 5 | 0 | 4 | 18.18% | 28.57% |
| L1_P91 | L1_P91 | 16 | 3 | 4 | 0 | 6 | 23.08% | 42.86% |
| L1_P93 | L1_P93 | 11 | 6 | 2 | 0 | 3 | 54.55% | 75.00% |
| L1_P94 | L1_P94 | 18 | 12 | 4 | 0 | 1 | 70.59% | 75.00% |
| L1_P95 | L1_P95 | 20 | 11 | 8 | 0 | 0 | 57.89% | 57.89% |
| L1_P96 | L1_P96 | 19 | 7 | 10 | 0 | 0 | 41.18% | 41.18% |
| L1_P97 | L1_P97 | 20 | 7 | 11 | 0 | 0 | 38.89% | 38.89% |
| L1_P98 | L1_P98 | 19 | 6 | 9 | 0 | 1 | 37.50% | 40.00% |
| L1_P99 | L1_P99 | 22 | 13 | 5 | 0 | 4 | 59.09% | 72.22% |
| L2_P14 | L2_P14 | 17 | 17 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P17 | L2_P17 | 31 | 26 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P24 | L2_P24 | 24 | 21 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P28 | L2_P28 | 21 | 18 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P29 | L2_P29 | 16 | 16 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P30 | L2_P30 | 22 | 17 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P32 | L2_P32 | 17 | 16 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P4 | L2_P4 | 18 | 17 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P40 | L2_P40 | 22 | 17 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P41 | L2_P41 | 30 | 3 | 22 | 0 | 0 | 12.00% | 12.00% |
| L2_P44 | L2_P44 | 19 | 17 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P46 | L2_P46 | 17 | 17 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P53 | L2_P53 | 21 | 20 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P56 | L2_P56 | 16 | 14 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P58 | L2_P58 | 21 | 9 | 5 | 0 | 7 | 42.86% | 64.29% |
| L2_P59 | L2_P59 | 21 | 20 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P63 | L2_P63 | 21 | 20 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P64 | L2_P64 | 19 | 19 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P66 | L2_P66 | 17 | 0 | 3 | 1 | 13 | 0.00% | 0.00% |
| L2_P68 | L2_P68 | 18 | 17 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P73 | L2_P73 | 21 | 21 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P81 | L2_P81 | 22 | 20 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P84 | L2_P84 | 26 | 22 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P86 | L2_P86 | 21 | 21 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P87 | L2_P87 | 18 | 18 | 0 | 0 | 0 | 100.00% | 100.00% |
| L2_P9 | L2_P9 | 16 | 13 | 1 | 0 | 0 | 92.86% | 92.86% |
| L2_P98 | L2_P98 | 18 | 18 | 0 | 0 | 0 | 100.00% | 100.00% |

## Equivalent Mutant Details

| Kernel | Mutant | Operator | Level | Evidence |
|--------|--------|----------|-------|----------|
| L1_P1 | L1_P1__relop_replace__7 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 11271ms) |
| L1_P1 | L1_P1__const_perturb__0 | const_perturb | Strict Equivalent | Static rule: dead_host_constant |
| L1_P1 | L1_P1__const_perturb__1 | const_perturb | Strict Equivalent | Static rule: dead_host_constant |
| L1_P10 | L1_P10__arith_replace__23 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 47372ms) |
| L1_P10 | L1_P10__relop_replace__8 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 45624ms) |
| L1_P10 | L1_P10__relop_replace__7 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 46047ms) |
| L1_P10 | L1_P10__relop_replace__5 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 45946ms) |
| L1_P10 | L1_P10__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 45808ms) |
| L1_P100 | L1_P100__const_perturb__0 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3013ms) |
| L1_P100 | L1_P100__index_replace__1 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3178ms) |
| L1_P12 | L1_P12__arith_replace__17 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 16236ms) |
| L1_P12 | L1_P12__const_perturb__1 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 16240ms, CUDA strings identica |
| L1_P12 | L1_P12__const_perturb__5 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 16098ms) |
| L1_P12 | L1_P12__index_replace__10 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 16417ms) |
| L1_P13 | L1_P13__arith_replace__4 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 57480ms) |
| L1_P13 | L1_P13__relop_replace__4 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 57493ms) |
| L1_P13 | L1_P13__relop_replace__6 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 57728ms) |
| L1_P13 | L1_P13__relop_replace__0 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 58134ms) |
| L1_P13 | L1_P13__cast_remove__0 | cast_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 58310ms, CUDA strings identica |
| L1_P13 | L1_P13__cast_remove__1 | cast_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 58561ms, CUDA strings identica |
| L1_P14 | L1_P14__arith_replace__0 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 32463ms) |
| L1_P14 | L1_P14__arith_replace__30 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 32601ms) |
| L1_P14 | L1_P14__arith_replace__33 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 32648ms) |
| L1_P14 | L1_P14__relop_replace__6 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 32187ms) |
| L1_P14 | L1_P14__const_perturb__0 | const_perturb | Strict Equivalent | Static rule: dead_host_constant |
| L1_P14 | L1_P14__const_perturb__1 | const_perturb | Strict Equivalent | Static rule: dead_host_constant |
| L1_P14 | L1_P14__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 32155ms) |
| L1_P14 | L1_P14__launch_config_mut | launch_config_mutate | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 32058ms) |
| L1_P15 | L1_P15__arith_replace__11 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 34707ms) |
| L1_P15 | L1_P15__relop_replace__1 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 34314ms) |
| L1_P16 | L1_P16__relop_replace__0 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 14414ms) |
| L1_P16 | L1_P16__relop_replace__1 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 13946ms) |
| L1_P17 | L1_P17__relop_replace__7 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 15434ms) |
| L1_P17 | L1_P17__relop_replace__0 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 14202ms) |
| L1_P17 | L1_P17__relop_replace__1 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 14293ms) |
| L1_P17 | L1_P17__index_replace__19 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 14857ms) |
| L1_P18 | L1_P18__relop_replace__1 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 12729ms) |
| L1_P18 | L1_P18__relop_replace__3 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 11919ms) |
| L1_P18 | L1_P18__index_replace__14 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 12147ms) |
| L1_P19 | L1_P19__arith_replace__3 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2790ms) |
| L1_P2 | L1_P2__relop_replace__2 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 14487ms) |
| L1_P2 | L1_P2__relop_replace__3 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 13760ms) |
| L1_P20 | L1_P20__arith_replace__18 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2906ms) |
| L1_P20 | L1_P20__relop_replace__3 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3123ms) |
| L1_P20 | L1_P20__relop_replace__7 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3081ms) |
| L1_P20 | L1_P20__relop_replace__4 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3401ms) |
| L1_P20 | L1_P20__const_perturb__11 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3605ms) |
| L1_P20 | L1_P20__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3151ms) |
| L1_P21 | L1_P21__relop_replace__3 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3264ms) |
| L1_P21 | L1_P21__relop_replace__0 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3291ms) |
| L1_P21 | L1_P21__relop_replace__5 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3195ms, CUDA strings identical |
| L1_P21 | L1_P21__const_perturb__2 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3316ms, CUDA strings identical |
| L1_P21 | L1_P21__const_perturb__19 | const_perturb | Strict Equivalent | Static rule: dead_host_constant |
| L1_P21 | L1_P21__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3245ms) |
| L1_P22 | L1_P22__relop_replace__6 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3537ms) |
| L1_P22 | L1_P22__relop_replace__8 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2806ms) |
| L1_P22 | L1_P22__const_perturb__21 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2794ms, CUDA strings identical |
| L1_P22 | L1_P22__index_replace__10 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3110ms) |
| L1_P22 | L1_P22__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3043ms) |
| L1_P22 | L1_P22__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3624ms) |
| L1_P23 | L1_P23__const_perturb__1 | const_perturb | Strict Equivalent | Textually equivalent (full program normalization) |
| L1_P23 | L1_P23__index_replace__6 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3006ms) |
| L1_P23 | L1_P23__sync_remove__1 | sync_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3247ms) |
| L1_P23 | L1_P23__sync_remove__0 | sync_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2967ms) |
| L1_P23 | L1_P23__sync_remove__3 | sync_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3181ms) |
| L1_P23 | L1_P23__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3442ms) |
| L1_P23 | L1_P23__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2869ms) |
| L1_P23 | L1_P23__init_modify__0 | init_modify | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3422ms) |
| L1_P24 | L1_P24__const_perturb__9 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4536ms) |
| L1_P24 | L1_P24__const_perturb__4 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3326ms, CUDA strings identical |
| L1_P24 | L1_P24__const_perturb__13 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3490ms, CUDA strings identical |
| L1_P24 | L1_P24__index_replace__12 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3429ms) |
| L1_P24 | L1_P24__cast_remove__0 | cast_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3566ms) |
| L1_P24 | L1_P24__init_modify__0 | init_modify | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3297ms) |
| L1_P25 | L1_P25__arith_replace__12 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3526ms, CUDA strings identical |
| L1_P25 | L1_P25__relop_replace__3 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3012ms) |
| L1_P27 | L1_P27__arith_replace__2 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4530ms) |
| L1_P27 | L1_P27__arith_replace__6 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3203ms) |
| L1_P27 | L1_P27__relop_replace__5 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3260ms) |
| L1_P27 | L1_P27__relop_replace__19 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3593ms) |
| L1_P27 | L1_P27__const_perturb__22 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3165ms, CUDA strings identical |
| L1_P27 | L1_P27__const_perturb__35 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3461ms) |
| L1_P27 | L1_P27__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3196ms) |
| L1_P27 | L1_P27__launch_config_mut | launch_config_mutate | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3337ms) |
| L1_P27 | L1_P27__launch_config_mut | launch_config_mutate | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3353ms) |
| L1_P28 | L1_P28__relop_replace__3 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4285ms) |
| L1_P28 | L1_P28__relop_replace__1 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2999ms) |
| L1_P28 | L1_P28__relop_replace__2 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3539ms) |
| L1_P28 | L1_P28__const_perturb__3 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3335ms, CUDA strings identical |
| L1_P28 | L1_P28__const_perturb__4 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3286ms, CUDA strings identical |
| L1_P28 | L1_P28__const_perturb__26 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3186ms) |
| L1_P28 | L1_P28__index_replace__6 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3422ms) |
| L1_P28 | L1_P28__index_replace__11 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3396ms) |
| L1_P28 | L1_P28__index_replace__10 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3439ms) |
| L1_P28 | L1_P28__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3535ms) |
| L1_P28 | L1_P28__launch_config_mut | launch_config_mutate | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3597ms) |
| L1_P29 | L1_P29__arith_replace__16 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 30487ms) |
| L1_P29 | L1_P29__arith_replace__19 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3341ms) |
| L1_P29 | L1_P29__relop_replace__9 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3357ms) |
| L1_P29 | L1_P29__cast_remove__4 | cast_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3305ms) |
| L1_P3 | L1_P3__arith_replace__31 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 19103ms) |
| L1_P3 | L1_P3__relop_replace__4 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 18825ms) |
| L1_P3 | L1_P3__relop_replace__1 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 18415ms) |
| L1_P3 | L1_P3__relop_replace__2 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 18213ms) |
| L1_P30 | L1_P30__relop_replace__1 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3122ms) |
| L1_P30 | L1_P30__index_replace__11 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3138ms) |
| L1_P30 | L1_P30__index_replace__8 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2864ms) |
| L1_P30 | L1_P30__index_replace__9 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2837ms) |
| L1_P30 | L1_P30__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3296ms) |
| L1_P31 | L1_P31__arith_replace__3 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4397ms) |
| L1_P31 | L1_P31__relop_replace__14 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3392ms) |
| L1_P31 | L1_P31__const_perturb__18 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3326ms) |
| L1_P31 | L1_P31__const_perturb__2 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3352ms) |
| L1_P31 | L1_P31__index_replace__6 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3312ms) |
| L1_P31 | L1_P31__index_replace__8 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3406ms) |
| L1_P31 | L1_P31__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3098ms) |
| L1_P31 | L1_P31__launch_config_mut | launch_config_mutate | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3029ms) |
| L1_P32 | L1_P32__arith_replace__9 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3476ms) |
| L1_P32 | L1_P32__arith_replace__8 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3081ms) |
| L1_P32 | L1_P32__relop_replace__2 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3900ms) |
| L1_P32 | L1_P32__const_perturb__1 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3168ms, CUDA strings identical |
| L1_P32 | L1_P32__const_perturb__5 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3021ms) |
| L1_P32 | L1_P32__const_perturb__2 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3200ms) |
| L1_P32 | L1_P32__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3323ms) |
| L1_P33 | L1_P33__arith_replace__58 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56466ms) |
| L1_P33 | L1_P33__relop_replace__0 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 55611ms) |
| L1_P33 | L1_P33__const_perturb__21 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56246ms, CUDA strings identica |
| L1_P33 | L1_P33__const_perturb__15 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56754ms) |
| L1_P33 | L1_P33__const_perturb__11 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 55442ms, CUDA strings identica |
| L1_P33 | L1_P33__index_replace__22 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56173ms) |
| L1_P33 | L1_P33__index_replace__10 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56357ms) |
| L1_P33 | L1_P33__sync_remove__0 | sync_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56051ms) |
| L1_P33 | L1_P33__sync_remove__1 | sync_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56013ms) |
| L1_P33 | L1_P33__mask_boundary__2 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56669ms) |
| L1_P33 | L1_P33__mask_boundary__3 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56393ms) |
| L1_P33 | L1_P33__mask_boundary__6 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56310ms) |
| L1_P34 | L1_P34__sync_remove__7 | sync_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56415ms) |
| L1_P35 | L1_P35__relop_replace__7 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 55786ms) |
| L1_P35 | L1_P35__index_replace__35 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56705ms) |
| L1_P35 | L1_P35__index_replace__39 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 68947ms) |
| L1_P35 | L1_P35__mask_boundary__2 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 57349ms) |
| L1_P35 | L1_P35__mask_boundary__3 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 57128ms) |
| L1_P35 | L1_P35__cast_remove__0 | cast_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 56246ms) |
| L1_P38 | L1_P38__relop_replace__11 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3323ms) |
| L1_P38 | L1_P38__relop_replace__5 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3239ms) |
| L1_P38 | L1_P38__index_replace__8 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2898ms) |
| L1_P38 | L1_P38__cast_remove__0 | cast_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3131ms) |
| L1_P38 | L1_P38__cast_remove__3 | cast_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2772ms) |
| L1_P38 | L1_P38__cast_remove__1 | cast_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2819ms) |
| L1_P39 | L1_P39__const_perturb__0 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4472ms, CUDA strings identical |
| L1_P39 | L1_P39__const_perturb__2 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3462ms, CUDA strings identical |
| L1_P40 | L1_P40__index_replace__12 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 102713ms) |
| L1_P40 | L1_P40__sync_remove__2 | sync_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 59199ms) |
| L1_P40 | L1_P40__cast_remove__0 | cast_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 59312ms) |
| L1_P41 | L1_P41__arith_replace__54 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 11116ms) |
| L1_P41 | L1_P41__arith_replace__29 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3324ms) |
| L1_P41 | L1_P41__relop_replace__6 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3129ms) |
| L1_P41 | L1_P41__const_perturb__0 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3343ms) |
| L1_P41 | L1_P41__const_perturb__7 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3107ms) |
| L1_P41 | L1_P41__index_replace__19 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3250ms) |
| L1_P41 | L1_P41__mask_boundary__3 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3555ms) |
| L1_P41 | L1_P41__launch_config_mut | launch_config_mutate | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3151ms) |
| L1_P41 | L1_P41__broadcast_unsafe_ | broadcast_unsafe | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3189ms) |
| L1_P42 | L1_P42__mask_boundary__7 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 10207ms) |
| L1_P42 | L1_P42__mask_boundary__3 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 10230ms) |
| L1_P42 | L1_P42__init_modify__0 | init_modify | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 9880ms) |
| L1_P42 | L1_P42__broadcast_unsafe_ | broadcast_unsafe | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 9699ms) |
| L1_P44 | L1_P44__relop_replace__5 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3093ms) |
| L1_P44 | L1_P44__relop_replace__0 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3270ms) |
| L1_P44 | L1_P44__const_perturb__4 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2967ms) |
| L1_P44 | L1_P44__index_replace__6 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 2735ms) |
| L1_P44 | L1_P44__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3136ms) |
| L1_P46 | L1_P46__relop_replace__5 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 94601ms) |
| L1_P46 | L1_P46__relop_replace__0 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 95436ms) |
| L1_P46 | L1_P46__const_perturb__0 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 96786ms) |
| L1_P46 | L1_P46__index_replace__5 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 109086ms) |
| L1_P46 | L1_P46__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 96715ms) |
| L1_P47 | L1_P47__index_replace__3 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3633ms) |
| L1_P47 | L1_P47__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3905ms) |
| L1_P47 | L1_P47__mask_boundary__5 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4150ms) |
| L1_P48 | L1_P48__const_perturb__0 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4333ms) |
| L1_P49 | L1_P49__relop_replace__12 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3861ms) |
| L1_P49 | L1_P49__const_perturb__8 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4332ms) |
| L1_P49 | L1_P49__index_replace__6 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3636ms) |
| L1_P49 | L1_P49__index_replace__19 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3973ms) |
| L1_P5 | L1_P5__arith_replace__5 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 54613ms) |
| L1_P50 | L1_P50__const_perturb__3 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3889ms) |
| L1_P50 | L1_P50__index_replace__4 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3821ms) |
| L1_P50 | L1_P50__mask_boundary__4 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3694ms) |
| L1_P51 | L1_P51__arith_replace__15 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3641ms) |
| L1_P51 | L1_P51__index_replace__3 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3842ms) |
| L1_P51 | L1_P51__mask_boundary__3 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3795ms) |
| L1_P52 | L1_P52__relop_replace__17 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3466ms) |
| L1_P52 | L1_P52__const_perturb__0 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3928ms) |
| L1_P52 | L1_P52__const_perturb__10 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3444ms) |
| L1_P52 | L1_P52__const_perturb__7 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3885ms) |
| L1_P52 | L1_P52__index_replace__5 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3220ms) |
| L1_P52 | L1_P52__index_replace__4 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3623ms) |
| L1_P52 | L1_P52__mask_boundary__4 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3565ms) |
| L1_P52 | L1_P52__launch_config_mut | launch_config_mutate | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3568ms) |
| L1_P53 | L1_P53__const_perturb__0 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3690ms, CUDA strings identical |
| L1_P53 | L1_P53__const_perturb__4 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3528ms) |
| L1_P53 | L1_P53__const_perturb__7 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4608ms) |
| L1_P53 | L1_P53__index_replace__8 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4005ms) |
| L1_P53 | L1_P53__index_replace__12 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4243ms) |
| L1_P6 | L1_P6__arith_replace__0 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 51570ms) |
| L1_P6 | L1_P6__const_perturb__4 | const_perturb | Strict Equivalent | Static rule: dead_host_constant |
| L1_P6 | L1_P6__const_perturb__5 | const_perturb | Strict Equivalent | Static rule: dead_host_constant |
| L1_P6 | L1_P6__const_perturb__2 | const_perturb | Strict Equivalent | Static rule: dead_host_constant |
| L1_P7 | L1_P7__relop_replace__1 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 58446ms) |
| L1_P7 | L1_P7__relop_replace__5 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 54907ms) |
| L1_P7 | L1_P7__relop_replace__8 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 55276ms) |
| L1_P7 | L1_P7__const_perturb__2 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 54682ms) |
| L1_P7 | L1_P7__const_perturb__17 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 54620ms) |
| L1_P7 | L1_P7__sync_remove__1 | sync_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 55282ms) |
| L1_P8 | L1_P8__relop_replace__11 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 88398ms) |
| L1_P8 | L1_P8__const_perturb__5 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 90199ms) |
| L1_P8 | L1_P8__index_replace__4 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 88969ms) |
| L1_P8 | L1_P8__epsilon_modify__1 | epsilon_modify | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 90172ms, CUDA strings identica |
| L1_P8 | L1_P8__epsilon_modify__0 | epsilon_modify | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 89395ms, CUDA strings identica |
| L1_P88 | L1_P88__index_replace__9 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 5301ms) |
| L1_P89 | L1_P89__arith_replace__11 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4699ms) |
| L1_P89 | L1_P89__relop_replace__2 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3467ms) |
| L1_P89 | L1_P89__relop_replace__12 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3440ms) |
| L1_P89 | L1_P89__const_perturb__9 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3145ms) |
| L1_P89 | L1_P89__const_perturb__0 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3533ms) |
| L1_P89 | L1_P89__const_perturb__12 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3226ms) |
| L1_P89 | L1_P89__index_replace__1 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3036ms) |
| L1_P89 | L1_P89__index_replace__9 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3546ms) |
| L1_P89 | L1_P89__mask_boundary__3 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3421ms) |
| L1_P89 | L1_P89__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3044ms) |
| L1_P89 | L1_P89__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3652ms) |
| L1_P9 | L1_P9__const_perturb__32 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 92509ms) |
| L1_P9 | L1_P9__sync_remove__1 | sync_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 91748ms) |
| L1_P90 | L1_P90__relop_replace__1 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3301ms) |
| L1_P90 | L1_P90__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3541ms) |
| L1_P90 | L1_P90__mask_boundary__3 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3309ms) |
| L1_P90 | L1_P90__mask_boundary__2 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3276ms) |
| L1_P91 | L1_P91__arith_replace__30 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4474ms) |
| L1_P91 | L1_P91__const_perturb__0 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3498ms, CUDA strings identical |
| L1_P91 | L1_P91__index_replace__13 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3575ms) |
| L1_P91 | L1_P91__index_replace__4 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3236ms) |
| L1_P91 | L1_P91__mask_boundary__2 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3301ms) |
| L1_P91 | L1_P91__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3791ms) |
| L1_P93 | L1_P93__index_replace__2 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3638ms) |
| L1_P93 | L1_P93__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 3894ms) |
| L1_P93 | L1_P93__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4057ms) |
| L1_P94 | L1_P94__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 4124ms) |
| L1_P98 | L1_P98__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 44558ms) |
| L1_P99 | L1_P99__relop_replace__2 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 41694ms) |
| L1_P99 | L1_P99__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 42000ms) |
| L1_P99 | L1_P99__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 41659ms) |
| L1_P99 | L1_P99__cast_remove__0 | cast_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 41993ms, CUDA strings identica |
| L2_P58 | L2_P58__arith_replace__9 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 98121ms) |
| L2_P58 | L2_P58__const_perturb__3 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 60238ms, CUDA strings identica |
| L2_P58 | L2_P58__const_perturb__17 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 60082ms) |
| L2_P58 | L2_P58__sync_remove__0 | sync_remove | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 59686ms) |
| L2_P58 | L2_P58__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 59942ms) |
| L2_P58 | L2_P58__mask_boundary__2 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 59589ms) |
| L2_P58 | L2_P58__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 59627ms) |
| L2_P66 | L2_P66__arith_replace__4 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 84010ms) |
| L2_P66 | L2_P66__arith_replace__2 | arith_replace | Strict Equivalent | Textually equivalent (full program normalization) |
| L2_P66 | L2_P66__arith_replace__14 | arith_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 42586ms, CUDA strings identica |
| L2_P66 | L2_P66__relop_replace__6 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 42302ms) |
| L2_P66 | L2_P66__relop_replace__4 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 42939ms) |
| L2_P66 | L2_P66__relop_replace__9 | relop_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 44113ms, CUDA strings identica |
| L2_P66 | L2_P66__const_perturb__3 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 45737ms) |
| L2_P66 | L2_P66__const_perturb__0 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 45226ms) |
| L2_P66 | L2_P66__const_perturb__2 | const_perturb | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 43626ms) |
| L2_P66 | L2_P66__index_replace__2 | index_replace | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 42954ms) |
| L2_P66 | L2_P66__mask_boundary__1 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 42754ms) |
| L2_P66 | L2_P66__mask_boundary__0 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 44323ms) |
| L2_P66 | L2_P66__mask_boundary__3 | mask_boundary | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 44319ms) |
| L2_P66 | L2_P66__scale_modify__0 | scale_modify | Candidate Equivalent | Candidate equivalent (112 rounds (random+stress), 44544ms, CUDA strings identica |
