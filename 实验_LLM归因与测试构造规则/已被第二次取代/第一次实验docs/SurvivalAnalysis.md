# 存活变异体分析报告

> 基于三阶段实验数据（初始变异测试 → 4层增强测试 → LLM迭代分析）的综合分析。

## 一、三阶段杀伤汇总

| 阶段 | 输入变异体 | 杀死 | 存活 | 增量杀伤率 |
|------|----------|------|------|-----------|
| Phase 0: 初始变异测试 | 1663 (有效) | 944 | 322 | — |
| Phase 1: 4层增强测试 | 322 | 50 | 270 | 15.5% |
| Phase 2: LLM 分析 | 270 | 31 | 218 | 11.5% |

> 另有 163 个 stillborn（编译失败）、234 个 equivalent（静态等价）不计入有效变异体。

## 二、Phase 1 增强测试：各策略杀伤效果

| 策略 | 杀死数 | 主要针对算子 |
|------|--------|------------|
| `structured_ramp` | 10 | const_perturb(3), epsilon_modify(3), index_replace(2) |
| `all_negative` | 7 | sync_remove(3), const_perturb(2), arith_replace(1) |
| `large_magnitude` | 5 | index_replace(2), arith_replace(1), stab_remove(1) |
| `boundary_last_element` | 4 | mask_boundary(2), const_perturb(2) |
| `sparse` | 4 | sync_remove(2), mask_boundary(1), arith_replace(1) |
| `tail_heavy` | 3 | relop_replace(2), mask_boundary(1) |
| `near_zero` | 3 | epsilon_modify(3) |
| `uniform_constant` | 3 | arith_replace(2), sync_remove(1) |

> 14 个策略中 6 个杀伤为 0：`all_positive`, `alternating_sign`, `denormals`, `head_heavy`, `mixed_extremes`, `near_overflow`。

## 三、最终存活变异体 — 存活原因与变异体对应关系

共 **218** 个变异体最终存活，DeepSeek-R1 将其归为 **7** 个类别：

### 3.0 总览

| 类别 | 数量 | 占比 | 不可杀 | 理论可杀 | 主要算子 |
|------|------|------|--------|---------|---------|
| Kernel Design Resilience | 81 | 37.2% | 69 | 12 | mask_boundary(25), relop_replace(24), index_replace(17) |
| Ineffective Mutation | 53 | 24.3% | 51 | 2 | const_perturb(17), cast_remove(16), mask_boundary(5) |
| Out-of-Bounds Access Non-Observability | 29 | 13.3% | 15 | 14 | const_perturb(10), relop_replace(10), mask_boundary(5) |
| Algorithmic Invariance | 17 | 7.8% | 15 | 2 | index_replace(8), relop_replace(4), const_perturb(3) |
| Numerical Tolerance / Masking | 13 | 6.0% | 4 | 9 | const_perturb(11), relop_replace(1), mask_boundary(1) |
| Test Framework Limitations | 13 | 6.0% | 4 | 9 | const_perturb(6), mask_boundary(3), init_modify(1) |
| Race Condition / Non-Determinism Not Triggered | 12 | 5.5% | 7 | 5 | sync_remove(11), arith_replace(1) |

### 3.1 Kernel Design Resilience (81 个)

**典型存活机制**: The mutant survived because the grid-stride loop pattern in the kernel contains boundary checks (likely `if (idx < size) {...}`), making the extra block launched by the mutant compute zeros without accessing out-of-bounds memory. The block sums are then reduced to produce the same final result as the original. Previous attempts failed because they assumed incomplete boundary checks, but the kernel

- **mask_boundary** (25 个): `L1_P23__mask_boundary__0`, `L1_P34__mask_boundary__0`, `L1_P35__mask_boundary__0`, `L1_P35__mask_boundary__2`, `L1_P35__mask_boundary__3`, `L1_P47__mask_boundary__1`, `L1_P47__mask_boundary__2`, `L1_P50__mask_boundary__4`, `L1_P89__mask_boundary__0`, `L1_P89__mask_boundary__1`, `L1_P89__mask_boundary__3`, `L1_P90__mask_boundary__1`, `L1_P90__mask_boundary__2`, `L1_P90__mask_boundary__3`, `L1_P91__mask_boundary__1`, `L1_P91__mask_boundary__4`, `L1_P93__mask_boundary__0`, `L1_P93__mask_boundary__1`, `L1_P94__mask_boundary__0`, `L1_P95__mask_boundary__0`, `L1_P95__mask_boundary__1`, `L1_P97__mask_boundary__1`, `L1_P98__mask_boundary__0`, `L1_P98__mask_boundary__1`, `L1_P99__mask_boundary__0`
- **relop_replace** (24 个): `L1_P100__relop_replace__4`, `L1_P16__relop_replace__0`, `L1_P16__relop_replace__1`, `L1_P16__relop_replace__3`, `L1_P17__relop_replace__0`, `L1_P17__relop_replace__1`, `L1_P18__relop_replace__1`, `L1_P1__relop_replace__2`, `L1_P22__relop_replace__6`, `L1_P22__relop_replace__8`, `L1_P28__relop_replace__1`, `L1_P2__relop_replace__1`, `L1_P2__relop_replace__2`, `L1_P2__relop_replace__3`, `L1_P31__relop_replace__14`, `L1_P35__relop_replace__7`, `L1_P38__relop_replace__11`, `L1_P89__relop_replace__12`, `L1_P90__relop_replace__0`, `L1_P90__relop_replace__1`, `L1_P91__relop_replace__5`, `L1_P97__relop_replace__4`, `L1_P98__relop_replace__0`, `L1_P99__relop_replace__2`
- **index_replace** (17 个): `L1_P100__index_replace__0`, `L1_P18__index_replace__14`, `L1_P23__index_replace__6`, `L1_P28__index_replace__11`, `L1_P28__index_replace__6`, `L1_P31__index_replace__6`, `L1_P35__index_replace__35`, `L1_P38__index_replace__0`, `L1_P50__index_replace__10`, `L1_P50__index_replace__4`, `L1_P89__index_replace__1`, `L1_P90__index_replace__2`, `L1_P90__index_replace__3`, `L1_P91__index_replace__13`, `L1_P91__index_replace__2`, `L1_P91__index_replace__4`, `L1_P93__index_replace__3`
- **arith_replace** (10 个): `L1_P100__arith_replace__7`, `L1_P16__arith_replace__23`, `L1_P16__arith_replace__6`, `L1_P22__arith_replace__20`, `L1_P27__arith_replace__43`, `L1_P31__arith_replace__8`, `L1_P50__arith_replace__13`, `L1_P50__arith_replace__18`, `L1_P91__arith_replace__30`, `L1_P91__arith_replace__31`
- **launch_config_mutate** (3 个): `L1_P27__launch_config_mutate__2`, `L1_P28__launch_config_mutate__0`, `L1_P96__launch_config_mutate__0`
- **const_perturb** (2 个): `L1_P28__const_perturb__26`, `L1_P89__const_perturb__9`
- ⚠ 其中 **16 个**来自融合 kernel（含 fused/fusion 代码）

### 3.2 Ineffective Mutation (53 个)

**典型存活机制**: The mutant survives because the mutation only affects threads in the else branch of the A-tile loading. However, with the given input shapes (M=1024, K=4096, N=2048) and typical TILE_SIZE=32, K is a multiple of TILE_SIZE, so the column-out-of-bounds condition (t*TILE_SIZE + threadIdx.x) < K is always true. The else branch is only triggered for row-out-of-bounds threads (row >= M), but those thread

- **const_perturb** (17 个): `L1_P1__const_perturb__0`, `L1_P1__const_perturb__1`, `L1_P22__const_perturb__21`, `L1_P24__const_perturb__13`, `L1_P24__const_perturb__4`, `L1_P27__const_perturb__22`, `L1_P28__const_perturb__3`, `L1_P28__const_perturb__4`, `L1_P31__const_perturb__18`, `L1_P39__const_perturb__0`, `L1_P39__const_perturb__2`, `L1_P40__const_perturb__2`, `L1_P50__const_perturb__5`, `L1_P88__const_perturb__7`, `L1_P91__const_perturb__0`, `L1_P94__const_perturb__3`, `L1_P97__const_perturb__0`
- **cast_remove** (16 个): `L1_P23__cast_remove__0`, `L1_P23__cast_remove__1`, `L1_P23__cast_remove__2`, `L1_P24__cast_remove__0`, `L1_P34__cast_remove__2`, `L1_P34__cast_remove__3`, `L1_P34__cast_remove__4`, `L1_P35__cast_remove__0`, `L1_P38__cast_remove__0`, `L1_P38__cast_remove__1`, `L1_P38__cast_remove__3`, `L1_P39__cast_remove__0`, `L1_P39__cast_remove__1`, `L1_P40__cast_remove__0`, `L1_P98__cast_remove__0`, `L1_P98__cast_remove__1`
- **mask_boundary** (5 个): `L1_P22__mask_boundary__1`, `L1_P28__mask_boundary__0`, `L1_P31__mask_boundary__0`, `L1_P47__mask_boundary__5`, `L1_P97__mask_boundary__3`
- **sync_remove** (5 个): `L1_P23__sync_remove__0`, `L1_P23__sync_remove__3`, `L1_P34__sync_remove__7`, `L1_P89__sync_remove__4`, `L1_P97__sync_remove__2`
- **index_replace** (3 个): `L1_P17__index_replace__19`, `L1_P22__index_replace__10`, `L1_P88__index_replace__9`
- **arith_replace** (3 个): `L1_P27__arith_replace__2`, `L1_P88__arith_replace__27`, `L1_P88__arith_replace__33`
- **relop_replace** (2 个): `L1_P27__relop_replace__19`, `L1_P95__relop_replace__3`
- **scale_modify** (1 个): `L1_P33__scale_modify__0`
- **epsilon_modify** (1 个): `L1_P98__epsilon_modify__1`
- ⚠ 其中 **17 个**来自融合 kernel（含 fused/fusion 代码）

### 3.3 Out-of-Bounds Access Non-Observability (29 个)

**典型存活机制**: The mutant survives because the kernel launch configuration appears to use a fixed block size of 256, making the shared memory access always within bounds for both original (shared_mem[256]) and mutant (shared_mem[257]). Thread indices never reach 256 since blockDim.x=256, so the extra element in the mutant is never accessed. Even with different input sizes, the block size doesn't change to exploi

- **const_perturb** (10 个): `L1_P100__const_perturb__0`, `L1_P100__const_perturb__1`, `L1_P34__const_perturb__3`, `L1_P35__const_perturb__1`, `L1_P48__const_perturb__0`, `L1_P48__const_perturb__1`, `L1_P89__const_perturb__0`, `L1_P89__const_perturb__12`, `L1_P95__const_perturb__0`, `L1_P95__const_perturb__1`
- **relop_replace** (10 个): `L1_P18__relop_replace__2`, `L1_P28__relop_replace__2`, `L1_P28__relop_replace__3`, `L1_P34__relop_replace__1`, `L1_P34__relop_replace__9`, `L1_P47__relop_replace__5`, `L1_P89__relop_replace__16`, `L1_P89__relop_replace__2`, `L1_P95__relop_replace__5`, `L1_P98__relop_replace__4`
- **mask_boundary** (5 个): `L1_P19__mask_boundary__0`, `L1_P22__mask_boundary__0`, `L1_P23__mask_boundary__1`, `L1_P50__mask_boundary__0`, `L1_P99__mask_boundary__1`
- **arith_replace** (4 个): `L1_P17__arith_replace__7`, `L1_P29__arith_replace__10`, `L1_P89__arith_replace__10`, `L1_P89__arith_replace__11`
- ⚠ 其中 **10 个**来自融合 kernel（含 fused/fusion 代码）

### 3.4 Algorithmic Invariance (17 个)

**典型存活机制**: The mutant survives because the kernel computes a shift-invariant function (likely softmax or log-softmax). The maximum value is used as a shift for numerical stability, and the function's output is mathematically invariant to the choice of shift. When all row values are negative, the original uses the true (negative) maximum while the mutant uses 0, but due to shift-invariance, both produce ident

- **index_replace** (8 个): `L1_P24__index_replace__12`, `L1_P28__index_replace__10`, `L1_P31__index_replace__8`, `L1_P35__index_replace__39`, `L1_P38__index_replace__8`, `L1_P40__index_replace__12`, `L1_P89__index_replace__9`, `L1_P93__index_replace__2`
- **relop_replace** (4 个): `L1_P27__relop_replace__5`, `L1_P38__relop_replace__5`, `L1_P94__relop_replace__6`, `L1_P98__relop_replace__8`
- **const_perturb** (3 个): `L1_P50__const_perturb__0`, `L1_P97__const_perturb__33`, `L1_P98__const_perturb__11`
- **init_modify** (2 个): `L1_P23__init_modify__1`, `L1_P24__init_modify__1`
- ⚠ 其中 **7 个**来自融合 kernel（含 fused/fusion 代码）

### 3.5 Numerical Tolerance / Masking (13 个)

**典型存活机制**: The mutant survived previous attempts because: 1) Round 1's moderate inputs (1.0-1.5) produced differences too small to exceed the test's tolerance. 2) Round 2's large inputs caused both original and mutant approximations to diverge (outputs outside [-1,1]), making both fail the test's validation, so the mutant wasn't killed. The mutation changes a numerator constant by ~1%, which has minimal impa

- **const_perturb** (11 个): `L1_P22__const_perturb__0`, `L1_P22__const_perturb__6`, `L1_P29__const_perturb__6`, `L1_P29__const_perturb__8`, `L1_P34__const_perturb__13`, `L1_P40__const_perturb__5`, `L1_P88__const_perturb__0`, `L1_P88__const_perturb__11`, `L1_P96__const_perturb__1`, `L1_P96__const_perturb__2`, `L1_P97__const_perturb__35`
- **relop_replace** (1 个): `L1_P29__relop_replace__9`
- **mask_boundary** (1 个): `L1_P34__mask_boundary__1`
- ⚠ 其中 **4 个**来自融合 kernel（含 fused/fusion 代码）

### 3.6 Test Framework Limitations (13 个)

**典型存活机制**: The mutant survives because previous inputs failed to satisfy the tool's killing criteria: both original and mutant produced outputs that mismatched the reference (original_ok=False, mutant_ok=False). This indicates the inputs triggered numerical errors or implementation bugs that affected both versions equally, preventing isolation of the block-size difference. The kernel likely has a baseline co

- **const_perturb** (6 个): `L1_P24__const_perturb__9`, `L1_P27__const_perturb__35`, `L1_P31__const_perturb__2`, `L1_P35__const_perturb__4`, `L1_P50__const_perturb__3`, `L1_P98__const_perturb__10`
- **mask_boundary** (3 个): `L1_P88__mask_boundary__1`, `L1_P96__mask_boundary__0`, `L1_P96__mask_boundary__2`
- **init_modify** (1 个): `L1_P24__init_modify__0`
- **relop_replace** (1 个): `L1_P34__relop_replace__6`
- **scale_modify** (1 个): `L1_P34__scale_modify__0`
- **arith_replace** (1 个): `L1_P50__arith_replace__16`
- ⚠ 其中 **3 个**来自融合 kernel（含 fused/fusion 代码）

### 3.7 Race Condition / Non-Determinism Not Triggered (12 个)

**典型存活机制**: The mutant removes a critical __syncthreads() barrier that ensures all threads have written to shared memory before the reduction begins. Without it, the first reduction iteration may read uninitialized or stale values from slower threads. However, previous attempts failed because the kernel's execution is uniform (no divergence) and warp scheduling likely ensures writes happen before reads in pra

- **sync_remove** (11 个): `L1_P100__sync_remove__0`, `L1_P100__sync_remove__1`, `L1_P23__sync_remove__1`, `L1_P34__sync_remove__1`, `L1_P34__sync_remove__3`, `L1_P40__sync_remove__2`, `L1_P50__sync_remove__0`, `L1_P89__sync_remove__1`, `L1_P94__sync_remove__2`, `L1_P95__sync_remove__0`, `L1_P95__sync_remove__1`
- **arith_replace** (1 个): `L1_P38__arith_replace__17`
- ⚠ 其中 **9 个**来自融合 kernel（含 fused/fusion 代码）

## 四、算子融合（Operator Fusion）专项分析

初始 322 个存活变异体中，**99 个**来自含 fused/fusion 关键词的 kernel（占 30.7%）。

| 阶段 | 杀死 | 存活 |
|------|------|------|
| Phase 1 增强测试 | 25 | 74 |
| Phase 2 LLM 分析 | 8 | 66 |

### 4.1 最终存活的融合 kernel 变异体

按存活原因:

| 存活原因 | 数量 |
|---------|------|
| Ineffective Mutation | 17 |
| Kernel Design Resilience | 16 |
| Out-of-Bounds Access Non-Observability | 10 |
| Race Condition / Non-Determinism Not Triggered | 9 |
| Algorithmic Invariance | 7 |
| Numerical Tolerance / Masking | 4 |
| Test Framework Limitations | 3 |

按 kernel:

- **L1_P100** (7 个): 算子 {'arith_replace': 1, 'index_replace': 1, 'relop_replace': 1, 'const_perturb': 2, 'sync_remove': 2}，原因 {'Kernel Design Resilience': 3, 'Out-of-Bounds Access Non-Observability': 2, 'Race Condition / Non-Determinism Not Triggered': 2}
- **L1_P34** (14 个): 算子 {'mask_boundary': 2, 'const_perturb': 2, 'relop_replace': 3, 'sync_remove': 3, 'cast_remove': 3, 'scale_modify': 1}，原因 {'Kernel Design Resilience': 1, 'Out-of-Bounds Access Non-Observability': 3, 'Race Condition / Non-Determinism Not Triggered': 2, 'Ineffective Mutation': 4, 'Numerical Tolerance / Masking': 2, 'Test Framework Limitations': 2}
- **L1_P38** (8 个): 算子 {'index_replace': 2, 'relop_replace': 2, 'arith_replace': 1, 'cast_remove': 3}，原因 {'Kernel Design Resilience': 2, 'Race Condition / Non-Determinism Not Triggered': 1, 'Ineffective Mutation': 3, 'Algorithmic Invariance': 2}
- **L1_P40** (5 个): 算子 {'sync_remove': 1, 'cast_remove': 1, 'const_perturb': 2, 'index_replace': 1}，原因 {'Race Condition / Non-Determinism Not Triggered': 1, 'Ineffective Mutation': 2, 'Numerical Tolerance / Masking': 1, 'Algorithmic Invariance': 1}
- **L1_P94** (4 个): 算子 {'mask_boundary': 1, 'sync_remove': 1, 'const_perturb': 1, 'relop_replace': 1}，原因 {'Kernel Design Resilience': 1, 'Race Condition / Non-Determinism Not Triggered': 1, 'Ineffective Mutation': 1, 'Algorithmic Invariance': 1}
- **L1_P95** (8 个): 算子 {'mask_boundary': 2, 'const_perturb': 2, 'relop_replace': 2, 'sync_remove': 2}，原因 {'Kernel Design Resilience': 2, 'Out-of-Bounds Access Non-Observability': 3, 'Race Condition / Non-Determinism Not Triggered': 2, 'Ineffective Mutation': 1}
- **L1_P97** (7 个): 算子 {'mask_boundary': 2, 'relop_replace': 1, 'const_perturb': 3, 'sync_remove': 1}，原因 {'Kernel Design Resilience': 2, 'Ineffective Mutation': 3, 'Numerical Tolerance / Masking': 1, 'Algorithmic Invariance': 1}
- **L1_P98** (10 个): 算子 {'mask_boundary': 2, 'relop_replace': 3, 'cast_remove': 2, 'epsilon_modify': 1, 'const_perturb': 2}，原因 {'Kernel Design Resilience': 3, 'Out-of-Bounds Access Non-Observability': 1, 'Ineffective Mutation': 3, 'Algorithmic Invariance': 2, 'Test Framework Limitations': 1}
- **L1_P99** (3 个): 算子 {'mask_boundary': 2, 'relop_replace': 1}，原因 {'Kernel Design Resilience': 2, 'Out-of-Bounds Access Non-Observability': 1}

### 4.2 算子融合如何掩盖变异？

融合 kernel 将多个 PyTorch 操作合并到一个 CUDA kernel 中，中间结果不经过 global memory。
这导致以下掩盖机制：

1. **中间值截断吸收**: 融合链 `add → relu → mul` 中，add 的微扰可能被 relu 截断（负值→0）
2. **精度差异内部消化**: cast_remove 导致的精度差在融合链传播时，被后续操作的精度要求重新对齐
3. **边界检查冗余**: 融合 kernel 通常有统一的边界检查，mask_boundary 变异被冗余检查抵消

### 4.3 能否杀死融合 kernel 的存活变异体？

策略方向（不改 shape）：

| 策略 | 原理 | 目标算子 | 可行性 |
|------|------|---------|--------|
| `near_relu_zero` | 值域集中在 relu/激活函数的零点附近 [-0.01, 0.01] | const_perturb, arith_replace | 中 |
| `saturation_boundary` | 值域在 sigmoid/tanh 饱和区边界 (±4~±6) | epsilon_modify, const_perturb | 中 |
| `anti_cancellation` | 构造不对称输入使中间值不会被后续操作抵消 | arith_replace, cast_remove | 低 |

**核心困难**: 融合 kernel 的中间计算对外不可观测（封装在 kernel 内部），无法精确控制中间值。
上述策略只能间接影响中间值，效果取决于具体融合结构。

## 五、未覆盖的规则与盲区

LLM 判定结果:
- **不可杀死** (killable=False): 165 个 — 在 fixed-shape 约束下语义等价
- **理论可杀但未杀** (killable=True): 53 个 — **潜在规则盲区**

### 5.1 "理论可杀但未杀"的 53 个变异体

**Out-of-Bounds Access Non-Observability** (14 个): {'arith_replace': 3, 'mask_boundary': 3, 'relop_replace': 4, 'const_perturb': 4}
  - `L1_P17__arith_replace__7`: Previous attempts failed likely because the mutant's out-of-bounds memory access read uncontrolled or zero values, causing no observable difference in...
  - `L1_P19__mask_boundary__0`: The mutant changes the boundary condition from 'idx < size' to 'idx < size - 1', causing the last element (index size-1) to be skipped and remain unin...
  - ...还有 12 个

**Kernel Design Resilience** (12 个): {'relop_replace': 3, 'mask_boundary': 6, 'arith_replace': 1, 'const_perturb': 1, 'index_replace': 1}
  - `L1_P17__relop_replace__0`: The mutation changes the loop condition from `<` to `<=`, adding an extra iteration when `t` equals `(K + TILE_SIZE - 1) / TILE_SIZE`. However, in thi...
  - `L1_P23__mask_boundary__0`: The mutant changes the boundary guard from `batch_idx >= batch_size` to `batch_idx > batch_size`. This only affects blocks where `batch_idx == batch_s...
  - ...还有 10 个

**Numerical Tolerance / Masking** (9 个): {'const_perturb': 7, 'relop_replace': 1, 'mask_boundary': 1}
  - `L1_P22__const_perturb__0`: The mutant survived previous attempts because: 1) Round 1's moderate inputs (1.0-1.5) produced differences too small to exceed the test's tolerance. 2...
  - `L1_P22__const_perturb__6`: The mutant survived because the constant perturbation (135135.0 → 133783.65) is a small relative change (~1%) that may not produce detectable output d...
  - ...还有 7 个

**Test Framework Limitations** (9 个): {'const_perturb': 4, 'init_modify': 1, 'scale_modify': 1, 'arith_replace': 1, 'mask_boundary': 2}
  - `L1_P24__const_perturb__9`: The mutant survives because previous inputs failed to satisfy the tool's killing criteria: both original and mutant produced outputs that mismatched t...
  - `L1_P24__init_modify__0`: The previous attempts failed because they triggered numerical overflow in the mutant, causing infinite values in the output. However, the original ker...
  - ...还有 7 个

**Race Condition / Non-Determinism Not Triggered** (5 个): {'sync_remove': 4, 'arith_replace': 1}
  - `L1_P100__sync_remove__0`: The mutant removes a critical __syncthreads() barrier that ensures all threads have written to shared memory before the reduction begins. Without it, ...
  - `L1_P34__sync_remove__1`: The mutant survives because the removed __syncthreads() only causes race conditions during the tree reduction phase (lines 46-50). Previous attempts f...
  - ...还有 3 个

**Ineffective Mutation** (2 个): {'arith_replace': 1, 'epsilon_modify': 1}
  - `L1_P27__arith_replace__2`: The mutant likely survives because the mutation only affects the generic template, but the kernel may be instantiated with the `half` specialization (...
  - `L1_P98__epsilon_modify__1`: The mutant survives because both original and mutant produce identical outputs for the tested inputs. This suggests the fused CUDA kernel has internal...

**Algorithmic Invariance** (2 个): {'const_perturb': 2}
  - `L1_P50__const_perturb__0`: The previous attempts failed because they didn't account for how the kernel organizes threads into warps and reduces across the entire reduction dimen...
  - `L1_P98__const_perturb__11`: The mutant survives because the normalization step after clamping cancels out the effect of the changed clamp value when rows contain only uniform val...

### 5.2 盲区分类与对策

| 盲区类型 | 对应存活类别 | 变异体数 | 当前策略的局限 | 可能的对策 |
|---------|-----------|---------|-------------|----------|
| 值域不够极端 | Numerical Tolerance | 9 | 值虽极端但差异仍在 tolerance 内 | `epsilon_critical_norm`: 精确控制范数使 epsilon 成为主因 |
| OOB 读到零值 | OOB Non-Observability | 14 | 无法控制 OOB 内存内容 | 理论上需要内存布局控制，超出 value-only 范围 |
| grid-stride 吸收 | Kernel Design Resilience | 12 | grid-stride loop 使 block 数无关紧要 | 在 fixed-shape 下无对策（属于设计等价） |
| 测试框架局限 | Test Framework Limitations | 9 | original 和 mutant 都失败 | 需更宽松的 tolerance 策略或分段测试 |
| 竞态未触发 | Race Condition | 5 | 10 次重复不够 | 增加重复次数 / 更大 block_size |
| 算法不变性 | Algorithmic Invariance | 2 | 算法本身对变异不敏感 | 属于真等价变异体 |
| 无效变异 | Ineffective Mutation | 2 | 变异不改变执行路径 | 属于真等价变异体 |

### 5.3 当前可实施的新增策略（不改 shape）

| # | 策略名 | 构造方式 | 目标算子 | 预期增量 | 来源 |
|---|--------|---------|---------|---------|------|
| 1 | `epsilon_critical_norm` | L2 范数 ∈ [1e-9, 1e-2] | epsilon_modify, const_perturb | ~5-10 | LLM rule |
| 2 | `reduction_identity_extreme` | 整行 < -1e10 | init_modify | ~1-3 | LLM rule |
| 3 | `near_activation_zero` | 50% 值 ∈ [-0.01, 0.01] | 融合 kernel 各算子 | ~2-5 | 导师建议 |
| 4 | `block_sparse` | 连续块为零 + 其余大值 | mask_boundary | ~1-3 | 导师建议 |

> **注**: 以上估计基于 LLM 的 killable 判定和存活原因分析。实际效果需重新运行实验验证。