# 第三次实验：外部 CUDA Kernel 差分测试方案

> **核心目标**：将 MutaKernel 两阶段方法中提炼出的 21 种 stress 输入策略应用于外部开源 CUDA C++ kernel 仓库，通过差分测试检测这些 kernel 相对于 PyTorch 参考实现的数值偏差。
>
> **数据来源追溯**：本文档严格标注每种输入策略的设计来源（变异算子 → 策略设计 → 差分测试应用），确保方法论的可追溯性。

---

## 一、实验总体架构

### 1.1 方法论来源

本实验的输入策略**不是凭空设计的**，而是从 MutaKernel V2.0 的两阶段实验中**自底向上提炼**出来的：

```
第一层：15 种 CUDA 变异算子（见 §二）
   ↓  "什么样的故障需要什么样的输入来暴露？"
第二层：21 种 stress 输入策略（见 §三）
   ↓  "这些策略在变异测试中有效，能否迁移到真实 kernel 质量检测？"
第三层：本实验——外部 CUDA kernel 差分测试（见 §四）
```

### 1.2 与前两次实验的关系

| 实验 | 被测对象 | 比较方式 | 输入策略 |
|------|---------|---------|---------|
| 第一次（Phase 1） | KernelBench kernel 的变异体 | mutant vs original（bitwise） | OPERATOR_DIRECTED_POLICIES（算子定向，12 轮） |
| 第二次（Phase 2） | KernelBench kernel 的存活变异体 | mutant vs original（allclose） | 21 种 STRESS_POLICIES + 6 个测试维度 |
| **第三次（本实验）** | **外部开源 CUDA kernel** | **外部 kernel vs PyTorch（allclose）** | **复用第二次的 21 种 STRESS_POLICIES + 5 个确定性测试维度** |

### 1.3 实验流水线

```
对每个外部 CUDA kernel:
  1. 加载 PyTorch 参考实现 (Model + get_inputs)
  2. 加载外部 CUDA kernel 包装 (ModelNew)
  3. 维度 1 — 基线测试: 50 个随机种子, 标准 randn 输入
  4. 维度 2 — value_stress: 21 种策略 × 3 seeds = 63 轮
  5. 维度 3 — dtype_stress: float16 / bfloat16 精度切换
  6. 维度 4 — training_stress: .train() 模式下 21 种策略
  7. 维度 5 — repeated_run: 同一输入执行 10 次, 检测非确定性
  8. 维度 6 — config_stress: 7 种 batch_size 变化
  9. 多容差后处理: 对发现偏差的维度, 用 5 级容差重新判定
  10. 输出: 每个 kernel 的多维度偏差报告
```

### 1.4 六维度测试设计与追溯

> **来源**: `StressEnhance_Plan.md` §6 Worker 的 6 种测试模式 + §8 执行顺序

六个测试维度的设计来自第二次实验的增强测试框架。每个维度覆盖不同的 GPU kernel 故障触发路径：

| 维度 | 来源（StressEnhance_Plan.md） | 检测目标 | 追溯到的 GPU 特有问题 | 本实验中的适配 |
|------|---------------------------|---------|-------------------|----|
| **value_stress** | §6.1 | 数值边界、极端分布 | 浮点精度、overflow/underflow | 直接复用 21 种策略 |
| **dtype_stress** | §6.2 | 低精度类型行为 | 精度退化、类型转换 bug | 直接复用 float16/bfloat16 |
| **training_stress** | §6.4 | .train() 模式行为 | BN/LN 运行统计 vs 训练统计差异 | 直接复用（对 Norm 类 kernel 最有价值） |
| **repeated_run** | §6.3 | 非确定性 / data race | CUDA kernel 内部竞态条件 | 直接复用（10 次重复） |
| **config_stress** | §6.5 | 非标准配置行为 | grid/block 计算边界、shape 边界 | 直接复用 7 种 batch_size |
| **LLM 分析** | §10 | 需语义推理的深层问题 | 综合源码语义理解 | **不适用**（无变异上下文，改为偏差原因分析） |

**为什么 LLM 迭代分析不直接复用**：原 Phase 2 的 LLM 分析基于变异体代码 diff 进行推理（"为什么这个变异没被杀死？"）。本实验中没有变异代码，只有原始 CUDA kernel，因此不适用原始 LLM 杀伤协议。但在实验报告阶段，可以用 LLM 分析偏差原因。

---

## 二、第一层追溯：15 种 CUDA 变异算子

> **来源文件**: `EquivalentDetection_V2_Plan.md` §一 + `src/mutengine/operators/*.py`

21 种 stress 策略的设计根源是 MutaKernel 定义的 15 种 CUDA 变异算子。每种算子模拟了 CUDA kernel 中一类常见的编程错误：

| 编号 | 变异算子 | 类别 | 模拟的故障类型 | 代码中的变异示例 |
|------|---------|------|-------------|----------------|
| 1 | `arith_replace` | A-算术 | 算术运算符错误 | `a + b` → `a - b` |
| 2 | `relop_replace` | A-算术 | 关系运算符错误 | `x <= N` → `x < N` |
| 3 | `const_perturb` | A-算术 | 常量值错误 | `eps = 1e-5` → `eps = 1e-4` |
| 4 | `index_replace` | B-GPU并行 | 线程索引错误 | `blockIdx.x` → `blockIdx.y` |
| 5 | `sync_remove` | B-GPU并行 | 同步缺失 | `__syncthreads()` → 删除 |
| 6 | `mask_boundary` | B-GPU并行 | 边界条件错误 | `idx < size` → `idx < size-1` |
| 7 | `launch_config_mutate` | B-GPU并行 | 启动配置错误 | `blockDim(256)` → `blockDim(128)` |
| 8 | `stab_remove` | C-ML语义 | 数值稳定化移除 | `log(softmax(x))` → `log(exp(x)/sum(exp(x)))` |
| 9 | `acc_downgrade` | C-ML语义 | 累加精度降级 | `float64 累加` → `float32 累加` |
| 10 | `epsilon_modify` | C-ML语义 | epsilon 值变更 | `eps=1e-5` → `eps=1e-6` |
| 11 | `scale_modify` | C-ML语义 | 缩放因子变更 | `1/sqrt(d)` → `1/d` |
| 12 | `cast_remove` | C-ML语义 | 类型转换移除 | `.to(float32)` → 删除 |
| 13 | `reduction_reorder` | C-ML语义 | 归约顺序变更 | 行归约 → 列归约 |
| 14 | `init_modify` | C-ML语义 | 初始值变更 | `max_val = -inf` → `max_val = 0` |
| 15 | `broadcast_unsafe` | D-LLM模式 | 广播安全性移除 | 安全广播 → 不安全广播 |

---

## 三、第二层追溯：21 种 Stress 输入策略

> **来源文件**: `StressEnhance_Plan.md` §9 + `src/stress/policy_bank.py`

每种策略的设计都源自对特定变异算子的"state infection condition"分析——即**什么样的输入值能让变异引入的差异被放大到可观测的程度**。

### 3.1 完整策略清单与追溯

#### 第一组：14 种通用策略（V1，来自第一次实验分析）

| # | 策略名 | 数值生成方式 | 设计动机（源自哪类变异算子） | 追溯到的 state infection condition |
|---|--------|-----------|-------------------------|--------------------------------|
| 1 | `large_magnitude` | `randn × 1000` | `arith_replace`: `a+b` vs `a-b`，大值放大差异 | 算术操作的输出差异与操作数量级成正比 |
| 2 | `near_overflow` | `randn × dtype 上界` | `cast_remove`, `stab_remove`: 移除类型转换或数值稳定化后溢出 | 浮点溢出是数值稳定化保护的直接目标 |
| 3 | `near_zero` | `randn × 1e-7` | `epsilon_modify`: `eps=1e-5` → `1e-6`，接近零时 epsilon 分支关键 | 除法 `x/(y+eps)` 中 eps 的绝对值影响随 y→0 指数增长 |
| 4 | `denormals` | `randn × 1e-38` | `epsilon_modify`, `cast_remove`: 非正规数触发特殊浮点行为 | IEEE 754 非正规数在不同精度下的 flush-to-zero 行为差异 |
| 5 | `all_negative` | `-|randn| × 100` | `init_modify`: `max_val=0` → `max_val=1`，全负输入让 max 初始值偏差暴露 | max/min 归约中初始值仅在输入全同号时成为边界条件 |
| 6 | `all_positive` | `|randn| × 100` | `init_modify`: 全正值让 min 初始值偏差暴露 | 同上（对称） |
| 7 | `mixed_extremes` | `50% × 10000 + 50% × 0.0001` | `acc_downgrade`: 极端混合值最大化浮点累加误差 | 大小值混合导致浮点加法的灾难性抵消 |
| 8 | `alternating_sign` | 交替 `+/- × 100` | `reduction_reorder`: 交替正负值让不同归约顺序产生不同累积误差 | 浮点加法的非结合性在交替正负值时误差最大 |
| 9 | `sparse` | `90% 零 + 10% randn×100` | `mask_boundary`: 稀疏输入让边界元素处理成为关键路径 | 零值掩码下，边界条件 `idx < size` vs `idx < size-1` 的差异仅在非零元素处可见 |
| 10 | `uniform_constant` | 全部 `88.0` | `scale_modify`: 均匀常量让缩放因子差异在所有元素上一致可见 | `x * (1/√d)` vs `x * (1/d)` 的差异在 x 恒定时最清晰 |
| 11 | `structured_ramp` | `[0, 1/n, 2/n, ...]` | `index_replace`: 线性递增值让索引偏移直接可观测 | 当 `output[i] = input[f(i)]` 时，f 的偏移在单调输入上差异最大 |
| 12 | `boundary_last_element` | `randn + 末位=1e4` | `mask_boundary`, `relop_replace`: 末位极端值暴露 off-by-one | `idx < size` vs `idx < size-1` 恰好在最后一个元素产生差异 |
| 13 | `head_heavy` | 前25%极端值 + 其余近零 | `index_replace`: 索引错误导致只处理头部 → 聚合结果偏差 | `blockIdx.x` → `blockIdx.y` 可能导致所有 block 仅处理前 N/gridDim 个元素 |
| 14 | `tail_heavy` | 后25%极端值 + 其余近零 | `index_replace`: 与 `head_heavy` 互补，检测跳过尾部的索引错误 | 同上（对称） |

#### 第二组：5 种算子定向策略（V2，来自第一次实验中等价检测的分析）

> **来源**: `EquivalentDetection_V2_Plan.md` §5.1

| # | 策略名 | 数值生成方式 | 设计动机（目标变异算子） | 追溯到的 state infection condition |
|---|--------|-----------|-------------------------|--------------------------------|
| 15 | `relop_boundary_hit` | `arange % 10`（整数值） | `relop_replace`: `<=N` vs `<N` 在 `x=N` 时分歧 | 关系运算符差异仅在操作数**恰好等于**比较阈值时触发 |
| 16 | `extreme_magnitude` | `randn × 1e6` | `arith_replace`, `scale_modify`: 比 large_magnitude 更极端 | 算术差异 `|a+b - (a-b)| = 2|b|`，|b| 越大差异越大 |
| 17 | `near_epsilon` | `[1e-7, 1e-5] 均匀` | `epsilon_modify`: 输入恰在 epsilon 量级时，epsilon 改变直接影响结果 | `x/(x+eps)` 中当 `x ≈ eps` 时，eps 的微小变化导致结果 ±50% |
| 18 | `reduction_adversarial` | 交替 `+1e4/-1e4` | `acc_downgrade`, `reduction_reorder`: 最大化 FP 归约误差 | 有限精度下 `Σ(+1e4, -1e4, +1e4, ...)` 的累积误差随精度降级指数增长 |
| 19 | `init_sensitive` | 随机全正或全负 | `init_modify`: 让 min/max 初始值成为结果的决定性因素 | `reduce_max(全负数)` 中 `init=0` vs `init=-inf` 直接决定输出 |

#### 第三组：2 种稀疏梯度策略（V2.1，来自第二次实验的数据分析）

> **来源**: `StressEnhance_Plan.md` §9 稀疏性三档梯度

| # | 策略名 | 数值生成方式 | 设计动机 | 追溯到的 state infection condition |
|---|--------|-----------|---------|--------------------------------|
| 20 | `dense_nonzero` | `|randn| + 1.0` | `arith_replace`, `epsilon_modify`: 消除零值掩盖差异 | 当 `x=0` 时 `x+b = x-b = 0`，零值让算术变异不可见 |
| 21 | `sparse_extreme` | `99% 零 + 1% randn×1e4` | `mask_boundary`, `relop_replace`: 极端稀疏下的边界传播 | 近乎全零输入中的极少数极端值，若恰好落在边界位置则差异被放大 |

### 3.2 策略在两阶段中的使用方式

| 使用场景 | 使用的策略 | 比较标准 | 数据来源 |
|---------|----------|---------|---------|
| Phase 1 EMD Layer 2（等价检测） | `OPERATOR_DIRECTED_POLICIES`（每算子 3-4 种定向策略） | `bitwise_identical` | `EquivalentDetection_V2_Plan.md` §5.2 |
| Phase 2 value_stress（增强测试） | 全部 21 种（去重 Layer 2 已测策略） | `allclose(atol=1e-2)` | `StressEnhance_Plan.md` §6.1 |
| Phase 2 training_stress（训练模式） | 全部 21 种（不去重，`.train()` 模式） | `allclose(atol=1e-2)` | `StressEnhance_Plan.md` §6.4 |
| **第三次实验（本实验）** | **全部 21 种（无去重，无变异上下文）** | **`allclose(多级容差)`** | **本文档** |

---

## 四、第三层：本实验的输入与输出定义

### 4.1 实验输入（精确定义）

对每个外部 CUDA kernel，实验输入由以下要素组成：

#### 4.1.1 参考实现（Model）

| 字段 | 说明 | 来源 |
|------|------|------|
| `Model` 类 | PyTorch 原生实现 (nn.Module) | 人工编写，保证语义等价 |
| `get_inputs()` | 返回固定 shape 的随机 tensor 列表 | 人工编写，shape 匹配外部 kernel 接口 |
| `get_init_inputs()` | 返回模型构造参数 | 人工编写 |

#### 4.1.2 被测 kernel（ModelNew）

| 字段 | 说明 | 来源 |
|------|------|------|
| `ModelNew` 类 | 包装外部 CUDA kernel 的 nn.Module | 人工编写，调用外部库 API |
| 导入路径 | `from apex.normalization import FusedLayerNorm` 等 | 外部库的 Python binding |

#### 4.1.3 测试输入的生成

每一轮测试的输入生成过程：

```python
# Step 1: 用种子生成模板输入（shape/dtype 由 get_inputs 决定）
torch.manual_seed(seed)
template_inputs = get_inputs()  # e.g., [torch.randn(32, 128, 1024)]

# Step 2: 用 stress 策略转换数值（shape/dtype 不变）
stress_inputs = STRESS_POLICIES[policy_name](template_inputs, seed)
# e.g., near_zero: 每个浮点元素变为 randn * 1e-7

# Step 3: 移动到 GPU
inputs_on_gpu = [x.to("cuda") if isinstance(x, torch.Tensor) else x
                 for x in stress_inputs]
```

**关键约束**：所有 21 种策略只改变数值，不改变 shape/dtype。这保证了与 `get_inputs()` 接口的兼容性。

来源追溯：
- 模板生成逻辑 → `_stress_worker.py` 第 77-78 行
- 策略转换逻辑 → `policy_bank.py` 第 33-45 行（`_make_policy` 包装器）
- shape 不变约束 → `StressEnhance_Plan.md` §9 头部说明："全部保持 shape/dtype 不变"

#### 4.1.4 每个外部 kernel 的完整输入矩阵（六维度）

**维度 1: 基线测试**

| 测试阶段 | 描述 | 轮次 | 来源 |
|---------|------|------|------|
| baseline | `__identity__` policy（标准 randn），50 seeds | 50 | 基础正确性验证 |

**维度 2: value_stress（数值分布压力）**

> 来源：`StressEnhance_Plan.md` §6.1 + `policy_bank.py`

| 策略名 | seeds | 轮次 | 追溯的变异算子 |
|--------|-------|------|-------------|
| `large_magnitude` | 50000-50002 | 3 | arith_replace |
| `near_overflow` | 50003-50005 | 3 | cast_remove, stab_remove |
| `near_zero` | 50006-50008 | 3 | epsilon_modify |
| `denormals` | 50009-50011 | 3 | epsilon_modify, cast_remove |
| `all_negative` | 50012-50014 | 3 | init_modify |
| `all_positive` | 50015-50017 | 3 | init_modify |
| `mixed_extremes` | 50018-50020 | 3 | acc_downgrade |
| `alternating_sign` | 50021-50023 | 3 | reduction_reorder |
| `sparse` | 50024-50026 | 3 | mask_boundary |
| `uniform_constant` | 50027-50029 | 3 | scale_modify |
| `structured_ramp` | 50030-50032 | 3 | index_replace |
| `boundary_last_element` | 50033-50035 | 3 | mask_boundary, relop_replace |
| `head_heavy` | 50036-50038 | 3 | index_replace |
| `tail_heavy` | 50039-50041 | 3 | index_replace |
| `relop_boundary_hit` | 50042-50044 | 3 | relop_replace |
| `extreme_magnitude` | 50045-50047 | 3 | arith_replace, scale_modify |
| `near_epsilon` | 50048-50050 | 3 | epsilon_modify |
| `reduction_adversarial` | 50051-50053 | 3 | acc_downgrade, reduction_reorder |
| `init_sensitive` | 50054-50056 | 3 | init_modify |
| `dense_nonzero` | 50057-50059 | 3 | arith_replace, epsilon_modify |
| `sparse_extreme` | 50060-50062 | 3 | mask_boundary, relop_replace |
| **小计** | | **63** | |

**维度 3: dtype_stress（精度切换测试）**

> 来源：`StressEnhance_Plan.md` §6.2
> 追溯变异算子：`cast_remove`（移除类型转换）、`acc_downgrade`（精度降级）

| 目标精度 | seeds | 轮次 | 检测目标 |
|---------|-------|------|---------|
| `float16` | 60000-60002 | 3 | FP16 下的精度退化、overflow |
| `bfloat16` | 60003-60005 | 3 | BF16 下的动态范围保持、精度损失 |
| **小计** | | **6** | |

执行方式（来源：`_stress_worker.py` `run_dtype_stress`）：
- 将输入 tensor 和模型参数同时 cast 到目标精度
- 用 allclose 比较 ref_model(fp16_input) vs external_model(fp16_input)
- 检测外部 kernel 在低精度下是否有异常行为（如 overflow、NaN）

**维度 4: training_stress（训练模式测试）**

> 来源：`StressEnhance_Plan.md` §6.4
> 追溯变异算子：`epsilon_modify`、`scale_modify`、`init_modify`、`stab_remove`、`acc_downgrade`、`cast_remove`、`reduction_reorder`（ML 数值语义类算子）

| 描述 | 策略 | seeds | 轮次 | 检测目标 |
|------|------|-------|------|---------|
| .train() 模式 × 21 策略 | 全部 21 种 | 70000+ | 63 | LayerNorm/RMSNorm 的 running_mean/running_var 差异 |
| **小计** | | | **63** | |

执行方式（来源：`_stress_worker.py` `run_training_stress`）：
- 三个模型（ref/external）均使用 `.train()` 而非 `.eval()`
- BatchNorm、LayerNorm 在 .train() 模式下使用 batch 统计而非 running 统计
- 某些 CUDA 优化 kernel 在 .train() 模式下的数值路径不同

**维度 5: repeated_run（非确定性检测）**

> 来源：`StressEnhance_Plan.md` §6.3
> 追溯变异算子：`sync_remove`（移除 __syncthreads 导致竞态）

| 描述 | seeds | 重复次数 | 轮次 | 检测目标 |
|------|-------|---------|------|---------|
| 同一输入，执行外部 kernel 10 次 | 80000-80002 | 10 | 3 | CUDA kernel 内部竞态条件、非确定性结果 |
| **小计** | | | **3** | |

执行方式（来源：`_stress_worker.py` `run_repeated`）：
- 固定输入，对外部 kernel 运行 10 次
- 检查任意一次是否与 ref 不一致（allclose/NaN）
- 检查 kernel 自身输出跨次是否不一致（self-inconsistency, atol=1e-6）
- 真实 CUDA kernel（如 FlashAttention）可能因算法设计有非确定性

**维度 6: config_stress（配置变化测试）**

> 来源：`StressEnhance_Plan.md` §6.5
> 追溯变异算子：`launch_config_mutate`（启动配置变更）、`mask_boundary`（边界条件变更）

| batch_size | seeds | 轮次 | 检测目标 |
|-----------|-------|------|---------|
| 1 | 90000-90002 | 3 | 退化情况（单样本） |
| 2 | 90003-90005 | 3 | 极小 batch |
| 4 | 90006-90008 | 3 | 非标准 batch |
| 8 | 90009-90011 | 3 | 非标准 batch |
| 16 | 90012-90014 | 3 | 中等 batch |
| 32 | 90015-90017 | 3 | 原始 batch（对照） |
| 64 | 90018-90020 | 3 | 大 batch |
| **小计** | | **21** | |

执行方式（来源：`_stress_worker.py` `run_config_stress`）：
- 用 `_rebatch_inputs()` 将 `get_inputs()` 的第一维截断/重复到目标 batch_size
- 在新 batch_size 下执行 ref vs external 比较
- 检测 CUDA kernel 对不同 shape 的健壮性（grid/block 边界计算）

**总计**

| 维度 | 轮次/kernel |
|------|-----------|
| 基线 | 50 |
| value_stress | 63 |
| dtype_stress | 6 |
| training_stress | 63 |
| repeated_run | 3 |
| config_stress | 21 |
| **合计** | **206** |
| 多容差后处理（仅偏差策略） | ≤ 105 |
| **最大总计** | **≤ 311 轮/kernel** |

### 4.2 被测目标 Kernel

> 原计划 6 个 kernel，因 `apex.transformer` 模块在当前 Apex 构建中不可用（需要 Megatron 特定编译标志），
> FusedSoftmax 已被移除。最终测试 **5 个 kernel**。

| 编号 | kernel ID | 仓库 | 外部 API | PyTorch 参考 | CUDA 源文件 |
|------|----------|------|---------|-------------|------------|
| K1 | `apex__fused_layer_norm` | NVIDIA/apex | `FusedLayerNorm` | `nn.LayerNorm` | `layer_norm_cuda_kernel.cu` |
| K2 | `apex__fused_rms_norm` | NVIDIA/apex | `FusedRMSNorm` | 手写 RMSNorm | `layer_norm_cuda_kernel.cu` |
| K3 | `apex__fused_dense` | NVIDIA/apex | `FusedDense` | `nn.Linear` | `mlp_cuda.cu` |
| K4 | `apex__fused_dense_gelu_dense` | NVIDIA/apex | `FusedDenseGeluDense` | Linear+GELU+Linear | `mlp_cuda.cu` |
| K5 | `flash_attn__flash_attention_2` | Dao-AILab/flash-attention | `flash_attn_func` | `F.scaled_dot_product_attention` | `flash_fwd_kernel.h` |

#### 4.2.1 Kernel 详细数据集信息

**K1: Apex FusedLayerNorm**

| 属性 | 值 |
|------|------|
| 仓库 | [NVIDIA/apex](https://github.com/NVIDIA/apex) (>8k stars) |
| 版本 | 0.1 (commit 安装于 2026-04-29) |
| CUDA 源文件 | `csrc/layer_norm_cuda_kernel.cu` |
| 算法 | Layer Normalization: `y = (x - mean) / sqrt(var + eps) * weight + bias` |
| 外部 API | `apex.normalization.FusedLayerNorm(normalized_shape)` |
| PyTorch 参考 | `torch.nn.LayerNorm(normalized_shape)` |
| 可学习参数 | `weight` (shape=H, init=1.0), `bias` (shape=H, init=0.0) |
| 权重同步需求 | **不需要** — weight/bias 的初始化是确定性的 (1.0/0.0) |
| 输入 shape | `(B=32, S=128, H=1024)` → 3D tensor, float32 |
| 输入参数 | `get_init_inputs() = [1024]` (normalized_shape) |
| 潜在数值差异来源 | 归约算法（Welford vs two-pass）、`rsqrt` 精度、浮点非结合性 |
| 实际用途 | Megatron-LM、DeepSpeed 等大规模 Transformer 训练 |

**K2: Apex FusedRMSNorm**

| 属性 | 值 |
|------|------|
| 仓库 | NVIDIA/apex |
| CUDA 源文件 | `csrc/layer_norm_cuda_kernel.cu` (共享实现) |
| 算法 | RMS Normalization: `y = x / sqrt(mean(x²) + eps) * weight` (无 bias, 无 mean-centering) |
| 外部 API | `apex.normalization.FusedRMSNorm(hidden_size, eps=1e-6)` |
| PyTorch 参考 | 手写 `x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight` |
| 可学习参数 | `weight` (shape=H, init=1.0) |
| 权重同步需求 | **不需要** — weight 初始化为 1.0 |
| 输入 shape | `(B=32, S=128, H=1024)` → 3D tensor, float32 |
| 输入参数 | `get_init_inputs() = [1024, 1e-6]` (hidden_size, eps) |
| 潜在数值差异来源 | `rsqrt` 精度、归约顺序、eps 加法精度 |
| 实际用途 | LLaMA、Mistral 等现代 LLM |

**K3: Apex FusedDense**

| 属性 | 值 |
|------|------|
| 仓库 | NVIDIA/apex |
| CUDA 源文件 | `csrc/mlp_cuda.cu` |
| 算法 | 融合 GEMM+bias: `y = x @ W^T + b` (使用 cuBLAS) |
| 外部 API | `apex.fused_dense.FusedDense(in_features, out_features)` |
| PyTorch 参考 | `torch.nn.Linear(in_features, out_features)` |
| 可学习参数 | `weight` (1024×2048, kaiming_uniform), `bias` (2048, uniform) |
| 权重同步需求 | **需要** — 随机初始化，必须从 ref 复制到 ext (`load_state_dict` 直接兼容) |
| 输入 shape | `(B=32, F=1024)` → **2D** tensor, float32 |
| 输入约束 | FusedDense 仅支持 2D 输入；3D 输入会被展平导致输出 shape 不匹配 |
| 输入参数 | `get_init_inputs() = [1024, 2048]` (in_features, out_features) |
| 潜在数值差异来源 | cuBLAS GEMM 的浮点累加顺序 vs PyTorch 默认 GEMM |
| 实际用途 | Megatron-LM MLP 层 |

**K4: Apex FusedDenseGeluDense**

| 属性 | 值 |
|------|------|
| 仓库 | NVIDIA/apex |
| CUDA 源文件 | `csrc/mlp_cuda.cu` |
| 算法 | 融合 MLP: `y = Linear2(GELU(Linear1(x)))` 三步合并为一次 CUDA kernel |
| 外部 API | `apex.fused_dense.FusedDenseGeluDense(in_f, inter, out_f)` |
| PyTorch 参考 | `Linear(inter, out_f)(GELU(Linear(in_f, inter)(x)))` |
| 可学习参数 | `weight1` (4096×1024), `bias1` (4096), `weight2` (1024×4096), `bias2` (1024) |
| 权重同步需求 | **需要** — 参数名不同（`weight1/bias1` vs `linear1.weight/linear1.bias`），按 shape 顺序匹配 |
| 输入 shape | `(B=16, F=1024)` → **2D** tensor, float32 |
| 输入约束 | 仅支持 2D 输入 |
| 输入参数 | `get_init_inputs() = [1024, 4096, 1024]` (in, intermediate, out) |
| 潜在数值差异来源 | GELU 近似算法差异（tanh 近似 vs erf 精确）、融合 kernel 的中间精度 |
| 诊断预测试结果 | 权重同步后 2D 输入 max_diff=**1.84e-4** — **存在真实数值差异**（GELU 计算路径不同） |
| 实际用途 | Megatron-LM FFN 层 |

**K5: FlashAttention-2**

| 属性 | 值 |
|------|------|
| 仓库 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) (>16k stars) |
| 版本 | 2.8.3 |
| CUDA 源文件 | `csrc/flash_attn/flash_fwd_kernel.h`, `flash_fwd_launch_template.h` |
| 算法 | IO-aware exact attention: `O = softmax(QK^T / √d) V` 使用 tiling 优化显存访问 |
| 外部 API | `flash_attn.flash_attn_func(q, k, v)` |
| PyTorch 参考 | `F.scaled_dot_product_attention(q, k, v)` (may use math backend) |
| 可学习参数 | **无** (纯计算 kernel) |
| 权重同步需求 | **不需要** — 无可学习参数 |
| 输入 shape | Q, K, V 各 `(B=4, S=256, H=8, D=64)` → 4D tensor, **float16** |
| 输入约束 | FlashAttention 要求 float16 或 bfloat16 输入 |
| 输入参数 | `get_init_inputs() = []` |
| 潜在数值差异来源 | online softmax（无需存储完整 attention matrix）的累加精度差异、tiling 边界处理 |
| 诊断预测试结果 | max_diff=**1.22e-4**, allclose at atol=1e-3 — **存在真实数值差异** |
| 实际用途 | GPT、LLaMA 等几乎所有现代 LLM 的注意力计算 |

#### 4.2.2 数据集特性总结

| 特性 | K1 FusedLN | K2 FusedRMS | K3 FusedDense | K4 FDGD | K5 FlashAttn |
|------|-----------|------------|-------------|---------|-------------|
| 参数类型 | 确定性初始化 | 确定性初始化 | 随机初始化 | 随机初始化 | 无参数 |
| 需要权重同步 | 否 | 否 | 是 | 是（名称映射） | 否 |
| 输入精度 | float32 | float32 | float32 | float32 | float16 |
| 输入维度 | 3D | 3D | 2D | 2D | 4D |
| 有归约操作 | 是(mean,var) | 是(rms) | 否(GEMM) | 是(GELU) | 是(softmax) |
| 诊断 max_diff | 0 | 0 | 0(同步后) | 1.84e-4 | 1.22e-4 |
| CUDA 源代码行数 | ~800 | ~800(共享) | ~600 | ~600(共享) | ~2000 |
| GitHub Stars | >8k | >8k | >8k | >8k | >16k |

### 4.3 实验输出定义

#### 4.3.1 每轮测试的原始输出

每次调用 `_stress_worker.py` 子进程返回一个 JSON：

```json
{
    "ref_ok": true,           // PyTorch 参考是否正常执行（无 crash/NaN）
    "ref_nan_fallback": false, // 参考输出是否含 NaN/Inf
    "original_ok": true,       // 外部 kernel 是否与参考一致（allclose 判定）
    "mutant_ok": true,         // 本实验中 = original_ok（无变异）
    "bitwise_orig_mut_eq": true, // 本实验中恒为 true（无变异）
    "time_ms": 234.5           // 本轮执行耗时
}
```

**核心判定字段**：`original_ok`
- `true` → 外部 kernel 在此 (policy, seed, atol) 下与 PyTorch 输出一致
- `false` → 外部 kernel 与 PyTorch 存在超出容差的数值偏差

#### 4.3.2 每个 kernel 的汇总输出

存储路径：`第三次实验汇总/results/details/{kernel_id}.json`

```json
{
    "id": "apex__fused_layer_norm",
    "repo": "NVIDIA/apex",
    "kernel_name": "FusedLayerNorm",
    "baseline": {
        "total": 50, "passed": 50, "failed": 0, "errors": 0
    },
    "value_stress": {
        "large_magnitude": {
            "has_discrepancy": false,
            "results": [{"seed": 50000, "status": "pass"}, "..."]
        },
        "near_overflow": {
            "has_discrepancy": true,
            "results": [{"seed": 50003, "status": "discrepancy"}, "..."]
        }
    },
    "dtype_stress": {
        "float16": {"has_discrepancy": false, "results": ["..."]},
        "bfloat16": {"has_discrepancy": true, "results": ["..."]}
    },
    "training_stress": {
        "large_magnitude": {"has_discrepancy": false, "results": ["..."]},
        "near_overflow": {"has_discrepancy": true, "results": ["..."]}
    },
    "repeated_run": {
        "seeds_tested": 3,
        "max_self_inconsistency": 0,
        "any_ref_mismatch": false,
        "results": [
            {"seed": 80000, "self_consistent": true, "matches_ref": true},
            {"seed": 80001, "self_consistent": true, "matches_ref": true},
            {"seed": 80002, "self_consistent": true, "matches_ref": true}
        ]
    },
    "config_stress": {
        "batch_1":  {"has_discrepancy": false, "results": ["..."]},
        "batch_2":  {"has_discrepancy": false, "results": ["..."]},
        "batch_4":  {"has_discrepancy": false, "results": ["..."]},
        "batch_8":  {"has_discrepancy": false, "results": ["..."]},
        "batch_16": {"has_discrepancy": false, "results": ["..."]},
        "batch_32": {"has_discrepancy": false, "results": ["..."]},
        "batch_64": {"has_discrepancy": true, "results": ["..."]}
    },
    "multi_tolerance": {
        "near_overflow": {
            "0.1": true, "0.01": false, "0.001": false, "0.0001": false, "1e-05": false
        }
    },
    "summary": {
        "status": "COMPLETED",
        "baseline_pass_rate": "50/50",
        "value_stress_discrepancies": 5,
        "dtype_stress_discrepancies": 1,
        "training_stress_discrepancies": 2,
        "repeated_run_inconsistencies": 0,
        "config_stress_discrepancies": 1,
        "total_discrepancies": 9,
        "total_rounds": 206,
        "discrepant_dimensions": ["value_stress", "dtype_stress", "training_stress", "config_stress"]
    }
}
```

#### 4.3.3 全局汇总输出

存储路径：`第三次实验汇总/results/summary.json`

#### 4.3.4 期望的最终报告

存储路径：`第三次实验汇总/第三次实验报告.md`

**表 1：各 kernel 六维度测试汇总**

| Kernel | 仓库 | 基线 | value_stress | dtype_stress | training_stress | repeated_run | config_stress | 总偏差率 |
|--------|------|------|-------------|-------------|----------------|-------------|-------------|---------|
| FusedLayerNorm | NVIDIA/apex | ?/50 | ?/63 | ?/6 | ?/63 | ?/3 | ?/21 | ?/206 |
| FusedRMSNorm | NVIDIA/apex | ?/50 | ?/63 | ?/6 | ?/63 | ?/3 | ?/21 | ?/206 |
| FusedDense | NVIDIA/apex | ?/50 | ?/63 | ?/6 | ?/63 | ?/3 | ?/21 | ?/206 |
| FusedDenseGeluDense | NVIDIA/apex | ?/50 | ?/63 | ?/6 | ?/63 | ?/3 | ?/21 | ?/206 |
| FlashAttention-2 | Dao-AILab | ?/50 | ?/63 | ?/6 | ?/63 | ?/3 | ?/21 | ?/206 |

**表 2：多容差精度谱（每个发现偏差的 kernel）**

| Kernel | 偏差维度 | atol=1e-1 | atol=1e-2 | atol=1e-3 | atol=1e-4 | atol=1e-5 |
|--------|--------|----------|----------|----------|----------|----------|
| ? | value_stress | pass/fail | pass/fail | pass/fail | pass/fail | pass/fail |
| ? | dtype_stress | pass/fail | pass/fail | pass/fail | pass/fail | pass/fail |
| ? | training_stress | pass/fail | pass/fail | pass/fail | pass/fail | pass/fail |

**表 3：维度有效性分析——各维度的整体偏差检出能力**

| 维度 | 检测到偏差的 kernel 数 / 总数 | 偏差总轮次 / 总轮次 | 追溯的主要变异算子类别 |
|------|---------------------------|-------------------|-------------------|
| value_stress | ? / 6 | ? / 378 | 15 种全覆盖 |
| dtype_stress | ? / 6 | ? / 36 | cast_remove, acc_downgrade |
| training_stress | ? / 6 | ? / 378 | ML 数值语义类 |
| repeated_run | ? / 6 | ? / 18 | sync_remove |
| config_stress | ? / 6 | ? / 126 | launch_config_mutate, mask_boundary |

**表 4：策略有效性分析——value_stress 中检测到最多偏差的策略**

| 策略 | 源自变异算子 | 检测到偏差的 kernel 数 | 偏差的 kernel 列表 |
|------|-----------|-------------------|------------------|
| large_magnitude | arith_replace | ? | ? |
| near_overflow | cast_remove, stab_remove | ? | ? |
| near_zero | epsilon_modify | ? | ? |
| ... | ... | ? | ? |

---

## 五、完整追溯链总结

### 5.1 从变异算子到输入策略的完整映射

以下展示**每种输入策略的完整追溯路径**：

| 策略 | 直接来源文件 | 来源章节 | 追溯到的变异算子 | 变异算子文件 | State Infection Condition |
|------|-----------|---------|--------------|-----------|--------------------------|
| `large_magnitude` | `StressEnhance_Plan.md` | §9 第 1 行 | `arith_replace` | `operators/arithmetic.py` | 算术差异与操作数量级正比 |
| `near_overflow` | `StressEnhance_Plan.md` | §9 第 2 行 | `cast_remove`, `stab_remove` | `operators/ml_semantic.py` | 类型转换/稳定化在溢出边界失效 |
| `near_zero` | `StressEnhance_Plan.md` | §9 第 3 行 | `epsilon_modify` | `operators/ml_semantic.py` | 除法中 eps 在 x→0 时关键 |
| `denormals` | `StressEnhance_Plan.md` | §9 第 4 行 | `epsilon_modify`, `cast_remove` | `operators/ml_semantic.py` | 非正规数的 flush-to-zero 行为差异 |
| `all_negative` | `StressEnhance_Plan.md` | §9 第 5 行 | `init_modify` | `operators/ml_semantic.py` | max 初始值在全负输入时成为结果 |
| `all_positive` | `StressEnhance_Plan.md` | §9 第 6 行 | `init_modify` | `operators/ml_semantic.py` | min 初始值在全正输入时成为结果 |
| `mixed_extremes` | `StressEnhance_Plan.md` | §9 第 7 行 | `acc_downgrade` | `operators/ml_semantic.py` | 大小值混合放大精度降级影响 |
| `alternating_sign` | `StressEnhance_Plan.md` | §9 第 8 行 | `reduction_reorder` | `operators/ml_semantic.py` | 浮点非结合性在交替正负时最大 |
| `sparse` | `StressEnhance_Plan.md` | §9 第 9 行 | `mask_boundary` | `operators/gpu_parallel.py` | 稀疏下边界元素成为关键路径 |
| `uniform_constant` | `StressEnhance_Plan.md` | §9 第 10 行 | `scale_modify` | `operators/ml_semantic.py` | 均匀值让缩放差异一致可见 |
| `structured_ramp` | `StressEnhance_Plan.md` | §9 第 11 行 | `index_replace` | `operators/gpu_parallel.py` | 线性递增让索引偏移直接可测 |
| `boundary_last_element` | `StressEnhance_Plan.md` | §9 第 12 行 | `mask_boundary`, `relop_replace` | `operators/gpu_parallel.py`, `arithmetic.py` | off-by-one 在末位元素暴露 |
| `head_heavy` | `StressEnhance_Plan.md` | §9 第 13 行 | `index_replace` | `operators/gpu_parallel.py` | 索引头部偏移导致只处理前段 |
| `tail_heavy` | `StressEnhance_Plan.md` | §9 第 14 行 | `index_replace` | `operators/gpu_parallel.py` | 索引尾部偏移导致跳过后段 |
| `relop_boundary_hit` | `EquivalentDetection_V2_Plan.md` | §5.1 第 1 行 | `relop_replace` | `operators/arithmetic.py` | 关系运算差异仅在 x=阈值时 |
| `extreme_magnitude` | `EquivalentDetection_V2_Plan.md` | §5.1 第 2 行 | `arith_replace`, `scale_modify` | `operators/arithmetic.py`, `ml_semantic.py` | 极端值放大算术/缩放差异 |
| `near_epsilon` | `EquivalentDetection_V2_Plan.md` | §5.1 第 3 行 | `epsilon_modify` | `operators/ml_semantic.py` | x≈eps 时 eps 变化直接影响结果 |
| `reduction_adversarial` | `EquivalentDetection_V2_Plan.md` | §5.1 第 4 行 | `acc_downgrade`, `reduction_reorder` | `operators/ml_semantic.py` | 对消序列让精度降级/顺序变更可见 |
| `init_sensitive` | `EquivalentDetection_V2_Plan.md` | §5.1 第 5 行 | `init_modify` | `operators/ml_semantic.py` | 全同号输入让初始值成为决定因素 |
| `dense_nonzero` | `StressEnhance_Plan.md` | §9 稀疏梯度第 1 行 | `arith_replace`, `epsilon_modify` | `operators/arithmetic.py`, `ml_semantic.py` | 零值让 `a+b = a-b` 不可见 |
| `sparse_extreme` | `StressEnhance_Plan.md` | §9 稀疏梯度第 3 行 | `mask_boundary`, `relop_replace` | `operators/gpu_parallel.py`, `arithmetic.py` | 极稀疏中极端值命中边界时放大 |

### 5.2 策略在本实验中的意义转换

在 MutaKernel 变异测试中，策略的目的是**杀死变异体**（让 mutant 的输出偏离 original）。

在本实验（差分测试）中，同样的策略被用于**检测外部 kernel 的数值偏差**（让外部 kernel 的输出偏离 PyTorch 参考）。

转换的合理性在于：变异算子模拟的 15 种故障模式（算术错误、边界错误、精度降级等）恰好也是真实 CUDA kernel 开发中最常见的 Bug 类型。因此，针对这些故障设计的 stress 输入，同样适用于暴露真实 kernel 中的同类问题。

---

## 六、技术实现

### 6.1 代码文件清单

| 文件 | 职责 | 是否修改现有代码 |
|------|------|---------------|
| `external_benchmarks/registry.py` | 5 个外部 kernel 的清单和 ModelNew 源码 | 新建 |
| `external_benchmarks/apex/*.py` (×4) | Apex kernel 的 PyTorch 参考实现 | 新建 |
| `external_benchmarks/flash_attention/flash_attn_ref.py` | FlashAttention 的 PyTorch 参考 | 新建 |
| `scripts/run_external_diff_test.py` | 差分测试驱动脚本 | 新建 |
| `scripts/_stress_worker.py` | 子进程 worker（执行三路比较） | **最小修改**：+`_sync_weights()` 函数和 `sync_weights` 配置项 |
| `src/stress/policy_bank.py` | 21 种 stress 策略定义 | **不修改** |

### 6.2 执行环境

| 项目 | 值 |
|------|------|
| OS | Ubuntu 22.04 (WSL2) |
| Python | 3.10.12 |
| PyTorch | 2.5.0+cu124 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| NVIDIA Apex | 0.1 (with CUDA extensions) |
| FlashAttention | 2.8.3 |
| Triton | 3.1.0 |

### 6.3 结果存储路径

```
MutaKernel/第三次实验汇总/
├── docs/
│   └── ExternalDiffTest_Plan.md    ← 本文档
├── results/
│   ├── summary.json                ← 全局汇总
│   └── details/
│       ├── apex__fused_layer_norm.json
│       ├── apex__fused_rms_norm.json
│       ├── apex__fused_dense.json
│       ├── apex__fused_dense_gelu_dense.json
│       └── flash_attn__flash_attention_2.json
├── external_diff_test.log          ← 运行日志
└── 第三次实验报告.md               ← 最终报告（实验完成后生成）
```

---

## 七、参考文献

| # | 引用 | 出处 | 与本实验的关系 |
|---|------|------|-------------|
| 1 | Offutt, A.J. & Lee, S.D. "An Empirical Evaluation of Weak Mutation" | IEEE TSE 22(5), 1996 | state infection condition 理论基础 |
| 2 | Du, H. et al. "To Kill a Mutant: An Empirical Study of Mutation Testing Kills" | ACM ISSTA, 2023 | 多维度杀伤策略的动机来源 |
| 3 | Laguna, I. et al. "Testing GPU Numerics" | SC Workshop, 2024 | GPU 浮点数值差异的实证 |
| 4 | Ouyang, S. et al. "KernelBench: Can LLMs Write GPU Kernels?" | NeurIPS, 2024 | KernelBench 基准与 shape 契约 |
| 5 | CuFuzz. "CuFuzz: API-Knowledge-Graph Coverage-Driven Fuzzing for CUDA Libraries" | ACM FSE, 2026 | CUDA 库模糊测试 |
