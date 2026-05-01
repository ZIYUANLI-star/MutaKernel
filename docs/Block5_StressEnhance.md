# Block 5: 诊断驱动的 GPU Kernel 测试增强与 LLM 辅助分析

> **对应文件**:
> - Phase 1 (压力测试): `src/stress/policy_bank.py`, `src/stress/differential_tester.py`,
>   `src/mutrepair/enhanced_inputs.py`, `scripts/run_stress_enhance.py`, `scripts/_stress_worker.py`
> - Phase 2+3 (LLM 分析): `src/stress/llm_analyzer.py`, `scripts/_pilot_llm20.py`,
>   `scripts/_verify_llm_suggestions.py`
> - Baselines/Ablation: `scripts/run_baselines.py`, `scripts/run_ablation.py`
> - 流水线: `scripts/_run_full_pipeline.sh`
>
> **论文位置**: Section 4 (核心方法) + RQ2 (Effectiveness) + RQ3 (Ablation) + RQ4 (鲁棒性)
>
> **定位**: 核心贡献 — 诊断驱动的测试增强 + LLM 迭代分析闭环
>
> **最后更新**: 2026-04-16 — LLM 迭代分析 + 事后聚类 + 鲁棒性建议

---

## 一、问题背景

### 1.1 Block 1-2 揭示的问题

90 个 GPU kernel 的变异测试 (1663 变异体) 揭示:
- killed=944, survived=322, Baseline MS = 0.7457
- 322 个存活变异体并非随机分布，而是因 6 类系统性盲区而存活 (详见 `docs/attribution_criteria.md`)

### 1.2 核心问题

> 存活变异体为什么没被杀掉？根据诊断结果，能否设计针对性增强？

### 1.3 与旧版的区别

| 维度 | 旧版 StressEnhance | 新版 (当前) |
|------|-------------------|------------|
| 目标范围 | 仅 C/D 类存活变异体 | **全部 322 个存活变异体 (A/B/C/D)** |
| 归因分类 | 4 类 (Type 1-4) | **LLM 自由分析 + 事后聚类** |
| 策略数 | 11 个通用策略 | **14 个** (新增 boundary_last_element, head_heavy, tail_heavy) |
| 增强层次 | 仅 value-stress | **4 层: Value + Configuration + Execution + Training-Mode** |
| 策略选择 | 遍历全部, 无优先级 | **STRATEGY_MAP 诊断→策略映射, 优先执行** |
| 每 policy 输入量 | 1 seed | **3 seeds** (多随机种子) |
| Baseline | 无 | **2 个: 20x random + random stress** |
| 消融 | 简单策略消融 | **leave-one-out + 策略数量曲线** |

---

## 二、方法论创新

| 组件 | 功能 | 文件 |
|------|------|------|
| **Stress Policy Bank** | 14 种数值分布策略 (仅改变值, 不改 shape/dtype) | `policy_bank.py` |
| **4-Layer Augmentation** | Value / Configuration / Execution / Training-Mode 四层独立增强 | `run_stress_enhance.py` |
| **Diagnosis→Strategy Mapping** | 算子→策略优先级表 | `enhanced_inputs.py` STRATEGY_MAP |
| **LLM Iterative Analysis** | 自由分析 + 迭代验证 + 事后聚类 | `llm_analyzer.py`, `_pilot_llm20.py` |
| **Baselines** | 20x random + random stress | `run_baselines.py` |
| **Ablation** | Leave-one-out + 策略曲线 | `run_ablation.py` |

---

## 三、Stress Policy Bank (14 策略)

> **适用层**: 仅 Layer 1 (Value-Distribution Augmentation)。
> **约束**: 所有策略保持 **shape 和 dtype 不变**，仅改变 tensor 的数值分布。
> Layer 2 的 dtype 切换是独立机制, 不属于 Policy Bank。

| 策略 | 生成方式 | 暴露的缺陷类型 |
|------|---------|---------------|
| `large_magnitude` | randn × 1000 | 大数值下精度/溢出 |
| `near_overflow` | randn × dtype_max | dtype 上界附近 |
| `near_zero` | randn × 1e-7 | 极小值除法/归一化 |
| `denormals` | randn × 1e-38 | 非规格化浮点数 |
| `all_negative` | −\|randn\| × 100 | 全负值暴露 init 缺失 |
| `all_positive` | \|randn\| × 100 | 全正值对比 |
| `mixed_extremes` | 50%×10000 + 50%×0.0001 | 极端混合 |
| `alternating_sign` | \|randn\|×100, 奇偶交替正负 | 符号交替 |
| `sparse` | 90% 零 + 10% randn×100 | 稀疏输入 |
| `uniform_constant` | 全部填充 88.0 | 暴露平移不变性 |
| `structured_ramp` | arange(N)/N | 线性递增, 暴露索引边界 |
| `boundary_last_element` | randn + 最后位置 1e4 | off-by-one 边界 |
| **`head_heavy`** | 前 1/4 × 10000 + 其余 × 0.01 | **index 退化只处理前段** (新增) |
| **`tail_heavy`** | 前段 × 0.01 + 后 1/4 × 10000 | **index 退化跳过尾段** (新增) |

---

## 四、四层增强架构与输入预算

### 4.0 输入预算总览

| 层 | 触发条件 | 计算方式 | 最大 worker 调用数 |
|----|---------|---------|------------------|
| **Layer 1** | 所有 322 个存活变异体 | 14 policies × 3 seeds | **42** |
| **Layer 2** | Layer 1 未杀死的变异体 | 3 seeds × 2 dtypes | **6** |
| **Layer 3** | Layer 1+2 均未杀死的变异体 | 3 seeds × 10 trials/seed | **3** (worker 调用), 30 次执行 |
| **Layer 4** | Layer 1-3 均未杀死 + 目标算子 | 14 policies × 3 seeds | **42** (仅 5 个目标算子) |
| **合计** | — | — | **最多 93 次 worker 调用/变异体** |

> **早停机制**: 任一层杀死即跳过后续层; 同一层内任一 (policy, seed) 组合杀死即跳过该层后续调用。
> **Layer 4 选择性触发**: 仅对 `epsilon_modify`, `const_perturb`, `init_replace`, `arith_replace`, `cast_remove` 5 类算子触发 (其余算子跳过)。

### 4.1 Layer 1: Value-Distribution Augmentation

**目的**: 用多样化数值分布暴露 "标准 randn 盲区" 中的差异。

**流程**:
1. 按 STRATEGY_MAP 排列 14 个 policy 的优先级 (推荐策略先执行)
2. 对每个 policy, 用 **3 个不同 seed** 生成 3 组不同的 stress 输入
3. 每组输入: 分别运行 ref / original / mutant, 三方对比
4. 任一组: `original OK ∧ mutant FAIL` → **KILLED**, early stop

**核心映射表 (诊断→策略):**

| 算子 | 推荐策略 | 存活机制 | 文献依据 |
|------|---------|---------|---------|
| epsilon_modify | `near_zero`, `denormals` | var >> eps 掩盖差异 | FPGen (ICSE'20) |
| scale_modify | `uniform_constant`, `structured_ramp` | var ≈ 1 | Magneto (ISSTA'21) |
| stab_remove | `large_magnitude`, `near_overflow` | randn 不触发溢出 | FPGen (ICSE'20) |
| cast_remove | `near_overflow`, `large_magnitude` | float32 下恒等 | — |
| init_modify | `all_negative`, `sparse` | 正值掩盖 0.0 初值 | BVA |
| acc_downgrade | `mixed_extremes`, `large_magnitude` | 累加精度差异被 randn 掩盖 | — |
| reduction_reorder | `mixed_extremes`, `alternating_sign` | 求和顺序差异被抵消 | — |
| broadcast_unsafe | `structured_ramp` | 对称输入掩盖广播错误 | — |
| layout_assume | `structured_ramp` | 连续内存布局恰好正确 | — |
| index_replace | `structured_ramp`, `large_magnitude`, **`head_heavy`**, **`tail_heavy`** | 1D launch 退化 | Mu2 (ISSTA'23) |
| mask_boundary | `boundary_last_element`, `sparse` | off-by-one | BVA |
| sync_remove | `large_magnitude`, `mixed_extremes` | 竞态条件 | — |
| launch_config_mutate | `structured_ramp`, `large_magnitude` | grid-stride 吸收 | Mu2 (ISSTA'23) |
| arith_replace | `large_magnitude`, `mixed_extremes` | batch_idx=0 | FPGen + BVA |
| relop_replace | `boundary_last_element`, `structured_ramp` | 索引不达边界 | BVA |
| const_perturb | `near_zero`, `large_magnitude` | 微扰被 tolerance 吸收 | Magneto (ISSTA'21) |

> **注**: `all_positive` 未被任何算子列为推荐策略，作为通用对照策略参与 Layer 1 全量执行。全部 14 个策略对每个变异体均会执行，STRATEGY_MAP 仅决定执行**优先级**（推荐策略先执行，配合早停机制提高效率）。

### 4.2 Layer 2: Configuration Augmentation

**目的**: 切换输入 dtype 暴露 "float32 恒等操作" 中的隐藏差异。

> **注意**: 此层 **不使用 Policy Bank 策略**, 而是独立的配置增强机制。
> 它改变的是 **dtype** (float32 → float16/bfloat16), 而非数值分布。

**流程**:
1. 用 **3 个不同 seed** 生成 3 组 base 输入 (`get_inputs()`)
2. 每组输入分别 cast 到 float16 和 bfloat16 (model 同步转换)
3. 每种 dtype: 分别运行 ref / original / mutant, 三方对比
4. 任一 (seed, dtype) 组合: `original OK ∧ mutant FAIL` → **KILLED**, early stop

**适用范围**: 主要针对 `cast_remove` (22个), 因为 `static_cast<float>(float_value)`
在 float32 下是恒等操作, 但在 float16/bfloat16 下会丢失精度。

### 4.3 Layer 3: Execution Augmentation

**目的**: 通过同一输入多次执行, 检测因 GPU 竞态条件导致的概率性存活。

> **注意**: 此层的 **"多 seed"** 指使用多组不同的 base 输入。
> 每组 base 输入在同一 worker 调用中执行 10 次 (same input, 10 trials)。

**流程**:
1. 用 **3 个不同 seed** 生成 3 组 base 输入
2. 每组输入: mutant 连续执行 10 次 (共 30 次执行)
3. **any-divergence 检测** (非多数投票):
   - 任一次 mutant 与 ref 不一致 → **KILLED**
   - mutant 多次输出自身不一致 → **KILLED (self-inconsistency)**
4. 任一 seed 检测到 divergence → early stop

**适用范围**: 主要针对 `sync_remove` (~25个) 和部分 `index_replace`。

### 4.4 Layer 4: Training-Mode Augmentation (eval-mode masking)

**目的**: 将 model 从 `.eval()` 切换到 `.train()` 模式, 暴露被 eval 模式中的
固定统计量 (running_mean, running_var) 掩盖的差异。

**背景**: Pilot 实验发现 `epsilon_modify` 类变异体 (如 `eps: 1e-5 → 0`) 在 eval 模式下
存活, 因为 `running_var=1.0` (未经训练的默认值) 使得
`rsqrt(1.0 + 0) ≈ rsqrt(1.0 + 1e-5)`, 差异远小于 atol。
在 training 模式下, BatchNorm 使用当前 batch 的实际方差 (可能接近 0),
使 eps 差异被放大到可观测水平。

**流程**:
1. 仅对 `TRAINING_TARGET_OPS` 中的算子触发 (当前 5 个)
2. 按 STRATEGY_MAP 排列 14 个 policy 的优先级
3. 对每个 policy, 用 **3 个不同 seed** 生成 stress 输入
4. 所有 model (ref / original / mutant) 均以 `.train()` 模式运行
5. 任一组: `original OK ∧ mutant FAIL` → **KILLED**, early stop

**Worker 模式**: `mode="training_stress"` (复用 value-stress 的三方对比逻辑, 仅改模型模式)

**适用范围**: 主要针对 `epsilon_modify` (13个) 和部分 `const_perturb`, `init_replace`。

---

## 五、两阶段处理

### 5.1 Phase 1: 4 层压力测试（自动杀死）

Layer 1-4 的作用是 **杀死** 变异体，不做分类归因。被杀死的变异体只记录：
- `killed_by_layer`: 被哪一层杀死 (layer1_value / layer2_dtype / layer3_repeated / layer4_training)
- `killing_policy`: 具体杀死的策略名
- `killing_seed`: 杀死时使用的 seed

**对应文件**: `scripts/run_stress_enhance.py`, `src/stress/differential_tester.py`

### 5.2 Phase 2: LLM 迭代分析 + 验证（核心归因层）

**目的**: 对 4 层压力测试后 **所有未被杀死的变异体**，由 LLM 进行 **自由分析**，
不预设分类体系。LLM 负责：
- 分析存活原因（自然语言，不受预设 Type 约束）
- 判断是否可被杀死
- 如可杀 → 给出 5 个建议输入 → GPU 验证 → 迭代（最多 3 轮）
- 如不可杀 → 给出鲁棒性建议 + 修复后的 kernel 代码

**对应文件**: `src/stress/llm_analyzer.py`, `scripts/_pilot_llm20.py`,
`scripts/_verify_llm_suggestions.py`, `scripts/_stress_worker.py`

**流程**:
```
对每个存活变异体 (最多 3 轮迭代):
  Round 1:
    ├── LLM 自由分析存活原因 + 判断是否可杀
    ├── 可杀 → 建议 5 个输入 → GPU 验证
    │   ├── 杀死 → 提取测试规则 (LLM)
    │   └── 未杀死 → Round 2
    └── 不可杀 → 生成鲁棒性建议 + 修复代码
  Round 2-3:
    ├── LLM 结合前轮失败信息重新分析
    ├── 新建议输入 → GPU 验证
    └── ...
  3 轮均未杀死 → 生成鲁棒性建议 + 修复代码
```

### 5.3 Phase 3: 事后聚类（分类体系涌现）

所有分析完成后，将全部 `survival_reason` 交给 LLM 进行聚类归纳：
- 从数据中发现 3-8 个存活原因类别
- 为每个类别生成定义和标签
- 将每个变异体分配到类别

**输出**: `llm_analysis_results/taxonomy.json`

**论文价值**: 分类体系从实验数据中涌现，而非预设，具有更高的科学性和说服力。

### 5.4 最终输出

| 输出文件 | 内容 | 论文用途 |
|---------|------|---------|
| `llm_analysis_results/details/{mutant_id}.json` | 每个变异体的完整分析 | 案例分析 |
| `llm_analysis_results/taxonomy.json` | 数据驱动的存活原因分类 | 分类表格 |
| `llm_analysis_results/robustness_suggestions.json` | 鲁棒性增强建议 + 修复代码 | RQ4 案例 |
| `llm_analysis_results/test_construction_rules.json` | 测试用例构造规则 | RQ2 改进策略 |
| `llm_analysis_results/summary.json` | 汇总统计 | 实验数据 |

---

## 六、Baseline 与 Ablation

### 6.1 Baseline 1: 20x Random Runs

- 对 322 个存活变异体，用 `get_inputs()` 生成 20 次随机输入 (seed 42-61)
- 仍用标准 `torch.randn`，不改变值分布, 不切换 dtype
- **目的**: 证明 randn 值域集中在 [-3,3]，多跑无用

### 6.2 Baseline 2: Random Stress (Unguided)

- 对每个存活变异体，从 14 个 policy 中**随机选 3 个**
- 每个 policy × 3 seeds = 9 次 worker 调用/变异体 (同主实验 seed 预算)
- 不使用 STRATEGY_MAP，不看变异体类型
- **目的**: 证明诊断驱动 > 盲目 stress

### 6.3 Ablation: Leave-one-out

- 逐一去掉某个 policy，观察杀灭率下降 → 策略贡献排名
- 每 policy × 3 seeds (同主实验 seed 预算)
- 策略数量曲线: {1, 3, 5, 7, 14} 个策略时的杀灭率 → 是否饱和

---

## 七、Differential Testing 流程

### 7.1 三方对比

```
对每个 (policy, seed) 组合:
  input = policy(get_inputs(), seed)       # Layer 1 — Policy Bank 生成
  ref_output    = reference_model(input)   # PyTorch 标准实现
  orig_output   = original_kernel(input)
  mutant_output = mutant_kernel(input)

  if ref has NaN/Inf → 跳过 (无效输入)
  if allclose(ref, orig) ∧ ¬allclose(ref, mutant) → KILLED_BY_STRESS
  if ¬allclose(ref, orig) ∧ ¬allclose(ref, mutant) → ORIGINAL_ALSO_FAILS (Type 4 候选)
  if allclose(ref, orig) ∧ allclose(ref, mutant) → 继续下一个 seed/policy
```

### 7.2 子进程隔离

每个 (mode, mutant, seed) 三元组在独立子进程 `_stress_worker.py` 中执行:
- 编译挂起 → 120s 超时
- Worker 支持 4 种 mode: `value_stress` / `dtype_stress` / `repeated_run` / `training_stress`
- `__identity__` policy 支持 Baseline 1 (原始 get_inputs 无变换)

### 7.3 四层执行流程

```
对每个存活变异体:
  ├── Layer 1: value-stress (14 policies × 3 seeds, STRATEGY_MAP 优先)
  │   └── 杀死 → 记录 killed_by=layer1_value + killing_policy + seed, 停止
  ├── Layer 2: dtype-stress (3 seeds × {float16, bfloat16})
  │   └── 杀死 → 记录 killed_by=layer2_dtype + killing_dtype + seed, 停止
  ├── Layer 3: repeated-run (3 seeds × 10 trials, any-divergence)
  │   └── 杀死 → 记录 killed_by=layer3_repeated + divergent_trial + seed, 停止
  ├── Layer 4: training-stress (14 policies × 3 seeds, 仅目标算子)
  │   └── 杀死 → 记录 killed_by=layer4_training + killing_policy + seed, 停止
  └── 未杀死 → 进入 Phase 2 (LLM 迭代分析)
```

---

## 八、数据流

```
full_block12_results/details/*.json
  │
  ├── load_all_survived() — 筛选全部 322 个存活变异体 (A/B/C/D)
  │
  └── 对每个变异体:
      │
      ├── Layer 1: _run_layer1_value_stress()
      │   └── _stress_worker.py (mode=value_stress, 14 policies × 3 seeds)
      │
      ├── Layer 2: _run_layer2_dtype_stress()
      │   └── _stress_worker.py (mode=dtype_stress, 3 seeds × 2 dtypes)
      │
      ├── Layer 3: _run_layer3_repeated()
      │   └── _stress_worker.py (mode=repeated_run, 3 seeds × 10 trials)
      │
      ├── Layer 4: _run_layer4_training()  (仅目标算子)
      │   └── _stress_worker.py (mode=training_stress, 14 policies × 3 seeds)
      │
      └── 记录 killed/survived 状态
          │
          └── 保存到 stress_enhance_results/
              ├── details/{mutant_id}.json
              ├── completed.json (断点续传)
              └── stress_summary.json

baseline_results/
  ├── b1_random20.json       (Baseline 1: 20 random seeds, __identity__ policy)
  └── b2_random_stress.json  (Baseline 2: 3 random policies × 3 seeds)

ablation_results/
  ├── leave_one_out.json     (每 policy × 3 seeds)
  └── strategy_curve.json    ({1,3,5,7,14} policies × 3 seeds × 3 repeats)
```

---

## 九、报告指标

| 指标 | 定义 | 意义 |
|------|------|------|
| Baseline MS | 944/1266 = 0.746 | 起点 |
| Augmented MS | (944 + new_kills) / 1266 | 增强后 |
| Adjusted MS | (944 + new_kills) / (1266 - equivalent_count) | 排除不可杀 (由 LLM 聚类判定) |
| Visibility Lift | (Augmented - Baseline) / Baseline | 相对改善 |
| Per-category Lift | 各类别 MS 变化 | 哪类受益最大 |
| Kill Attribution | 每个 policy/mode/seed 的 kill 数 | 策略有效性排名 |
| Guided vs Random Precision | 命中率对比 | 诊断的价值 |

---

## 十、局限性 (论文中坦承)

1. **策略覆盖的有限性**: 14 种策略不能保证覆盖所有触发条件。报告 residual analysis。
2. **LLM 分析的可重复性**: LLM 归因依赖模型输出，不同运行可能有差异。报告置信度分布。
3. **dtype 兼容性**: 部分 kernel 的 model 可能不支持 float16/bfloat16 转换。Worker 中 try/except 保护。
4. **ref implementation 可靠性**: 使用 PyTorch 标准实现作为 oracle，通过 NaN/Inf 检查做基本防护。
5. **STRATEGY_MAP 可能过拟合**: 映射是基于人工分析建立的。Baseline 2 (unguided) 用于对比验证诊断的价值。
6. **seed 数量选择**: 3 seeds/policy 是效率与覆盖率的折衷, 后续可通过 pilot 实验调优。
7. **事后聚类的主观性**: LLM 聚类得出的分类体系可能因 prompt 不同而变化。报告 taxonomy rationale。

---

## 十一、鲁棒性修复与因果隔离（原 Block 4 MutRepair 合并）

> **原独立文件**: `docs/Block4_MutRepair.md` (已归档)
>
> MutRepair 的核心功能（LLM 辅助修复 + 鲁棒性建议）已合并进 Phase 2 的
> LLM 迭代分析流水线。本节保留因果隔离实验设计作为 RQ4 的方法论参考。

### 11.1 变异分析如何驱动修复

Phase 2 中，当 LLM 判定变异体不可杀 (killable=false) 或 3 轮验证均失败时，
LLM 生成两项输出：

1. **鲁棒性建议** (robustness_suggestion): 自然语言描述原始 kernel 的潜在缺陷
2. **修复代码** (robustness_code): 修复后的 kernel 实现

与旧 MutRepair 的关键区别：

| 维度 | 旧 MutRepair (Block 4) | 当前 Phase 2 LLM |
|------|----------------------|-----------------|
| 触发条件 | 仅 Type 4 (预判定后) | 所有 3 轮未杀死的存活变异体 |
| 分类依赖 | 需先完成 Type 1-6 归因 | 无需预分类，自由分析 |
| 输入生成 | EnhancedInputGenerator | LLM 建议 + GPU 验证 |
| 修复反馈 | 5 级 Baseline (B0-Ours) | 单一最强模式 (含完整上下文) |
| 验证 | 双重验证 (标准+增强) | GPU 三方对比验证 |

### 11.2 因果隔离实验设计 (RQ4 可选)

如需回答 "变异分析信息的增量贡献是什么"，可设计以下对比：

```
B0: 标准修复          — 给 LLM: kernel_code + "修复此 kernel"
B1: 通用提示          — 给 LLM: B0 + "请注意数值稳定性"
B2: 增强测试反馈      — 给 LLM: B0 + 失败输入描述 + 失败现象
Ours: 完整变异分析    — 给 LLM: B2 + 变异位置 + 算子类型 + 存活原因分析
```

此设计保留在 `src/mutrepair/feedback_builder.py` 中，可在论文需要时启用。

### 11.3 对应代码文件

| 文件 | 角色 | 状态 |
|------|------|------|
| `src/stress/llm_analyzer.py` | Prompt 模板 + LLM 调用 + 响应解析 | ✅ 主用 |
| `scripts/_pilot_llm20.py` | Phase 2+3 编排 (迭代分析 + 聚类) | ✅ 主用 |
| `scripts/_verify_llm_suggestions.py` | GPU 验证 LLM 建议输入 | ✅ 主用 |
| `src/mutrepair/feedback_builder.py` | 因果隔离 Prompt 构造 (B0-Ours) | 📦 保留，RQ4 可选 |
| `src/mutrepair/repair_loop.py` | 修复循环 + 双重验证 | 📦 保留，RQ4 可选 |
| `src/mutrepair/enhanced_inputs.py` | STRATEGY_MAP (供 Layer 1 优先排序) | ✅ 主用 |
