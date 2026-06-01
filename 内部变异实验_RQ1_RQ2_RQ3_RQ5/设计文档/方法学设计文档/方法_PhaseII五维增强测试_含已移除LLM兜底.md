# Phase II: 多维度增强测试 — 完整方法说明（含已移除的 LLM 迭代分析层）

> 与代码完全同步，最后更新: 2026-05-08
> 代码版本: `run_stress_enhance.py` + `_stress_worker.py` + `llm_analyzer.py` + `policy_bank.py` + `enhanced_inputs.py` + `differential_tester.py` + `equivalent_detector.py`
>
> ⚠️ **方法学口径说明（2026-05-15 定稿）**：论文最终的 **MutaKernel 方法只含 5 个确定性 stress 维度**（`value_stress / dtype_stress / training_stress / repeated_run / config_stress` + Tier 1 回放）。本文档第 14 节描述的 **DeepSeek-R1「LLM 迭代分析」层是 Phase II 早期内嵌组件，已从论文最终方法中移除**：其在 534 个变异体上产生的 3 个 kill **不计入** MutaKernel 的 166（论文 Table 7），这 3 个变异体改由 Task A（RQ2 审计，Opus 4.5）重新处理。下文凡出现"5 维 + 1 LLM / 6 个维度"的表述，请按"5 个确定性维度（LLM 层仅作历史留档）"理解。统计口径以 `kill_summary.deterministic_killed` 为准。

---

## 1. 研究背景与动机

### 1.1 核心问题: survived mutant 的处置困境

变异测试的核心产出是 survived mutant（未被测试套件杀死的变异体），它们可能是: (1) **等价变异体** — 语义等价于原程序，不可能被任何测试杀死; (2) **stubborn mutant**（顽固变异体）— 非等价但现有测试不够强，未触发其差异行为。区分两者是变异测试领域的根本难题。

Du et al. (ISSTA 2023) 的实证研究 "To Kill a Mutant" 揭示了变异体杀死机制的多样性: 一个变异体可能因断言失败、程序崩溃或输出偏差等不同原因被杀死，且 crash 对变异体杀死的贡献远超预期。这意味着**单一维度的测试策略难以覆盖所有杀死路径**。

### 1.2 传统方法的不足

现有工具（如 PIT for Java, Stryker for JavaScript）在检测到 survived mutant 后，通常只报告"测试不足"的诊断信息。Google 在工业实践中将变异测试集成到 code review 流程，但也只是标记 survived mutant 供人工审查。**没有现有工具自动化地针对 survived mutant 生成多维度测试策略并执行验证。**

### 1.3 GPU/CUDA kernel 的测试复杂性

GPU kernel 的测试远比传统顺序程序复杂:

1. **数值敏感性**: GPU 浮点运算受精度（FP16/BF16/FP32）、运算顺序、编译器优化等多重因素影响。
2. **执行模式差异**: kernel 在 `.eval()` 和 `.train()` 模式下行为不同，BatchNorm/LayerNorm 使用不同统计量。
3. **配置敏感性**: kernel 的正确性可能依赖于 batch size、tensor shape 等配置参数。
4. **并发非确定性**: `__syncthreads()` 移除等变异可能引入 data race，表现为非确定性错误。
5. **参考实现边界行为**: PyTorch 参考模型在极端输入下可能产生 NaN/Inf，使差分比较失效。

### 1.4 LLM 辅助测试生成的兴起

近年来 LLM 在测试生成领域展现了显著潜力:

- **Meta ACH** (FSE 2025 Industry): mutation-guided LLM-based test generation
- **PRIMG** (EASE 2025): mutant prioritization + 增量自适应测试
- **STING** (arXiv 2025): mutation-guided test strengthening

但这些工作全部针对**传统软件**，**没有任何工作将 LLM 辅助的 mutation-guided 测试生成应用于 GPU CUDA kernel 源码级变异体**。

### 1.5 Phase II 解决的关键问题

Phase II 是（据我们所知）**首个专门针对 GPU CUDA kernel 源码级 survived mutant 的多维度自动化增强测试系统**，它提供:

1. **5 种确定性测试维度**（论文最终方法）+ 1 种 LLM 迭代分析（早期内嵌，**已从最终方法移除**，见顶部说明），覆盖不同杀伤路径
2. **算子定向输入策略** (STRATEGY_MAP)，21 种 stress policy 映射到 16 类 CUDA 变异算子
3. **跨维度不早停**设计，获取完整多维度敏感性画像
4. **双轨道架构** (Main Track + Config-Stress Track)，分离 fixed-shape 与 batch-variation 实验
5. **LLM 迭代修复机制**，将 LLM 与 EMD 证据链和确定性测试结果深度整合

---

## 2. Phase I → Phase II 衔接

### 2.1 Phase I (EMD) 产出

Phase I 对每个变异体执行四层等价变异体检测 (EMD):

| 层级 | 功能 | 比较方式 |
|------|------|----------|
| Layer 0 | CUDA-aware 源码规范化 | 文本相等 |
| Layer 1 | 算子特定静态等价规则 (4 条: `boundary_unreachable`, `dead_write`, `mask_noreach`, `dead_host_constant`) | 规则匹配 |
| Layer 2 | 动态 bitwise 检测 (最多 112 轮: 100 random + 6 policies × 2 seeds) | NaN-aware bitwise |
| Layer 3 | LLM 等价性验证 (DeepSeek) | LLM 判断 |

Layer 2 具体执行:
- **随机阶段**: 100 轮 (seed 10000~10099)，每轮用 `get_inputs()` 生成随机输入，bitwise 比较 original 与 mutant 输出
- **定向压力阶段**: 每个算子对应 6 个定向 stress policy (来自 `OPERATOR_DIRECTED_POLICIES` 映射)，每个 policy 执行 2 个 seed = 12 轮
- 全部通过则 Layer 2 判定 `is_equivalent=True`，进入 Layer 3 LLM 审核
- **提前退出**: 任一轮发现 bitwise 差异、或连续 infra error 超过阈值时，Layer 2 提前退出并标记 `is_equivalent=False`。此时 `tested_policies` 可能不完整。

**Phase I 最终产出 5 种状态**: killed (939)、survived (270)、candidate_equivalent (264)、strict_equivalent (10)、stillborn (163)。总计 1646 个变异体。

### 2.2 Phase II 入选变异体

Phase II 处理的是 Phase I 中 **survived (270) + candidate_equivalent (264) = 534 个变异体**。

每个变异体携带的关键数据字段:
```
equiv_detail:
├── layer0: cuda_strings_equal, python_host_equal, mutation_domain, cuda_diff_lines
├── layer1: rule_hit, rules_checked, rule_details
├── layer2: is_equivalent, total_rounds, equiv_runs,
│           tested_random_seeds (list), tested_policies (list of {name, status}),
│           first_input_summary, last_input_summary,
│           divergence (如果有: {round_type, seed, policy})
├── layer3: verdict, confidence, reasoning, kill_strategy,
│           suggested_test {description, python_code} | null,
│           input_spec, reason_category, proof_sketch
└── 顶级: input_spec, kernel_name, problem_file
```

---

## 3. 系统架构总览

### 3.1 全局流水线

```
Phase I (EMD) ──JSON──→ Phase II (Enhanced Testing)
                        ├── Step 1-2: 确定性多维度测试 (5 个维度)
                        │   ├── Main Track (fixed-shape): value_stress, dtype_stress, repeated_run, training_stress
                        │   └── Config Track (batch-variation): config_stress
                        ├── Step 3: LLM 迭代分析 (仅当 Step 1-2 全部未杀死时触发)
                        └── Step 4: 后处理 (算子×维度矩阵, 覆盖建议, 最终报告)
```

### 3.2 模块依赖

| 模块 | 文件 | 职责 |
|------|------|------|
| 主编排器 | `run_stress_enhance.py` | Tier 分类、维度调度、LLM 调用、结果聚合 |
| 子进程 Worker | `_stress_worker.py` | GPU 隔离执行 6 种测试模式 |
| 策略库 | `policy_bank.py` | 21 种输入策略的实现 |
| 策略映射 | `enhanced_inputs.py` | STRATEGY_MAP: 算子→优先策略 |
| 数据模型 | `differential_tester.py` | StressTestResult, StressSummary |
| LLM 分析器 | `llm_analyzer.py` | 提示词构建、响应解析 |

### 3.3 子进程隔离架构

**所有 CUDA 操作在子进程中执行**，主进程不做任何 GPU 计算。原因是: CUDA JIT 编译可能导致进程 hang、GPU 状态污染或内存泄漏；子进程隔离确保单个变异体的测试失败不影响全局。

执行流程:
1. 主进程创建临时 JSON 配置文件
2. `subprocess.Popen` 启动 `_stress_worker.py`，设 `start_new_session=True` 以便超时时可整组杀死
3. 超时处理: `proc.communicate(timeout)`，超时则 `os.killpg(SIGKILL)`
4. 读取结果 JSON 返回，清理临时文件

**超时设置** (`STRESS_TIMEOUT = 180s` 为基准):

| 测试维度 | 单次 worker 调用超时 | 倍率 | 主进程串行调用次数 | 说明 |
|----------|---------------------|------|-------------------|------|
| value_stress | 180 s | ×1 | 每策略×每seed 调用 1 次 (Tier 1 最多 63 次) | 每次 worker 执行 1 个 policy×1 seed |
| dtype_stress | 360 s | ×2 | 3 次 (3 seeds, 每次内部循环 2 dtype) | 低精度编译可能较慢 |
| repeated_run | 540 s | ×3 | 3 次 (3 seeds, 每次内部循环 10 trials) | worker 内部重复执行 10 次 |
| training_stress | 180 s | ×1 | 每策略×每seed 调用 1 次 (最多 63 次) | 与 value_stress 结构相同 |
| config_stress | 540 s | ×3 | **1 次** (worker 内部循环 7 batch_size × 3 seeds) | 全部组合在单次 worker 中完成 |
| llm_verify | 180 s | ×1 | 最多 3 次 (每轮 LLM 后执行 1 次) | LLM 建议代码执行 |

> **平台说明**: 子进程隔离依赖 POSIX API (`os.killpg`, `os.getpgid`, `start_new_session=True`, `signal.SIGKILL`)，实验在 Linux 环境下运行。

伪代码:

```
function RunStressWorker(cfg, timeout):
    cfg_path ← CreateTempJSON(cfg)
    res_path ← CreateTempJSON()
    proc ← Popen(["python", "_stress_worker.py", cfg_path, res_path], new_session=True)
    try:
        proc.communicate(timeout=timeout)
    except TimeoutExpired:
        KillProcessGroup(proc.pid)
        return NULL  // 超时
    if FileExists(res_path) and FileSize(res_path) > 2:
        return ReadJSON(res_path)
    return NULL
```

---

## 4. Tier 分类与执行策略

### 4.1 三层难度分类

每个变异体根据 Phase I EMD 结果分入三个 Tier:

```
function ClassifyTier(mutant):
    if mutant.status == "candidate_equivalent":
        return Tier 3
    if mutant.equiv_detail.layer2.is_equivalent == false:
        return Tier 1  // Layer 2 发现了 bitwise 差异
    if mutant.equiv_detail.layer3.verdict == "possibly_killable":
        return Tier 2  // 112 轮 bitwise 一致，但 LLM 认为可杀
    return Tier 2      // 默认回退
```

| Tier | 含义 | 来源 | 数量 |
|------|------|------|------|
| Tier 1 | Layer 2 拒绝等价 — 存在 bitwise 差异但 Phase I 的 allclose 容差下未杀死 | survived 中 `layer2.is_equivalent=false` | 151 |
| Tier 2 | 112 轮全部 bitwise 一致，但 LLM 认为可杀 或默认回退 | survived 中 Layer 2 通过 | 119 |
| Tier 3 | Candidate Equivalent — Layer 2 + Layer 3 双重确认等价 | candidate_equivalent | 264 |

### 4.2 Tier 3 子集筛选

最终版本中 Tier 3 的筛选函数包含 10 个算子的白名单:

```
function ShouldChallengeTier3(mutant):
    confidence ← mutant.equiv_detail.layer3.confidence (默认 1.0)
    if confidence < 0.98:
        return True  // LLM 置信度不足
    op ← mutant.operator_name
    if op ∈ {sync_remove, launch_config_mutate, mask_boundary, index_replace,
             relop_replace, const_perturb, arith_replace, cast_remove,
             init_modify, scale_modify}:
        return True  // 这些算子有配置/数值敏感的可能
    return False
```

> **注意**: `confidence` 来自 Phase I Layer 3 LLM 等价性验证 (`build_equiv_verify_prompt`) 的输出字段，不是 Phase II 的产出。
>
> **实际执行结果**: 全部 264 个 candidate_equivalent 变异体均满足上述双条件之一。其中 260 个命中 10 算子白名单 (arith_replace 29、relop_replace 61、const_perturb 50、index_replace 41、mask_boundary 48、launch_config_mutate 7、sync_remove 10、cast_remove 10、scale_modify 1、init_modify 3)，另有 4 个不在白名单 (epsilon_modify 2、broadcast_unsafe 2) 但因 `confidence < 0.98` 而通过第一条分支。因此 534 个变异体全部进入增强测试。

### 4.3 执行优先级

```
execution_order = Tier 1 全部(151) → Tier 2 全部(119) → Tier 3 全部(264)
```

支持**断点续跑**: `completed.json` 记录已完成的 mutant_id，重启后自动跳过。

---

## 5. 与 Phase I 的去重机制

### 5.1 去重的必要性与合理性

Phase I Layer 2 对每个变异体已经执行了最多 112 轮检测（100 random + 6 policies × 2 seeds）。Phase II 的 `value_stress` 维度需要排除 Layer 2 **已用的定向策略**，优先将测试预算分配给全新策略。

**Phase I Layer 2 与 Phase II value_stress 的比较方式不同**:

| 对比维度 | Phase I Layer 2 | Phase II value_stress |
|----------|----------------|----------------------|
| 比较方式 | **NaN-aware bitwise** (逐字节精确比较) | **allclose** (atol=1e-2, rtol=1e-2, 容差比较) |
| 比较对象 | original_kernel vs mutant_kernel | ref(PyTorch) / original / mutant 三路比较 |
| 判杀标准 | 任一 bit 不同 → 差异 | `original_ok ∧ ¬mutant_ok` → killed |
| 辅助判杀 | — | bitwise 差异作为补充判杀 (allclose 通过但 bitwise 不同) |

**去重合理性**: 虽然比较方式不同，但 **bitwise 是比 allclose 更严格的标准**。如果 Layer 2 用某策略的 bitwise 比较都没有发现差异 (原始 kernel 和变异体逐位相同)，那么在更宽松的 allclose 容差下，同一策略更不可能发现差异。因此，跳过这些策略不会遗漏 kill 机会，反而能将有限的测试预算投入到 **Layer 2 从未测试过的 ~15 个新策略**上。

**例外情况**: Tier 1 变异体在 Layer 2 中发现了 bitwise 差异但在 Phase I 的 allclose 下未被杀死。此类变异体的处理通过 **tier1_replay** (§15) 专门回放 Layer 2 记录的 divergence seed + policy，在 Phase II 的 allclose 容差下重新判定。这比在 value_stress 中重复测试所有已测策略更高效。

### 5.2 动态策略去重实现

```
function GetNewPolicies(operator_name, all_21_policies, mutant_meta):
    // 1. 从 equiv_detail 动态读取 Layer 2 实际执行过的策略
    tested ← ∅
    for p in mutant_meta.equiv_detail.layer2.tested_policies:
        if p is dict: tested.add(p.name)
        elif p is str: tested.add(p)

    // 2. STRATEGY_MAP 中该算子的优先策略排最前
    mapped ← STRATEGY_MAP[operator_name]  // 如 epsilon_modify → [near_zero, denormals, dense_nonzero]
    mapped_new ← [p for p in mapped if p ∉ tested]

    // 3. 剩余策略排在后面
    remaining ← [p for p in all_21_policies if p ∉ tested and p ∉ mapped]

    return mapped_new + remaining  // 优先策略在前
```

**关键细节**:
- **Tier 2/3**: Layer 2 正常完成了全部 112 轮，`tested_policies` 包含 6 个策略 → 去重后剩余 ~15 个新策略
- **Tier 1**: Layer 2 **未完成全部 112 轮**，`tested_policies` 可能为空或部分填充。原因有两种:
  - (a) Layer 2 在某轮发现 bitwise 差异后立即退出，此时 `tested_policies` 通常含若干策略但少于 6 个;
  - (b) Layer 2 worker 超时或崩溃 (`equiv_detail.layer2.error = "worker_timeout_or_crash"`)，此时 `tested_policies` 可能为空。
  两种情况都导致 `is_equivalent=False`，从而被分类为 Tier 1。
- 因此 Tier 1 的 value_stress 实际执行 **最多 21 策略 × 3 seeds = 63 轮**，Tier 2/3 约 **15 策略 × 3/5 seeds = 45~75 轮**
- `tested_policies` 字段格式为 `[{"name": "...", "status": "passed|infra_error|..."}, ...]` 或纯字符串列表，`_get_new_policies` 函数兼容两种格式

### 5.3 Layer 2 算子定向策略映射 (代码: `OPERATOR_DIRECTED_POLICIES`)

| 算子 | Layer 2 定向策略 (6 个/算子) |
|------|-----|
| `relop_replace` | relop_boundary_hit, boundary_last_element, structured_ramp, near_zero, sparse, large_magnitude |
| `arith_replace` | extreme_magnitude, large_magnitude, near_zero, all_negative, sparse, boundary_last_element |
| `epsilon_modify` | near_epsilon, near_zero, denormals, large_magnitude, sparse, boundary_last_element |
| `mask_boundary` | boundary_last_element, structured_ramp, head_heavy, tail_heavy, sparse, large_magnitude |
| `index_replace` | head_heavy, tail_heavy, structured_ramp, large_magnitude, sparse, boundary_last_element |
| `sync_remove` | structured_ramp, head_heavy, tail_heavy, large_magnitude, sparse, boundary_last_element |
| `const_perturb` | near_zero, boundary_last_element, sparse, large_magnitude, structured_ramp, all_negative |
| `launch_config_mutate` | structured_ramp, head_heavy, tail_heavy, large_magnitude, sparse, boundary_last_element |
| `stab_remove` | extreme_magnitude, large_magnitude, all_positive, near_zero, sparse, boundary_last_element |
| `scale_modify` | extreme_magnitude, large_magnitude, near_zero, all_negative, sparse, boundary_last_element |
| `acc_downgrade` | reduction_adversarial, large_magnitude, mixed_extremes, alternating_sign, sparse, boundary_last_element |
| `reduction_reorder` | reduction_adversarial, mixed_extremes, alternating_sign, large_magnitude, sparse, boundary_last_element |
| `init_modify` | init_sensitive, all_negative, all_positive, sparse, near_zero, boundary_last_element |
| `cast_remove` | extreme_magnitude, near_zero, mixed_extremes, large_magnitude, sparse, boundary_last_element |
| `broadcast_unsafe` | large_magnitude, sparse, mixed_extremes, near_zero, structured_ramp, boundary_last_element |
| `layout_assume` | structured_ramp, alternating_sign, large_magnitude, near_zero, sparse, boundary_last_element |

### 5.4 Phase II 增量维度一览

以下是 Phase I Layer 2 **完全没做过**的测试维度:

| 增量维度 | 内容 | 价值 |
|----------|------|------|
| Layer 2 未使用的 ~15 个 stress policy | near_overflow, denormals, all_positive, mixed_extremes, alternating_sign, uniform_constant, init_sensitive, reduction_adversarial, dense_nonzero, sparse_extreme 等 | 全新数值分布 |
| `dtype_stress` | float16/bfloat16 精度切换 | 全新精度环境 |
| `repeated_run` | 同一输入跑 10 次 × 3 seeds | 检测非确定性 |
| `training_stress` | 模型 `.train()` 模式 | 全新执行模式 |
| `config_stress` | 变 batch_size (7 种) | 全新并行配置 |
| LLM 迭代分析 | 最多 3 轮 LLM 生成定向输入 | 语义定向测试 |

---

## 6. 输入策略库与维度-策略关系 (policy_bank.py)

### 6.0 策略库与五个测试维度的关系

21 个 stress policy **仅被 `value_stress` 和 `training_stress` 两个维度使用**，其余三个维度有独立的输入生成机制，不使用 policy_bank:

| 测试维度 | 是否使用 21 策略 | 输入生成方式 | 策略选择 |
|----------|:---------------:|-------------|---------|
| **value_stress** | **是** | `policy_bank.STRESS_POLICIES[name](get_inputs(), seed)` | 去重后 ~15-21 策略 × 3/5 seeds |
| **training_stress** | **是** | 同上，但模型使用 `.train()` 模式 | **全部 21 策略** (不去重) × 3 seeds |
| **dtype_stress** | 否 | `get_inputs()` → cast 到 float16/bfloat16 | 不涉及策略，改变 **dtype** |
| **repeated_run** | 否 | `get_inputs()` 固定 seed，同一输入跑 10 次 | 不涉及策略，改变 **执行次数** |
| **config_stress** | 否 | `get_inputs()` → `_rebatch_inputs(bs)` 变 batch_size | 不涉及策略，改变 **batch 配置** |

> **总结**: Policy Bank 的 21 个策略是 **数值分布增强** (value augmentation) 工具，它们保持 shape/dtype 不变，仅改变 tensor 内的数值。其余三个维度分别从 **精度** (dtype)、**重复性** (execution)、**配置** (configuration) 三个正交方向进行增强，与 policy bank 无交集。

### 6.1 策略设计原理与文献依据

21 个策略**不是凭空设计的**，而是来源于以下四类测试理论和实证研究：

**来源 1: 浮点测试 (Floating-Point Testing)**
- FPGen [ICSE 2020] 提出 GPU 浮点程序的边界值覆盖方法，指出 **极小值 (near-zero)、极大值 (near-overflow)、非规格化数 (denormals)** 是暴露数值路径差异的关键分布。
- Laguna et al. [SC Workshop 2024] 在 GPU 数值差分测试中使用 **大幅度输入** 和 **混合极端值** 检测 NVIDIA 与 AMD GPU 间的数值差异。
- 据此设计: `near_zero`, `denormals`, `near_overflow`, `large_magnitude`, `extreme_magnitude`, `near_epsilon`, `mixed_extremes`

**来源 2: 边界值分析 (Boundary Value Analysis, BVA)**
- 经典软件测试理论 [Myers 1979; Offutt & Liu, TSE 1999]，要求在输入域边界处进行测试。
- 对 CUDA kernel 的特殊化: `boundary_last_element` (off-by-one 边界)、`relop_boundary_hit` (关系运算决策边界)、`sparse_extreme` (稀疏度极端边界)
- 据此设计: `boundary_last_element`, `relop_boundary_hit`, `all_negative`, `all_positive`, `sparse`, `sparse_extreme`, `dense_nonzero`

**来源 3: GPU 并行语义测试 (GPU Parallel Semantics Testing)**
- Mu2 [ISSTA 2023] 发现 GPU kernel 的线程索引变异（如 `blockIdx.x` → `blockIdx.y`）在 1-D launch 配置下会退化为相同值，导致变异体存活。需要 **结构化输入** (structured_ramp) 使不同位置的值可区分，以及 **头/尾偏重** 输入使索引退化可观测。
- 据此设计: `structured_ramp`, `head_heavy`, `tail_heavy`

**来源 4: 数值累加/归约误差 (Reduction Error)**
- Higham [Accuracy and Stability of Numerical Algorithms, 2002] 指出浮点求和的误差与数值分布强相关。**交替正负大数** 可最大化归约重排序引入的误差。
- 变异测试中的 `reduction_reorder` 和 `acc_downgrade` 算子改变了求和顺序或精度，标准 `randn` 因正负抵消而掩盖差异。
- 据此设计: `alternating_sign`, `reduction_adversarial`

**来源 5: 变异体存活机制的实证分析**
- 我们对 Phase I 中 534 个存活/候选等价变异体的 pilot 分析发现:
  - `epsilon_modify` (eps: 1e-5→0) 在 eval 模式下存活，因为 BatchNorm 的 `running_var=1.0` 使 `rsqrt(1.0+0) ≈ rsqrt(1.0+1e-5)` [Magneto, ISSTA 2021 的 BN 分析]
  - `init_modify` (min_val 初始化 0→1e10) 在全正输入下被掩盖
  - `const_perturb` 的微扰在 tolerance 范围内被吸收
  - 据此设计: `uniform_constant`, `init_sensitive`

### 6.2 策略总表 (21 种)

所有策略保持 **shape 和 dtype 不变**，仅改变数值分布。仅被 `value_stress` 和 `training_stress` 使用。

21 策略按设计来源分为 5 族:

**族 1: 数值极端分布 (来自浮点测试理论)**

| 策略名 | 数值特征 | 目标变异行为 | 文献依据 |
|--------|----------|-------------|---------|
| `large_magnitude` | randn × 1000 | 算术溢出路径分叉 | FPGen [ICSE'20] |
| `extreme_magnitude` | randn × 1e6 | 更激进的溢出暴露 | FPGen [ICSE'20] |
| `near_overflow` | randn × dtype上界 (fp32~1e30, fp16~60000) | 精度极限处的路径差异 | FPGen [ICSE'20] |
| `near_zero` | randn × 1e-7 | epsilon 相关分支 (÷0, rsqrt) | FPGen [ICSE'20] |
| `denormals` | randn × 1e-38 | 非正规数处理差异 | FPGen [ICSE'20] |
| `near_epsilon` | [1e-7, 1e-5] 均匀分布 | epsilon 判定边界 | FPGen [ICSE'20] |
| `mixed_extremes` | 50%×10000 + 50%×0.0001 | 极端混合暴露精度路径 | Laguna [SC'24] |

**族 2: 边界值与符号分布 (来自 BVA)**

| 策略名 | 数值特征 | 目标变异行为 | 文献依据 |
|--------|----------|-------------|---------|
| `all_negative` | -\|randn\| × 100 | 暴露 init_modify (0→正数) 的掩盖效应 | BVA |
| `all_positive` | \|randn\| × 100 | 全正值对照 | BVA |
| `boundary_last_element` | randn + 末位=1e4 | off-by-one 边界 (mask_boundary) | BVA |
| `relop_boundary_hit` | arange % 10 (整数值) | 关系运算判定边界 (< vs <=) | BVA |

**族 3: 稀疏性梯度 (来自 BVA + 实证分析)**

| 策略名 | 数值特征 (零值比例) | 目标变异行为 | 文献依据 |
|--------|-------------------|-------------|---------|
| `dense_nonzero` | \|randn\| + 1.0 (0% 零) | 消除零值掩盖算术差异 | Pilot 实证 |
| `sparse` | 90% 零 + 10% randn×100 | 常规稀疏激活 | BVA |
| `sparse_extreme` | 99% 零 + 1% ×1e4 | 极端稀疏边界 | BVA |

**族 4: 结构化/位置敏感 (来自 GPU 并行语义)**

| 策略名 | 数值特征 | 目标变异行为 | 文献依据 |
|--------|----------|-------------|---------|
| `structured_ramp` | [0, 1/n, 2/n, ...] | 使不同位置值可区分，暴露索引退化 | Mu2 [ISSTA'23] |
| `head_heavy` | 前25%极端+其余近零 | 暴露 index 退化只处理头部 (blockIdx.y=0) | Mu2 [ISSTA'23] |
| `tail_heavy` | 后25%极端+其余近零 | 暴露 index 退化跳过尾部 | Mu2 [ISSTA'23] |

**族 5: 归约/累加对抗 + 特殊行为 (来自数值分析 + 实证)**

| 策略名 | 数值特征 | 目标变异行为 | 文献依据 |
|--------|----------|-------------|---------|
| `alternating_sign` | 交替 +/- × 100 | 求和顺序敏感 (reduction_reorder) | Higham [2002] |
| `reduction_adversarial` | 交替 +1e4/-1e4 + 微噪声 | 最大化 FP 归约误差 | Higham [2002] |
| `uniform_constant` | 全 88.0 | 暴露平移不变假设 (scale_modify) | Magneto [ISSTA'21] |
| `init_sensitive` | 随机全正或全负 | 暴露 min/max 初始化差异 | Pilot 实证 |

### 6.3 各维度的输入策略汇总表

| 策略 / 维度 | value_stress | training_stress | dtype_stress | repeated_run | config_stress |
|:------------|:---:|:---:|:---:|:---:|:---:|
| **策略来源** | 21 策略 (去重后 ~15-21) | 全部 21 策略 | — | — | — |
| **输入变化方式** | 改数值分布 | 改数值分布 + .train() | 改 dtype | 同一输入多次 | 改 batch_size |
| `large_magnitude` | ✓ | ✓ | — | — | — |
| `extreme_magnitude` | ✓ | ✓ | — | — | — |
| `near_overflow` | ✓ | ✓ | — | — | — |
| `near_zero` | ✓ | ✓ | — | — | — |
| `denormals` | ✓ | ✓ | — | — | — |
| `near_epsilon` | ✓ | ✓ | — | — | — |
| `mixed_extremes` | ✓ | ✓ | — | — | — |
| `all_negative` | ✓ | ✓ | — | — | — |
| `all_positive` | ✓ | ✓ | — | — | — |
| `boundary_last_element` | ✓ | ✓ | — | — | — |
| `relop_boundary_hit` | ✓ | ✓ | — | — | — |
| `dense_nonzero` | ✓ | ✓ | — | — | — |
| `sparse` | ✓ | ✓ | — | — | — |
| `sparse_extreme` | ✓ | ✓ | — | — | — |
| `structured_ramp` | ✓ | ✓ | — | — | — |
| `head_heavy` | ✓ | ✓ | — | — | — |
| `tail_heavy` | ✓ | ✓ | — | — | — |
| `alternating_sign` | ✓ | ✓ | — | — | — |
| `reduction_adversarial` | ✓ | ✓ | — | — | — |
| `uniform_constant` | ✓ | ✓ | — | — | — |
| `init_sensitive` | ✓ | ✓ | — | — | — |
| float16 cast | — | — | ✓ | — | — |
| bfloat16 cast | — | — | ✓ | — | — |
| 同一输入 ×10 重复 | — | — | — | ✓ | — |
| batch_size ∈ {1,2,4,8,16,32,64} | — | — | — | — | ✓ |
| **每变异体轮次** | 45~75 轮 | 最多 63 轮 | 6 轮 | 30 轮 | 21 组合 |

> **关键区分**:
> - `value_stress` 与 `training_stress` 使用**相同的策略库**，区别在于模型模式 (`.eval()` vs `.train()`) 和去重逻辑 (value_stress 去除 Phase I Layer 2 已测策略; training_stress 不去重，因为 `.train()` 是全新执行模式)。
> - `dtype_stress`、`repeated_run`、`config_stress` **不使用** policy_bank 的 21 策略，它们从不同的正交维度增强测试：精度降级、非确定性检测、配置变异。

### 6.4 算子→策略优先映射 (STRATEGY_MAP, enhanced_inputs.py)

此映射用于 `value_stress` 和 `training_stress` 的策略排序——**推荐策略排在最前**。映射的依据是§6.1 中的文献分析和存活机制 pilot 实证。

> **优先排序的双重作用**:
> - **效率优化**: `value_stress` 和 `training_stress` 都有**维度内早停** (§8.1)，即同一维度内一旦杀死即停止该维度剩余轮次。STRATEGY_MAP 使最可能杀死该算子的策略最先执行，减少不必要的 worker 调用。
> - **不影响完整性**: 跨维度**不早停** — 即使 `value_stress` 已杀死变异体，后续的 `dtype_stress`、`repeated_run` 等维度仍然完整执行，确保获得全维度敏感性画像用于算子×维度交叉分析 (§21.1)。因此优先排序仅节省**单维度内**的开销，不减少维度覆盖。

| 算子 | 推荐策略 (STRATEGY_MAP) | 存活机制 | 依据 |
|------|------------------------|---------|------|
| `epsilon_modify` | `near_zero`, `denormals`, `dense_nonzero` | eval 模式 var≈1 掩盖 eps 差异 | FPGen + Magneto |
| `scale_modify` | `uniform_constant`, `structured_ramp` | running_var=1.0 消除 scale 差异 | Magneto [ISSTA'21] |
| `stab_remove` | `large_magnitude`, `near_overflow` | randn ∈[-3,3] 不触发溢出保护路径 | FPGen [ICSE'20] |
| `cast_remove` | `near_overflow`, `large_magnitude` | float32 下 static_cast 恒等 | FPGen |
| `init_modify` | `all_negative`, `sparse` | 正值 randn 掩盖 min_val=0 初始化差异 | BVA |
| `acc_downgrade` | `mixed_extremes`, `large_magnitude` | 小值累加精度损失不可观测 | Higham [2002] |
| `reduction_reorder` | `mixed_extremes`, `alternating_sign` | 随机值正负抵消，求和顺序差异消失 | Higham [2002] |
| `broadcast_unsafe` | `structured_ramp` | 对称值掩盖广播方向错误 | Mu2 [ISSTA'23] |
| `layout_assume` | `structured_ramp` | 连续内存布局恰好满足假设 | Mu2 [ISSTA'23] |
| `index_replace` | `structured_ramp`, `large_magnitude`, `head_heavy`, `tail_heavy` | 1-D launch 使 blockIdx.y/z=0 退化 | Mu2 [ISSTA'23] |
| `mask_boundary` | `boundary_last_element`, `sparse`, `sparse_extreme` | 默认输入不触及边界条件 | BVA |
| `sync_remove` | `large_magnitude`, `mixed_extremes` | 小值下竞态条件差异被容差吸收 | — |
| `launch_config_mutate` | `structured_ramp`, `large_magnitude` | grid-stride 循环吸收配置差异 | Mu2 [ISSTA'23] |
| `arith_replace` | `large_magnitude`, `mixed_extremes`, `dense_nonzero` | 小值下 +/-/×/÷ 差异不可观测 | FPGen + BVA |
| `relop_replace` | `boundary_last_element`, `structured_ramp`, `sparse_extreme` | randn 值不落在关系运算判定边界 | BVA |
| `const_perturb` | `near_zero`, `large_magnitude` | 微扰被 allclose 容差吸收 | Magneto [ISSTA'21] |

```python
STRATEGY_MAP = {
    "epsilon_modify":       ["near_zero", "denormals", "dense_nonzero"],
    "scale_modify":         ["uniform_constant", "structured_ramp"],
    "stab_remove":          ["large_magnitude", "near_overflow"],
    "cast_remove":          ["near_overflow", "large_magnitude"],
    "init_modify":          ["all_negative", "sparse"],
    "acc_downgrade":        ["mixed_extremes", "large_magnitude"],
    "reduction_reorder":    ["mixed_extremes", "alternating_sign"],
    "broadcast_unsafe":     ["structured_ramp"],
    "layout_assume":        ["structured_ramp"],
    "index_replace":        ["structured_ramp", "large_magnitude", "head_heavy", "tail_heavy"],
    "mask_boundary":        ["boundary_last_element", "sparse", "sparse_extreme"],
    "sync_remove":          ["large_magnitude", "mixed_extremes"],
    "launch_config_mutate": ["structured_ramp", "large_magnitude"],
    "arith_replace":        ["large_magnitude", "mixed_extremes", "dense_nonzero"],
    "relop_replace":        ["boundary_last_element", "structured_ramp", "sparse_extreme"],
    "const_perturb":        ["near_zero", "large_magnitude"],
}
```

---

## 7. 三路差分比较核心机制

Phase II 所有维度共享统一的**三路差分比较**框架。理解此框架是理解后续各维度的前提。

### 7.1 三路比较流程

```
function ThreeWayCompare(stress_inputs, ref_model, orig_model, mut_model, atol, rtol):
    // Step 1: 执行参考实现
    ref_out ← ref_model(stress_inputs)
    ref_nan ← HasNaNInf(ref_out)

    // Step 2: 执行原始 kernel (未变异版本)
    orig_out ← orig_model(stress_inputs)

    // Step 3: 确定比较目标 (ref NaN fallback 机制)
    if ref_nan:
        if HasNaNInf(orig_out):
            return SKIP  // 参考和原始都 NaN/Inf, 输入无效
        compare_target ← orig_out  // 退化为 orig vs mut 比较
        original_ok ← True
    else:
        original_ok ← AllClose(ref_out, orig_out, atol, rtol)
                       AND NOT HasNaNInf(orig_out)
        compare_target ← ref_out

    // Step 4: 执行变异体 kernel
    mutant_ok ← AllClose(compare_target, mut_out, atol, rtol)
                AND NOT HasNaNInf(mut_out)

    // Step 5: 判杀
    if original_ok AND NOT mutant_ok:
        return KILLED
    return SURVIVED
```

### 7.2 AllClose 定义

```
AllClose(a, b, atol=1e-2, rtol=1e-2):
    return torch.allclose(a.float().cpu(), b.float().cpu(), atol=atol, rtol=rtol)
```

### 7.3 NaN-aware Bitwise 比较

仅在 `value_stress` 中作为辅助判杀使用:

```
function BitwiseEqual(a, b):
    if a.shape ≠ b.shape or a.dtype ≠ b.dtype: return False
    if a is floating_point:
        nan_a ← isnan(a); nan_b ← isnan(b)
        if nan_a ≠ nan_b: return False
        mask ← ¬nan_a
        if any(mask):
            return a[mask].as_bytes() == b[mask].as_bytes()
        return True
    return torch.equal(a, b)
```

### 7.4 Ref NaN/Inf Fallback 机制

当参考实现 (PyTorch reference) 在极端输入下产生 NaN/Inf 时:
1. 检查原始 kernel 输出是否也为 NaN/Inf
2. 若原始 kernel 输出有效 → 将原始 kernel 输出作为 compare_target 代替 ref
3. 若原始 kernel 也 NaN/Inf → 跳过该轮测试 (标记为 `ref_nan_fallback`)

> **有效性威胁**: 此机制引入假设"原始 kernel 在该输入下的输出是正确的"。实际数据中 139/534 个变异体 (26.0%) 至少一次触发此回退，累计 1303 次。

---

## 8. 测试维度详细设计

### 8.1 设计原则

- **统一覆盖**: 所有 Tier 都运行全部 5 个确定性维度（LLM 迭代分析层已从论文最终方法移除，仅作历史留档；早期为 5 确定性 + 1 LLM 共 6 个）
- **双轨道结构**:
  - **主轨道 (Main Track)**: 严格 fixed-shape，只改值。包含 value_stress、dtype_stress、repeated_run、training_stress
  - **附加轨道 (Config Track)**: 允许变 batch_size。包含 config_stress
- **跨维度不早停**: 即使某个维度已杀死，后续维度仍然执行（为算子×维度交叉分析提供完整数据）
- **维度内保留早停**: 同维度内杀死即跳过剩余轮次（同类证据无增量价值）
- **Tier 差异仅体现在**: 执行优先级顺序、Tier 1 的 replay 步骤、value_stress 的 seeds 强度

### 8.2 各 Tier 维度执行顺序

**Tier 1** (已知 bitwise 差异，需放大到 allclose 失败):
```
主轨道: tier1_replay (见 §15) → value_stress → dtype_stress → training_stress → repeated_run
附加轨道: config_stress
```

**Tier 2** (112 轮 bitwise 一致，需新维度突破):
```
主轨道: dtype_stress → training_stress → value_stress → repeated_run
附加轨道: config_stress
```

**Tier 3** (高度疑似等价，最高强度挑战):
```
主轨道: dtype_stress → value_stress(5seeds) → repeated_run → training_stress
附加轨道: config_stress
```

### 8.3 总测试预算

| 场景 | value_stress | dtype_stress | repeated_run | training_stress | config_stress | 合计 |
|------|-------------|-------------|-------------|----------------|--------------|------|
| Tier 1 (适用 training) | 最多 63 轮 | 6 轮 | 30 轮 | 最多 63 轮 | 21 轮 | ~183 轮 |
| Tier 1 (不适用 training) | 最多 63 轮 | 6 轮 | 30 轮 | 跳过 | 21 轮 | ~120 轮 |
| Tier 2/3 (适用 training) | 45~75 轮 | 6 轮 | 30 轮 | 最多 63 轮 | 21 轮 | ~165~195 轮 |
| Tier 2/3 (不适用 training) | 45~75 轮 | 6 轮 | 30 轮 | 跳过 | 21 轮 | ~102~132 轮 |

---

## 9. 维度 1: Value-Distribution Stress

### 9.1 目的

用 Phase I Layer 2 **未覆盖的** stress policy 生成极端数值输入，在 fixed-shape 条件下暴露变异差异。

### 9.2 执行流程

```
function RunValueStress(problem_file, kernel_code, mutated_code, policies, seeds_per_policy):
    consecutive_timeouts ← 0
    for each policy in policies:
        for si in 0..seeds_per_policy-1:
            seed ← 42 + policy_index × seeds_per_policy + si
            // 在子进程中执行:
            result ← Worker.value_stress(problem_file, kernel_code, mutated_code,
                                         policy, seed, atol=1e-2, rtol=1e-2)

            if result == TIMEOUT:
                consecutive_timeouts += 1
                if consecutive_timeouts ≥ 5:
                    ABORT dimension  // 跳过该维度剩余轮次，不跳过其它维度
                continue

            consecutive_timeouts ← 0

            // 判杀条件 A: allclose 失败
            if result.original_ok AND NOT result.mutant_ok:
                return KILLED(policy, seed, type="allclose")

            // 判杀条件 B: allclose 通过但 bitwise 不一致
            if result.original_ok AND result.mutant_ok AND NOT BitwiseEqual(orig_out, mut_out):
                return KILLED(policy, seed, type="bitwise")

    return SURVIVED
```

### 9.3 Worker 内部 (value_stress 模式)

```
function WorkerValueStress(cfg):
    // 1. 加载参考实现、原始 kernel、变异 kernel
    ref_model ← LoadReference(cfg.problem_file).eval()
    orig_model ← CompileFromSource(cfg.kernel_code).eval()  // 编译失败 → 返回 error
    mut_model ← CompileFromSource(cfg.mutated_code).eval()  // 编译失败 → 返回 error

    // 2. 生成 stress 输入
    torch.manual_seed(cfg.seed)
    template_inputs ← get_inputs()
    stress_inputs ← STRESS_POLICIES[cfg.policy_name](template_inputs, cfg.seed)

    // 3. 三路比较 (见 §7)
    //    异常处理: ref crash → ref_ok=false; orig compile → error; mut compile → error
    return ThreeWayCompare(stress_inputs, ref_model, orig_model, mut_model,
                           cfg.atol, cfg.rtol)
    // 额外返回: bitwise_orig_mut_eq = BitwiseEqual(orig_out, mut_out)
```

### 9.4 参数

| 参数 | Tier 1 | Tier 2 | Tier 3 |
|------|--------|--------|--------|
| 每策略 seeds 数 | 3 | 3 | **5** |
| 策略数 (去重后) | 最多 21 | ~15 | ~15 |
| 总轮次 | 最多 63 | ~45 | ~75 |

---

## 10. 维度 2: Dtype Stress (精度切换)

### 10.1 目的

将模型和输入 cast 到 float16/bfloat16，检测精度退化是否暴露变异差异。

### 10.2 执行流程

主进程为每个 seed 调用一次 worker（超时 360s），worker 内部循环遍历目标 dtype:

```
function RunDtypeStress(problem_file, kernel_code, mutated_code):
    for si in 0..2:  // 主进程: 3 seeds, 每 seed 调用 1 次 worker
        seed ← 42 + 100 + si
        result ← Worker.dtype_stress(problem_file, kernel_code, mutated_code,
                                      seed, target_dtypes=[float16, bfloat16])
        if result.killed:
            return KILLED(result.killing_dtype, seed)
    return SURVIVED
```

**Worker 内部逻辑** (`_stress_worker.py: run_dtype_stress`):

```
function WorkerDtypeStress(cfg):
    // 一次性构建三个模型 (初始 fp32)
    ref_model, orig_model, mut_model ← BuildModels().eval()
    base_inputs ← get_inputs()

    for dtype in [float16, bfloat16]:
        // 输入 cast: 仅对浮点型 tensor 执行, 整型/布尔型保持原类型
        cast_inputs ← [
            x.to(dtype=dtype) if x.is_floating_point else x
            for x in base_inputs
        ]

        // 模型 cast: nn.Module.to(dtype) 是 in-place 操作
        // 因此第二轮 (bfloat16) 时模型是从第一轮 (float16) 状态 cast 过来
        ref_m ← ref_model.to(dtype)
        orig_m ← orig_model.to(dtype)
        mut_m ← mut_model.to(dtype)

        ref_out ← ref_m(cast_inputs)
        ref_nan ← HasNaNInf(ref_out)

        orig_out ← orig_m(cast_inputs)

        // Ref NaN fallback (同 §7.4):
        if ref_nan:
            if HasNaNInf(orig_out):
                SKIP(reason="ref and orig both NaN/Inf")
                continue
            compare_target ← orig_out
            orig_ok ← True
        else:
            orig_ok ← AllClose(ref_out, orig_out, 1e-2, 1e-2) AND NOT HasNaNInf(orig_out)
            compare_target ← ref_out

        mut_out ← mut_m(cast_inputs)
        mut_ok ← AllClose(compare_target, mut_out, 1e-2, 1e-2) AND NOT HasNaNInf(mut_out)

        if orig_ok AND NOT mut_ok:
            return KILLED(dtype)

        // 优雅处理 "not implemented for 'half'" 类 RuntimeError → 跳过该 dtype
```

> **实现细节**: `nn.Module.to(dtype)` 返回 `self`（in-place 修改），因此循环中按 `[float16, bfloat16]` 顺序执行时，第二轮 bfloat16 测试是从 float16 状态 cast 到 bfloat16，而非从原始 float32 重新 cast。这不影响正确性（`to(bfloat16)` 的最终结果与从 fp32 直接 cast 一致），但需知晓此行为。

### 10.3 参数

- 3 seeds × 2 dtype = 6 轮
- 比较方式: allclose (atol=1e-2, rtol=1e-2)
- 对 float16/bfloat16 不支持的操作 (如某些 CUDA 算子) 优雅跳过

---

## 11. 维度 3: Repeated Run (非确定性检测)

### 11.1 目的

同一输入重复执行 mutant 多次，检测 data race 或 GPU 调度导致的非确定性行为。主要目标是 `sync_remove` 算子。

### 11.2 执行流程

```
function RunRepeatedRun(problem_file, kernel_code, mutated_code):
    for si in 0..2:  // 3 个 seed, 主进程串行调用 3 次 worker
        seed ← 42 + 200 + si
        // ═══ 以下为 Worker 内部逻辑 ═══

        ref_out ← ref_model(inputs)
        // ref NaN fallback: 若 ref NaN 且 orig 有效, 用 orig_out 替代 ref_out
        if HasNaNInf(ref_out):
            orig_out ← orig_model(inputs)
            if HasNaNInf(orig_out): return SKIP  // 两者都无效
            ref_out ← orig_out  // 退化为 orig vs mut

        mut_outputs ← []
        divergent_trial ← NULL

        for trial in 0..9:  // 10 次重复
            try:
                mut_out ← mut_model(inputs)
            except:
                return KILLED(seed, trial, type="mutant_crash")  // 崩溃立即返回

            // 记录首次发散 trial 索引, 但 **不立即返回, 继续跑完全部 10 次**
            if (NOT AllClose(ref_out, mut_out, 1e-2, 1e-2) OR HasNaNInf(mut_out))
               AND divergent_trial == NULL:
                divergent_trial ← trial

            mut_outputs.append(mut_out.float().cpu())

        // 判杀条件 A: 任一 trial 与 ref 不一致
        if divergent_trial ≠ NULL:
            return KILLED(seed, divergent_trial, type="diverged")

        // 判杀条件 B: 所有 trial 都与 ref 一致, 但 mutant 自身跨 trial 不一致
        for i in 1..len(mut_outputs)-1:
            if NOT AllClose(mut_outputs[0], mut_outputs[i], atol=1e-6, rtol=1e-6):
                return KILLED(seed, trial=i, type="self_inconsistent")

    return SURVIVED
```

> **设计要点**: 即使某个 trial 与 ref 不一致，worker 仍然跑完全部 10 次并收集所有 `mut_outputs`。这确保了 self-inconsistency 检查的统计完整性——只有当全部 10 次 trial 都通过了 ref 比较后，才会进入更严格的 self-inconsistency 检查 (atol=1e-6)。

### 11.3 参数

- 3 seeds × 10 trials = 30 轮
- 对 ref 的比较: allclose (atol=1e-2, rtol=1e-2)
- 自身一致性检查: allclose (atol=1e-6, rtol=1e-6) — 更严格

---

## 12. 维度 4: Training Stress (训练模式)

### 12.1 目的

将模型从 `.eval()` 切换到 `.train()` 模式。在 eval 模式下 BatchNorm/LayerNorm 使用 fixed running_var (通常为 1.0)，掩盖了 eps/scale 变异。在 train 模式下强制从 batch 统计计算，可暴露如 `rsqrt(tiny_var + 0)` vs `rsqrt(tiny_var + 1e-5)` 的差异。

### 12.2 仅对以下算子启用

```python
TRAINING_TARGET_OPS = {"epsilon_modify", "const_perturb", "init_modify",
                        "arith_replace", "cast_remove"}
```

对不适用的算子直接跳过，返回 `skipped_reason="operator_not_applicable"`。

### 12.3 执行流程

与 `value_stress` 几乎相同，关键区别:
- 模型使用 `.train()` 而非 `.eval()`
- **不做策略去重**: `.train()` vs `.eval()` 是不同执行模式，即使 Layer 2 已在 `.eval()` 下测过某策略，在 `.train()` 下仍有价值
- 使用**全部 21 个策略**，通过 STRATEGY_MAP 排序优先执行

```
function RunTrainingStress(problem_file, kernel_code, mutated_code, op, all_policies):
    if op ∉ TRAINING_TARGET_OPS:
        return SKIPPED("operator_not_applicable")

    // STRATEGY_MAP 优先排序
    mapped ← STRATEGY_MAP[op]
    remaining ← [p for p in all_policies if p ∉ mapped]
    priority_policies ← mapped + remaining  // 全部 21 策略

    for each policy in priority_policies:
        for si in 0..2:  // 3 seeds
            seed ← 42 + 300 + policy_index × 3 + si
            // Worker 内部: 三路比较，但 .train() 模式
            result ← Worker.training_stress(policy, seed)
            // 判杀: original_ok AND NOT mutant_ok → KILLED
            // 连续超时保护: ≥5 次 → ABORT dimension

    return SURVIVED
```

### 12.4 参数

- 全部 21 策略 × 3 seeds = 最多 63 轮
- 仅对 5 种适用算子执行
- 比较方式: allclose (atol=1e-2, rtol=1e-2)

---

## 13. 维度 5: Config Stress (配置压力，附加轨道)

### 13.1 目的

变化 batch_size 维度，检测 grid/block 计算边界、线程索引映射等配置敏感的变异差异。结论在论文中**单独报告**，不纳入主轨道 fixed-shape 结果。

### 13.2 执行流程

```
function RunConfigStress(problem_file, kernel_code, mutated_code):
    batch_sizes ← [1, 2, 4, 8, 16, 32, 64]
    seeds ← [42, 123, 7777]

    // 所有操作在单个 Worker 调用中完成 (超时 540s)
    ref_model, orig_model, mut_model ← BuildModels().eval()

    for bs in batch_sizes:
        for seed in seeds:
            torch.manual_seed(seed)
            base_inputs ← get_inputs()
            rebatched ← RebatchInputs(base_inputs, bs)

            ref_out ← ref_model(rebatched)
            ref_nan ← HasNaNInf(ref_out)

            orig_out ← orig_model(rebatched)  // 崩溃则 SKIP

            // ── 分支 A: ref 含 NaN/Inf ──
            if ref_nan:
                if HasNaNInf(orig_out):
                    SKIP  // ref 和 orig 都无效 → 标记 "ref_and_orig_nan_inf"
                // 注意: 此处 **不切换 compare_target 为 orig_out**
                // 后续仍然用 ref_out (含 NaN) 与 mut_out 做 allclose

            // ── 分支 B: ref 正常 ──
            else:
                orig_ok ← AllClose(ref_out, orig_out, 1e-2, 1e-2) AND NOT HasNaNInf(orig_out)
                if NOT orig_ok:
                    SKIP  // orig 本身在此配置下偏离 ref → 标记 "orig_diverges_from_ref"

            // ── mutant 执行与判杀 ──
            try:
                mut_out ← mut_model(rebatched)
            except:
                return KILLED(bs, seed, type="mutant_crash")

            // 关键: 始终用 ref_out 做比较 (而非 compare_target)
            mut_ok ← AllClose(ref_out, mut_out, 1e-2, 1e-2) AND NOT HasNaNInf(mut_out)
            if NOT mut_ok:
                return KILLED(bs, seed, type="output_divergence",
                              ref_nan_fallback=ref_nan)

    return SURVIVED
```

> **与其它维度的关键差异**: config_stress 的 ref NaN 处理**不同于** §7.4 的通用三路差分机制。当 `ref_nan=True ∧ orig 有效` 时，value_stress/training_stress 等维度会将 `compare_target` 切换为 `orig_out`，而 config_stress **始终用 ref_out 与 mut_out 做 allclose**。由于 `torch.allclose(NaN, x)` 永远返回 False，在 ref 产生 NaN 的输入配置下，config_stress 的 kill 判定可能产生 **false positive**。结果 JSON 中标记了 `ref_nan_fallback: true` 以便后续甄别。
>
> **RebatchInputs**: 通过截断 (`x[:target_bs]`) 或 repeat+截断 调整 tensor 第 0 维到目标 batch_size，非 tensor 参数和 0-D tensor 不做变换。

### 13.3 参数

- 7 batch_sizes × 3 seeds = 21 组合
- 全部在 1 个 worker 调用中完成
- 比较方式: allclose (atol=1e-2, rtol=1e-2)

---

## 14. LLM 迭代分析 (Step 3)

### 14.1 触发条件

`deterministic_killed == false` — Step 1-2 全部 5 个维度 (含 config_stress) 均未杀死该变异体。任一维度杀死即跳过 LLM 分析。

### 14.2 LLM 配置

| 参数 | 值 |
|------|------|
| 模型 | `deepseek-reasoner` (DeepSeek-R1) |
| max_tokens | 16384 |
| temperature | 不设置 (R1 推理模型不支持 temperature; 非 reasoner 模型使用 0.3) |
| API base | `https://api.deepseek.com` |
| 最大轮次 | 3 |

### 14.3 迭代流程

```
function RunLLMIterativeAnalysis(problem_file, kernel_code, mutated_code,
                                  op, site, input_spec, equiv_detail, stress_result):
    call_llm ← SetupLLMCaller()  // OpenAI-compatible client

    for round_num in 1..3:
        // Round 1: ANALYSIS_PROMPT
        if round_num == 1:
            prompt ← BuildAnalysisPrompt(
                original_code, mutated_code, operator_name, site, input_spec,
                equiv_detail,  // Phase I EMD 全部 4 层证据
                enhanced_results={  // Phase II 5 维度完整结果
                    main_track: stress_result.main_track,
                    config_track: stress_result.config_track,
                })

        // Round 2-3: REANALYSIS_PROMPT (含前轮失败反馈)
        else:
            prompt ← BuildReanalysisPrompt(
                ..., previous_rounds=rounds_history, ...)

        // 保存提示词到 prompts/{mutant_id}_r{N}.txt
        SavePrompt(prompt)

        // 调用 LLM
        llm_resp ← call_llm(prompt)
        // llm_resp = {content, reasoning_content, model, usage}

        // 保存原始响应到 llm_responses/{mutant_id}_r{N}_response.json
        SaveResponse(llm_resp)

        // 解析 LLM JSON 响应
        parsed ← ParseLLMResponse(llm_resp.content)
        // parsed = {reason_category, proof_sketch, survival_reason,
        //           killable, kill_strategy, suggested_test, recommendations}

        if parsed is NULL:
            continue  // 解析失败，尝试下一轮

        if NOT parsed.killable OR parsed.suggested_test is NULL:
            BREAK  // LLM 判定不可杀，终止迭代

        python_code ← parsed.suggested_test.python_code

        // Fixed-shape 过滤: 检查 kill_strategy 是否包含 shape 变更关键词
        if LLMSuggestionViolatesFixedShape(parsed.kill_strategy):
            continue  // 违反 fixed-shape 契约，跳过执行

        // 执行 LLM 建议的测试代码 (安全校验在 _verify_llm_suggestion 内部)
        exec_result ← VerifyLLMSuggestion(problem_file, kernel_code, mutated_code,
                                           python_code)
        // _verify_llm_suggestion 内部流程:
        //   1. ValidateSuggestedCode(python_code) → 若含 FORBIDDEN_PATTERNS → 返回 {killed:false, error:"safety_rejected:..."}
        //   2. 若通过安全校验 → Worker.llm_verify(..., atol=1e-2, rtol=1e-2)

        round_record.execution_result ← exec_result

        if exec_result.killed:
            return KILLED(round_num,
                          test_construction_rule={kill_strategy, suggested_code})

        // 记录失败信息供下一轮参考
        rounds_history.append(round_record)

    // 迭代结束: 未杀死
    result ← {killed: false, rounds: rounds_history}
    if rounds_history is NOT EMPTY:
        // 取最后一轮的 survival_reason 作为鲁棒性建议
        result.robustness_suggestion ← rounds_history[-1].survival_reason
    return result
```

### 14.4 ANALYSIS_PROMPT 输入内容 (Round 1)

| 输入块 | 内容 |
|--------|------|
| 完整源码 | 原始 + 变异的完整 Python+CUDA 源代码 |
| 变异信息 | operator_name, line_start, original_fragment, node_type |
| 输入规格 | input_spec (tensor shapes, dtypes) |
| EMD 证据 (L0-L3) | Layer 0 文本差异, Layer 1 规则, Layer 2 已测策略/seeds/divergence, Layer 3 verdict/reasoning/kill_strategy |
| 增强测试结果 | value_stress 详细策略结果, dtype_stress 结果, repeated_run 结果, training_stress 结果, config_stress 结果 |
| 策略语义参考 | 已测策略的数值含义 |

### 14.5 Mandatory Reasoning Steps (强制推理步骤)

LLM 被要求按以下顺序推理:

1. **Reachability analysis**: 推导涉及的循环变量、线程索引、维度变量的具体取值范围
2. **Semantic distinguishability**: 在推导的范围内，变异是否会产生不同结果
3. **Coverage gap identification**: 若差异可达，解释哪个值模式被遗漏
4. **Conclusion**: 仅在 Step 1-3 找到具体可达场景时输出 `killable: true`

### 14.6 REANALYSIS_PROMPT 输入内容 (Round 2-3)

在 Round 1 基础上额外提供:
- 前轮 suggested_test 的执行结果 (ref_ok, original_ok, mutant_ok, error, diff_summary)
- 前轮的 kill_strategy 和 suggested_code

### 14.7 Fixed-shape 过滤

LLM 的 `kill_strategy` 经过 14 个 shape 变更关键词检查:

```python
SHAPE_CHANGE_KEYWORDS = [
    "change shape", "different shape", "different dimension",
    "vary m/n/k", "vary m,n,k", "non-divisible size", "non-divisible",
    "change the input size", "change input dimensions",
    "different batch", "vary batch", "change batch_size",
    "modify the shape", "alter the dimensions",
]
```

包含任一关键词则拒绝执行该轮代码。

### 14.8 LLM Verify Worker

复用 `_stress_worker.py` 的 `llm_verify` 模式:

```
function WorkerLLMVerify(cfg):
    // 1. exec() 执行 LLM 生成的 python_code, 提取 generate_inputs(device)
    namespace ← {torch, math}
    exec(cfg.test_inputs_code, namespace)
    gen_fn ← namespace["generate_inputs"]

    // 2. 自动修正参数个数
    llm_inputs ← gen_fn(device)
    ref_inputs ← get_inputs()
    expected_n ← len(ref_inputs)
    // 仅在 LLM 生成的参数数量 > 期望数量时截断
    // 若 < expected_n 则不修正 (会在 ref_model(*llm_inputs) 处抛 TypeError)
    if len(llm_inputs) > expected_n:
        llm_inputs ← llm_inputs[:expected_n]

    // 3. 三路 allclose 比较 (同 §7, 包含 ref NaN fallback)
    // 4. 如果杀死, 额外记录 diff_summary (max_diff, mean_diff, range)
```

### 14.9 LLM 响应记录结构

```json
{
    "content": "LLM 最终回答 (JSON)",
    "reasoning_content": "R1 推理链 (仅 DeepSeek-R1)",
    "model": "实际使用的模型",
    "usage": {
        "prompt_tokens": 5000,
        "completion_tokens": 2000,
        "total_tokens": 7000,
        "reasoning_tokens": 1350
    }
}
```

### 14.10 LLM 输出字段

LLM 成功杀死变异体时，结果包含 `test_construction_rule` 字段:
```json
{
    "test_construction_rule": {
        "kill_strategy": "LLM 的杀死策略描述",
        "suggested_code": "成功杀死的 Python 测试代码"
    }
}
```

LLM 未能杀死时，结果包含 `robustness_suggestion` 字段（取自最后一轮的 `survival_reason`）。

### 14.11 Phase I Layer 3 suggested_test 的处理

- 不作为独立维度执行
- 其内容通过 equiv_detail 的 Layer 3 部分传入 LLM (作为"上次判断"的上下文)
- LLM 在 Step 3 中为每个存活变异体重新生成建议

### 14.12 策略语义参考的范围

`_build_policy_semantics(enhanced_results)` 只为该变异体的 `value_stress.policy_results` 中**实际出现过的策略名**生成语义说明，而非列出全部 21 个策略。这避免了 prompt 中包含无关策略信息，同时让 LLM 理解已测策略的数值含义。

---

## 15. Tier 1 特有步骤: Layer 2 Divergence Replay

### 15.1 目的

Tier 1 变异体在 Phase I Layer 2 中发现了 bitwise 差异（记录在 `equiv_detail.layer2.divergence` 中），但 Phase I 的 allclose 容差下未判定为 killed。Replay 尝试用同样的 (seed, policy) 在 allclose 容差下重新测试。

### 15.2 执行流程

```
function RunTier1Replay(problem_file, kernel_code, mutated_code, equiv_detail):
    divergence ← equiv_detail.layer2.divergence
    if divergence is EMPTY:
        return NOT_EXECUTED("no_divergence_data")

    replay_seed ← divergence.seed
    replay_policy ← divergence.policy

    // 复用 value_stress Worker
    result ← Worker.value_stress(replay_policy, replay_seed, atol=1e-2, rtol=1e-2)

    // 判杀条件 A: allclose 失败
    if result.original_ok AND NOT result.mutant_ok:
        return KILLED(type="allclose")

    // 判杀条件 B: allclose 通过但 bitwise 不一致
    if result.original_ok AND result.mutant_ok AND NOT BitwiseEqual(orig_out, mut_out):
        return KILLED(type="bitwise")

    return SURVIVED
```

---

## 16. Tier 3 特有: 统计置信度计算

对于 Phase II 全部步骤后仍存活的 Tier 3 变异体，计算等价置信度:

```
function ComputeTier3Confidence(stress_result):
    n_passed ← 0
    for each dimension_result in (main_track ∪ config_track):
        if NOT dimension_result.executed: continue
        // 统计 policy_results (value_stress, training_stress)
        for each pr in dimension_result.policy_results:
            if pr.ref_ok AND pr.original_ok AND pr.mutant_ok:
                n_passed += 1
        // 统计 results (dtype_stress, repeated_run, config_stress)
        for each rr in dimension_result.results:
            if NOT rr.error AND NOT rr.killed AND NOT rr.ref_fail:
                n_passed += 1
    n_passed ← max(n_passed, 1)

    // Bayesian-style 下界
    confidence_lower_bound ← 1.0 - 1.0 / (n_passed + 1)

    return {
        total_passed_rounds: n_passed,
        confidence_equivalent_lower_bound: confidence_lower_bound,
    }
```

**统计解释**:
- `confidence_lower_bound = 1 - 1/(N+1)`: 在均匀先验假设下，N 轮独立测试均未杀死时，等价概率的 Bayesian 后验下界。例如 N=132 时，下界为 0.9925。
- 隐含的 kill 概率上界: 若真实 kill 概率为 p，则 N 轮全部通过的概率为 `(1-p)^N`。在 95% 置信水平下，`p < 1 - 0.05^(1/N)`。N=132 时，p < 0.0226。

---

## 17. 安全机制

### 17.1 GPU 健康检查

每个变异体测试完成后执行:

```
function GPUHealthCheck():
    try:
        t ← torch.zeros(1, device="cuda")
        _ ← t + 1
        torch.cuda.synchronize()
        return True
    except:
        return False
```

失败则等待 10s 重试，仍失败则 abort 整个实验。

### 17.2 GPU 清理

每个变异体测试完成后:
- 清除 stale CUDA 模块 (`sys.modules` 中 `mutant_*`, `ref_*`, `stress_*` 前缀)
- `gc.collect()` + `torch.cuda.empty_cache()` + `torch.cuda.synchronize()`

### 17.3 连续超时中断

`value_stress` 和 `training_stress` 维度: 连续 5 次子进程超时 → 跳过该维度剩余轮次 (不跳过整个变异体或其它维度)。

### 17.4 GPU 内存限制

Worker 子进程启动时:
- 检查 GPU 空闲内存 ≥ 512MB
- 设置 `torch.cuda.set_per_process_memory_fraction(0.90)` — 超出时抛 CUDA OOM 而非冻结 GPU driver

### 17.5 LLM 代码安全校验

```python
FORBIDDEN_PATTERNS = [
    "import os", "import subprocess", "import sys", "import shutil",
    "open(", "__import__", "exec(", "eval(", "compile(",
    "globals(", "locals(", "getattr(", "setattr(",
    "os.system", "os.popen", "subprocess.",
]
```

LLM 生成代码包含任一 pattern → 拒绝执行。同时必须包含 `generate_inputs` 函数定义。

---

## 18. 主编排伪代码

```
function Main():
    // ═══ 数据加载 ═══
    enhanceable_list ← LoadAllEnhanceable(block12_results)
    // 筛选 status ∈ {survived, candidate_equivalent}
    all_policies ← GetAllPolicyNames()  // 21 个

    // ═══ Tier 分类 ═══
    tier_groups ← {1: [], 2: [], 3: []}
    for each (kernel_name, kernel_meta, mutant_meta) in enhanceable_list:
        tier ← ClassifyTier(mutant_meta)
        tier_groups[tier].append(item)

    tier3_filtered ← [item for item in tier_groups[3]
                       if ShouldChallengeTier3(item.mutant_meta)]

    // ═══ 执行顺序 ═══
    execution_order ← Tier1_all + Tier2_all + Tier3_filtered

    // ═══ 断点续跑 ═══
    completed ← LoadCompletedSet()

    for each (tier, kernel_name, kernel_meta, mutant_meta) in execution_order:
        if mutant_meta.id ∈ completed: SKIP

        // ═══ 准备代码 ═══
        kernel_code ← ReadKernelFile(kernel_meta)
        mutated_code ← mutant_meta.mutated_code  // 或重建

        // ═══ 策略去重 ═══
        new_policies ← GetNewPolicies(op_name, all_policies, mutant_meta)
        seeds_per_policy ← 5 if tier==3 else 3

        // ═══ 初始化结果对象 ═══
        stress_result ← StressTestResult(mutant_id, op_name, ...)

        // ═══ Step 1-2: 确定性测试 (按 Tier 排列维度顺序) ═══
        if tier == 1:
            // replay → value → dtype → training → repeated → config
            stress_result.record("main", "tier1_replay",
                RunTier1Replay(...))
            stress_result.record("main", "value_stress",
                RunValueStress(new_policies, seeds_per_policy))
            stress_result.record("main", "dtype_stress",
                RunDtypeStress(...))
            stress_result.record("main", "training_stress",
                RunTrainingStress(op_name, all_policies))
            stress_result.record("main", "repeated_run",
                RunRepeatedRun(...))
            stress_result.record("config", "config_stress",
                RunConfigStress(...))

        elif tier == 2:
            // dtype → training → value → repeated → config
            ...

        elif tier == 3:
            // dtype → value(5seeds) → repeated → training → config
            ...

        // ═══ Step 3: LLM 迭代分析 ═══
        if NOT stress_result.deterministic_killed:
            llm_result ← RunLLMIterativeAnalysis(
                ..., equiv_detail, stress_result)
            stress_result.record_llm(llm_result)
        else:
            stress_result.record_llm({executed: false, trigger: "already_killed"})

        // ═══ 输出详情 JSON ═══
        detail ← BuildDetailJSON(stress_result, tier, mutant_meta)
        if tier == 3 AND NOT stress_result.any_killed:
            detail.tier3_confidence ← ComputeTier3Confidence(stress_result)
        SaveJSON(detail, "details/{mutant_id}.json")

        // ═══ 断点保存 ═══
        completed.add(mutant_id)
        SaveCompleted()

        // ═══ GPU 维护 ═══
        GPUCleanup()
        if NOT GPUHealthCheck():
            Wait(10s)
            if NOT GPUHealthCheck(): ABORT

    // ═══ Step 4: 汇总 ═══
    SaveSummary(summary)
```

---

## 19. 结果存储

### 19.1 每个变异体的详情 JSON

路径: `stress_enhance_results/details/{mutant_id}.json`

字段按时间顺序排列:

```json
{
  "mutant_id": "L1_P1__relop_replace__2",
  "operator_name": "relop_replace",
  "operator_category": "A",
  "kernel_name": "L1_P1",
  "tier": 1,
  "original_status": "survived",
  "site_node_type": "cuda_Lt",

  "equiv_detail": { "layer0": {}, "layer1": {}, "layer2": {}, "layer3": {} },

  "main_track": {
    "tier1_replay": {
      "executed": true | false,
      "killed": false,
      "detail": { "seed": 10042, "policy": "structured_ramp" }
    },
    "value_stress": {
      "executed": true,
      "killed": false,
      "killing_policy": null,
      "killing_seed": null,
      "kill_type": null,
      "rounds_executed": 45,
      "rounds_total": 45,
      "policy_results": [
        {"policy": "near_zero", "seed": 42, "original_ok": true,
         "mutant_ok": true, "ref_ok": true, "time_ms": 1234.5, "error": ""}
      ],
      "original_failures": [],
      "aborted_reason": null
    },
    "dtype_stress": {
      "executed": true,
      "killed": false,
      "killing_dtype": null,
      "killing_seed": null,
      "results": [
        {"seed": 142, "killed": false, "tested_dtypes": ["float16", "bfloat16"]}
      ]
    },
    "repeated_run": {
      "executed": true,
      "killed": false,
      "inconsistency_detected": false,
      "divergent_trial": null,
      "killing_seed": null,
      "results": [{"seed": 242, "killed": false}]
    },
    "training_stress": {
      "executed": true | false,
      "killed": false,
      "skipped_reason": null | "operator_not_applicable",
      "killing_policy": null,
      "killing_seed": null,
      "rounds_executed": 63,
      "rounds_total": 63,
      "results": [],
      "original_failures": [],
      "aborted_reason": null
    }
  },

  "config_track": {
    "config_stress": {
      "executed": true,
      "killed": false,
      "killing_batch_size": null,
      "kill_type": null,
      "results_per_batch": {
        "1": {"seeds_tested": [{"seed": 42, "status": "passed"}], "status": "passed"},
        "64": {"seeds_tested": [{"seed": 42, "status": "orig_diverges_from_ref"}], "status": "passed"}
      }
    }
  },

  "original_failures": [],  // 原始 kernel 在某些策略下也失败的策略名列表 (从各维度的 original_failures 聚合)

  "llm_iterative_analysis": {
    "executed": true,
    "trigger": "all_dimensions_survived",
    "rounds": [
      {
        "round": 1,
        "prompt_type": "ANALYSIS_PROMPT_V2",
        "model": "deepseek-reasoner",
        "usage": {"prompt_tokens": 5000, "completion_tokens": 2000,
                  "total_tokens": 7000, "reasoning_tokens": 1350},
        "reason_category": "predicate_unreachable",
        "proof_sketch": "...",
        "survival_reason": "...",
        "killable": false,
        "kill_strategy": "unkillable under fixed-shape contract",
        "recommendations": "...",
        "suggested_code": "",
        "execution_result": null,
        "killed": false
      }
    ],
    "killed": false,
    "killing_round": 0,
    "robustness_suggestion": "...",       // 未杀死时: 最后一轮 survival_reason
    "test_construction_rule": null        // 杀死时: {kill_strategy, suggested_code}
  },

  "kill_summary": {
    "deterministic_killed": false,
    "llm_killed": false,
    "main_track_killed_by": [],
    "config_track_killed_by": [],
    "llm_killing_round": 0,
    "total_dimensions_executed": 6,
    "total_dimensions_killed": 0,
    "final_killed": false
  },
  "any_killed": false,
  "first_kill_mode": null,
  "total_time_ms": 0.0,  // ⚠ 当前实现中为占位字段, 始终为 0.0 (见 §20.1 说明)

  "tier3_confidence": {
    "total_passed_rounds": 132,
    "confidence_equivalent_lower_bound": 0.9925,
    "interpretation": "After 132 independent ..."
  }
}
```

### 19.2 文件结构

```
第二次实验汇总/
├── full_block12_results/           ← Phase I 输出 (已有)
│   ├── details/*.json
│   └── summary.md / summary.json
├── stress_enhance_results/         ← Phase II 输出
│   ├── details/                    ← 534 个变异体的完整 JSON
│   ├── prompts/                    ← LLM 调用的完整提示词
│   ├── llm_responses/              ← LLM 原始响应 + 推理链
│   ├── completed.json              ← 断点续跑 (534 个 id)
│   └── stress_summary.json         ← 汇总统计
└── docs/
    └── Phase II-增强测试.md         ← 本文件
```

---

## 20. 数据模型 (differential_tester.py)

### 20.1 StressTestResult

每个变异体的增强测试结果:

```
@dataclass
class StressTestResult:
    mutant_id: str
    operator_name: str
    operator_category: str
    kernel_name: str
    site_node_type: str
    total_time_ms: float            // ⚠ 当前实现为占位字段, 始终 0.0; 变异体耗时需从各维度 worker 返回的 time_ms 聚合
    original_failures: List[str]    // 原始 kernel 在某策略下也失败的策略名 (从各维度的 original_failures 汇总)

    main_track: Dict[str, Dict]     // 主轨道各维度结果
    config_track: Dict[str, Dict]   // 附加轨道各维度结果
    llm_analysis: Dict              // LLM 迭代分析结果
    _kill_order: List[str]          // 杀死顺序记录
```

关键属性:
- `deterministic_killed`: Step 1-2 任一维度 (main_track 或 config_track) killed
- `llm_killed`: Step 3 killed
- `any_killed`: deterministic_killed OR llm_killed
- `first_kill_mode`: `_kill_order[0]` — 第一个杀死该变异体的维度名

### 20.2 StressSummary

汇总统计:

```
@dataclass
class StressSummary:
    total_tested: int
    killed_count: int
    survived_count: int
    deterministic_kill_count: int
    llm_kill_count: int
    per_dimension_kills: Dict[str, int]  // 各维度杀死计数
    per_policy_kills: Dict[str, int]     // 各策略杀死计数
    multi_dimension_kill_count: int      // 被 2+ 确定性维度杀死的变异体数 (不含 LLM)
    llm_rounds_distribution: Dict[int, int]  // LLM 在第 N 轮杀死的分布
```

`add_result(r)` 方法:
- 一个变异体被多个确定性维度独立杀死时，每个维度都计入 `per_dimension_kills`
- `multi_dimension_kill_count` 仅统计 **确定性维度** (main_track + config_track) 中 2 个以上杀死的情况，**不含 LLM**
- LLM 杀死单独计入 `per_dimension_kills["llm_iterative_analysis"]` 和 `llm_kill_count`

---

## 21. Step 4: 后处理

> **注意**: Step 4 后处理在主编排脚本 (`run_stress_enhance.py`) 之外离线完成。`run_stress_enhance.py` 仅产出每个变异体的 detail JSON 和 `stress_summary.json`。以下描述的是基于这些数据的离线分析产出。

### 21.1 算子 × 维度交叉分析

构建 16 算子 × 7 维度 (6 确定性 + 1 LLM) 的 kill 计数矩阵:

```
               value  dtype  repeated  training  config  tier1_replay  llm
arith_replace    12     3       0         5        1         2          2
relop_replace     3     0       0         0        7         0          1
epsilon_modify    8     2       0        11        0         0          0
...
```

### 21.2 覆盖建议生成 (规则驱动)

| 杀死维度 | 建议 |
|----------|------|
| `value_stress` 杀死 | "补充 {killing_policy} 类数值分布测试" |
| `dtype_stress` 杀死 | "增加低精度 (float16/bfloat16) 验证" |
| `config_stress` 杀死 | "测试非默认 batch_size 配置" |
| `training_stress` 杀死 | ".train() 模式下的 BN/LN 行为需要验证" |
| `repeated_run` 杀死 | "非确定性检测: 重复执行" |
| LLM 杀死 | 引用 LLM 的 kill_strategy |
| 全部未杀死 | 引用 LLM 的 survival_reason |

### 21.3 存活原因聚类

全部存活变异体的 `survival_reason` 送入 LLM 做 taxonomy 聚类 (CLUSTER_PROMPT)。

### 21.4 最终统计报告

- Mutation score (保守/乐观口径)
- 等价变异体最终确认 (Tier 3 全部步骤后仍未杀死 → confirmed_equivalent)
- Kill rate by tier / by operator / by dimension
- LLM 迭代 kill rate (Round 1/2/3)
- 多维度敏感性分布

---

## 22. 有效性威胁

### 22.1 参考实现的边界行为 (Internal Validity)

差分测试依赖参考实现 (PyTorch reference) 的正确性。当参考实现在极端输入下产生 NaN/Inf 时，`ref_nan_fallback` 机制用原始 kernel 输出替代。这引入假设"原始 kernel 在该输入下的输出是正确的"。实际数据: 139/534 个变异体 (26.0%) 至少一次触发回退，累计 1303 次。

### 22.2 Fixed-shape 契约的局限性 (External Validity)

所有主轨道测试在 KernelBench 固定的 input shape 下进行。某些变异可能仅在特定 shape 下可杀 (如 grid size 恰好整除)。config_stress 的 batch size 变化仅是有限探索。

### 22.3 Stress Policy 的覆盖完整性 (Construct Validity)

21 种 stress policy 覆盖了常见数值边界模式，但可能遗漏 GPU 特有的 adversarial pattern (如 warp shuffle 语义、shared memory bank conflict)。

### 22.4 非确定性测试的统计置信度 (Construct Validity)

`repeated_run` 使用 3 seeds × 10 trials = 30 次执行。GPU 调度的非确定性可能需更多次数才能暴露低概率 race condition。

### 22.5 Worker 超时导致的信息缺失 (Internal Validity)

CUDA JIT 编译对复杂 kernel 可能需要数分钟。超时时该维度测试结果为空 (rounds_executed=0)。

### 22.6 LLM 迭代分析的效果 (Construct Validity)

LLM 生成的 kill strategy 在执行时可能因框架限制 (ref NaN/Inf、worker 超时、shape 违规过滤) 而未能体现真实效果。

---

## 23. 参考文献

| # | 引用 | 出处 |
|---|------|------|
| 1 | Du, H. et al. "To Kill a Mutant: An Empirical Study of Mutation Testing Kills" | *ACM ISSTA*, 2023 |
| 2 | Petrovic, G., Ivankovic, M. & Just, R. "Does Mutation Testing Improve Testing Practices?" | *IEEE ICSE*, 2021 |
| 3 | Offutt, A.J. & Lee, S.D. "An Empirical Evaluation of Weak Mutation" | *IEEE TSE*, 22(5), 1996 |
| 4 | Laguna, I. et al. "Testing GPU Numerics: Finding Numerical Differences Between NVIDIA and AMD GPUs" | *SC Workshop*, 2024 |
| 5 | Ouyang, S. et al. "KernelBench: Can LLMs Write GPU Kernels?" | *NeurIPS*, 2024 |
| 6 | Meta. "Mutation-Guided LLM-based Test Generation at Meta" (ACH) | *ACM FSE Industry*, 2025 |
| 7 | PRIMG. "Efficient LLM-driven Test Generation Using Mutant Prioritization" | *EASE*, 2025 |
| 8 | STING. "Are Benchmark Tests Strong Enough? Mutation-Guided Test Strengthening" | *arXiv:2604.01518*, 2025 |
| 9 | D³. "Differential Testing of Distributed Deep Learning with Model Equivalence Rules" | *IEEE TSE / ICSE*, 2025 |
| 10 | CuFuzz. "CuFuzz: An API-Knowledge-Graph Coverage-Driven Fuzzing Framework for CUDA Libraries" | *ACM FSE*, 2026 |
| 11 | Jia, Y. & Harman, M. "An Analysis and Survey of the Development of Mutation Testing" | *IEEE TSE*, 37(5), 2011 |
| 12 | Gutiérrez-Madroñal, L. et al. "Tempus: An Evolutionary Mutation Testing System" | *STVR*, 2025 |
| 13 | Chatzikonstantinou, G. et al. "MutateNN: Mutation Testing of Image Recognition Models" | *ACM FSE*, 2024 |
