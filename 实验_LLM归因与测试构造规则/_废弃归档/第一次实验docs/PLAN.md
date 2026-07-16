# Mutation-Guided Diagnosis and Augmentation of GPU Kernel Test Suites

> **目标会议**: ICSE 2027
>
> **项目代号**: MutaKernel
>
> **最后更新**: 2026-04-16 — LLM 迭代分析 + 事后聚类 + Block 4/5 合并

---

## 一、问题定义

### 1.1 背景

LLM 生成 GPU kernel 是 2025-2026 年软件工程与 ML Systems 交叉领域的活跃方向。KernelBench (ICML 2025) 等公开 benchmark 普遍采用 `torch.allclose` 配合少量随机输入作为核心正确性判据：

```python
torch.allclose(output, output_new, atol=1e-02, rtol=1e-02)
```

使用 3-5 次随机输入, 全部通过则判定正确。

### 1.2 核心问题

GPU kernel benchmark 使用的标准测试输入存在**系统性盲区**。存活变异体 (survived mutants) 不是随机分布的, 而是因为特定的测试不充分模式才存活。

核心问题链:
1. 存活变异体为什么没被杀掉? (诊断)
2. 根据诊断结果, 能否设计针对性增强策略? (增强)
3. 诊断驱动的增强是否优于盲目增加测试? (评估)

### 1.3 论文标题

**Mutation-Guided Diagnosis and Augmentation of GPU Kernel Test Suites**

### 1.4 核心论点

通过分析存活变异体的失效模式, 将测试盲区分为 6 个明确类型, 并为每类设计针对性增强策略。这种诊断驱动的增强显著优于盲目增加测试次数或随机增强。

---

## 二、已有基础

### 2.1 实验数据

- 90 个 GPU kernel (KernelBench L1+L2, DeepSeek-chat 生成)
- 1663 个变异体 (15 个算子, 4 类: A/B/C/D; 其中 12 个算子有存活变异体)
- killed=944, survived=322, stillborn=163, equivalent=234
- Baseline Mutation Score = 944/1266 = **0.7457** (排除 stillborn 和 equivalent: 1663-163-234=1266)
- 322 个存活变异体已逐一归因, 产出 12 份 .md 分析文档

### 2.2 变异算子

#### 类别 A: 通用算术变异 (Baseline 对照, 共 150 存活)

| 算子 | 定义 | 变异体总数 | 存活数 |
|------|------|----------|--------|
| A1 `arith_replace` | 替换二元运算符 (`+`→`-`) | 268 | 36 |
| A2 `relop_replace` | 替换关系运算符 (`<`→`<=`) | 267 | 53 |
| A3 `const_perturb` | 微扰常量值 (`1024`→`1023`) | 222 | 61 |

#### 类别 B: GPU 并行语义变异 (共 125 存活)

| 算子 | 定义 | 变异体总数 | 存活数 |
|------|------|----------|--------|
| B1 `index_replace` | 替换线程/块索引 | 267 | 40 |
| B2 `sync_remove` | 删除同步屏障 | 104 | 25 |
| B3 `mask_boundary` | 修改 mask 条件 (`<N`→`<N-1`) | 219 | 51 |
| B4 `launch_config_mutate` | 修改 grid/block 配置 | 112 | 9 |

#### 类别 C: ML 数值语义变异 (核心创新, 共 47 存活)

| 算子 | 对应属性 | 变异体总数 | 存活数 |
|------|---------|----------|--------|
| C1 `stab_remove` | 数值稳定性 | 7 | 1 |
| C2 `acc_downgrade` | 累积精度 | 46 | 0 |
| C3 `epsilon_modify` | 除零安全 | 32 | 13 |
| C4 `scale_modify` | 缩放正确性 | 33 | 7 |
| C5 `cast_remove` | 精度安全性 | 55 | 22 |
| C6 `reduction_reorder` | 归约确定性 | 0 | 0 |
| C7 `init_modify` | 边界初始化正确性 | 22 | 4 |

#### 类别 D: LLM 错误模式变异 (共 0 存活)

| 算子 | 来源 | 变异体总数 | 存活数 |
|------|------|----------|--------|
| D1 `broadcast_unsafe` | TritonBench 错误日志 | 4 | 0 |
| D2 `layout_assume` | LLM 常见 contiguous 假设 | 5 | 0 |

### 2.3 存活变异体归因方式（已重构）

**重要变更**: 不再使用预设 Type 1-6 分类体系。改为三阶段处理：

| 阶段 | 处理方式 | 归因输出 |
|------|---------|---------|
| Phase 1: 4 层压力测试 | 杀死变异体 | 仅记录 `killed_by_layer` + `killing_policy` + `killing_seed` |
| Phase 2: LLM 迭代分析 | 对存活变异体自由分析 | `survival_reason` (自然语言) + `killable` + 建议输入/鲁棒性修复 |
| Phase 3: 事后聚类 | 将所有 survival_reason 聚类 | 数据驱动的分类体系 (taxonomy) |

> 手工分析文档 (full_block12_results/*_survived_analysis/*.md) 提供的
> 初步存活原因认识仍有参考价值, 但最终分类体系由 Phase 3 LLM 聚类自动生成。
> 旧 Type 1-6 定义已移至 `docs/attribution_criteria.md` 作为历史参考。

### 2.4 普适性论证

存活原因的分类学本身是通用的 (适用于所有并行数值程序的变异测试),
但每类盲区在 GPU kernel 中有特有表现。最终分类体系由 Phase 3 事后聚类产出,
预期涵盖以下维度:

- **算法吸收/等价**: 通用 (equivalent mutant problem, 见 Papadakis et al. 2019 综述)
- **配置盲区**: 半通用 (配置覆盖不足, 类比"只测 x86 不测 ARM")
- **输入分布盲区**: 通用 (边界值测试不充分, 见 FPGen, ICSE 2020)
- **代码缺陷**: 通用 (变异测试的核心价值)
- **非确定性**: 半通用 (并发程序特有; warp 隐式同步是 GPU 特有表现)
- **测试基础设施**: 通用 (test oracle 不够强是普遍问题)

---

## 三、Research Questions

- **RQ1 (Taxonomy)**: 标准 benchmark 输入下, GPU kernel 变异体为何存活? 存活模式有哪些类型?
  - 输出: LLM 事后聚类产出的数据驱动分类学 + 322 个变异体的分布数据 + 普适性论证

- **RQ2 (Effectiveness)**: 诊断驱动的测试增强 + LLM 迭代分析能多大程度提升变异体杀灭率?
  - 方法: 4 层压力测试 + LLM 建议输入验证 vs 两个 baseline
  - 输出: mutation score lift (overall / per-category / per-operator)

- **RQ3 (Ablation)**: 各增强子策略的贡献如何? 诊断信息是否必要?
  - 方法: leave-one-out 消融 + 策略数量曲线
  - 输出: 策略贡献排名, 诊断价值量化

- **RQ4 (Robustness)**: 不可杀变异体揭示了哪些 kernel 鲁棒性问题? LLM 能否给出修复?
  - 方法: LLM 鲁棒性建议 + 修复代码 + 测试用例构造规则
  - 输出: 案例分析 + 修复成功率

---

## 四、方法设计 — 三阶段架构

### 4.1 总体架构

```
322 个存活变异体
    │
    ▼
┌─────────────────────────────────────┐
│  Phase 1: 4 层压力测试 (自动杀死)     │
│  Layer 1: Value-Distribution Stress  │
│  Layer 2: Configuration Augment.     │
│  Layer 3: Execution Augmentation     │
│  Layer 4: Training-Mode Augment.     │
│  → 记录 killed_by_layer, 不做分类     │
└────────────┬────────────────────────┘
             │ (未被杀死的变异体)
             ▼
┌─────────────────────────────────────┐
│  Phase 2: LLM 迭代分析 + 验证       │
│  → 自由分析 survival_reason          │
│  → 建议输入 → GPU 验证 (≤3 轮)      │
│  → 鲁棒性建议 / 修复代码             │
│  → 测试用例构造规则                   │
└────────────┬────────────────────────┘
             │ (所有 survival_reason)
             ▼
┌─────────────────────────────────────┐
│  Phase 3: 事后聚类                   │
│  → LLM 归纳分类体系 (3-8 类)         │
│  → 为每个变异体分配 cluster_label     │
│  → 输出 taxonomy.json                │
└─────────────────────────────────────┘
```

### 4.2 Phase 1: 4 层压力测试

Layer 1-4 负责杀死变异体，**不分配 Type 标签**，只记录被哪一层杀死。
详见 `Block5_StressEnhance.md` Section 四。

| 层 | 增强类型 | 输入预算 | 检测目标 |
|----|---------|---------|---------|
| Layer 1 | Value-Distribution | 14 policies × 3 seeds | 值分布盲区 |
| Layer 2 | Configuration (dtype) | 3 seeds × 2 dtypes | dtype 盲区 |
| Layer 3 | Execution (repeated) | 3 seeds × 10 trials | 非确定性 |
| Layer 4 | Training-Mode | 14 policies × 3 seeds (仅目标算子) | eval-mode masking |

**STRATEGY_MAP** (算子→策略优先级) 详见 `Block5_StressEnhance.md` Section 4.1。

### 4.3 Phase 2: LLM 迭代分析 + 验证

对 Layer 1-4 均未杀死的变异体, 由 DeepSeek-R1 进行自由分析:

```
对每个存活变异体 (最多 3 轮迭代):
  Round 1:
    ├── LLM 分析: survival_reason + killable + suggested_test + recommendations
    ├── 若 killable=true 且有 suggested_test:
    │   → GPU 三方对比验证
    │   ├── 杀死 → 提取测试规则 (test_construction_rule)
    │   └── 未杀死 → Round 2 (带失败信息 + diff_summary)
    └── 若 killable=false → 生成鲁棒性建议 + 修复代码
  Round 2-3: 结合前轮失败信息重新分析
  3 轮均未杀死 → 生成鲁棒性建议 + 修复代码
```

### 4.4 Phase 3: 事后聚类

所有变异体 Phase 2 完成后, 将全部 `survival_reason` 一次性交给 LLM:
- 归纳 3-8 个存活原因类别 (cluster)
- 为每类别生成 `category_name` + `description`
- 将每个变异体分配到类别 (cluster_label)
- 输出 `taxonomy.json`

### 4.5 Baselines

| 方法 | 配置 | 预期额外杀死 |
|------|------|-----------|
| Baseline 1: 20x Random | seed 42-61, 标准 randn | 0-5 |
| Baseline 2: Random Stress | 14 中随机选 3 × 3 seeds | 40-60 |
| **Our method** | Phase 1 + Phase 2 | **80-134** |

### 4.6 报告指标

| 指标 | 定义 | 意义 |
|------|------|------|
| Baseline MS | 944/1266 = 0.746 | 起点 |
| Augmented MS | (944 + new_kills) / 1266 | Phase 1 + Phase 2 增强后 |
| Adjusted MS | (944 + new_kills) / (1266 - unkillable) | 排除不可杀 (由 LLM 聚类判定) |
| Visibility Lift | (Augmented - Baseline) / Baseline | 相对改善 |
| Per-category Lift | 各类别 MS 变化 | 哪类受益最大 |
| Kill Attribution | 每个 layer/policy/seed 的 kill 数 | 策略有效性排名 |
| LLM Kill Rate | Phase 2 杀死数 / Phase 2 分析数 | LLM 建议有效性 |

---

## 五、代码实现状态

### 5.1 代码修改总览

#### Phase 1 相关（4 层压力测试）

| 文件 | 改动 | 状态 |
|------|------|------|
| `src/stress/differential_tester.py` | 简化为 killed/survived 记录; 删除 `AttributionType` 枚举和 `attribute()` 函数 | ✅ |
| `src/stress/policy_bank.py` | 14 policies (含 `boundary_last_element`, `head_heavy`, `tail_heavy`) | ✅ |
| `scripts/_stress_worker.py` | 4 modes: `value_stress` / `dtype_stress` / `repeated_run` / `training_stress` + `llm_verify` | ✅ |
| `scripts/run_stress_enhance.py` | 4 层增强逻辑; 优先策略排序; Layer 4 选择性触发; 不做 Type 归因 | ✅ |
| `src/mutrepair/enhanced_inputs.py` | `STRATEGY_MAP` 16 条, 覆盖 A/B/C 全部算子 | ✅ |

#### Phase 2+3 相关（LLM 迭代分析 + 事后聚类）

| 文件 | 改动 | 状态 |
|------|------|------|
| `src/stress/llm_analyzer.py` | Prompt 模板 (分析/再分析/鲁棒性/测试规则/聚类) + LLM 调用 + 响应解析 | ✅ |
| `scripts/_pilot_llm20.py` | Phase 2+3 编排: 迭代分析 + GPU 验证 + 聚类 + 结果汇总 | ✅ |
| `scripts/_verify_llm_suggestions.py` | GPU 验证 LLM 建议输入 (可复用模块) | ✅ |
| `config.py` | 新增 `LLMAttributionConfig` (model/rounds/tokens/timeout 等) | ✅ |

#### 流水线与文档

| 文件 | 改动 | 状态 |
|------|------|------|
| `scripts/_run_full_pipeline.sh` | Phase 1 + Phase 2+3 两步流水线 | ✅ |
| `docs/Block5_StressEnhance.md` | 合并 Block4 内容; 清理 Type 引用; 新增 Section 十一 | ✅ |
| `docs/Block4_MutRepair.md` | 标记为已归档, 核心功能合并进 Block5 Phase 2 | ✅ |
| `docs/attribution_criteria.md` | 重定位为参考文档; Type 1-6 保留为历史参考 | ✅ |
| `scripts/run_baselines.py` | Baseline 1 (20x random) + Baseline 2 (random stress) | ✅ |
| `scripts/run_ablation.py` | Leave-one-out 消融 + 策略数量曲线 | ✅ |

### 5.2 项目结构

```
MutaKernel/
├── PLAN.md                              # 本文件
├── config.py                            # 项目配置 (含 LLMAttributionConfig)
├── requirements.txt
├── best_kernels.json                    # 90 个 kernel 的路径索引
│
├── docs/
│   ├── attribution_criteria.md          # 参考文档: 存活原因分类 + MS 公式
│   ├── Block1_MutOperators.md
│   ├── Block2_MutEngine.md
│   ├── Block3_RealismGuard.md
│   └── Block5_StressEnhance.md          # ★ 核心文档 (含原 Block4 内容)
│
├── src/
│   ├── __init__.py
│   ├── models.py                        # MutantStatus, Mutant, KernelInfo 等
│   │
│   ├── mutengine/                       # 变异引擎 (已完成)
│   │   ├── operators/
│   │   │   ├── base.py                  # MutationOperator 基类
│   │   │   ├── arithmetic.py            # A 类 (3 算子)
│   │   │   ├── gpu_parallel.py          # B 类 (4 算子)
│   │   │   ├── ml_semantic.py           # C 类 (7 算子)
│   │   │   └── llm_pattern.py           # D 类 (2 算子)
│   │   ├── parser/
│   │   │   ├── triton_parser.py
│   │   │   └── cuda_parser.py
│   │   ├── mutant_runner.py
│   │   ├── equivalent_detector.py
│   │   ├── realism_validator.py         # RealismGuard
│   │   └── report.py
│   │
│   ├── stress/                          # ★ 核心增强 + LLM 分析模块
│   │   ├── __init__.py
│   │   ├── policy_bank.py              # 14 个 stress 策略
│   │   ├── differential_tester.py      # killed/survived 记录 (无 Type 归因)
│   │   └── llm_analyzer.py             # ★ LLM Prompt + 调用 + 解析
│   │
│   ├── mutrepair/                       # 修复模块 (RQ4 可选, 因果隔离实验)
│   │   ├── __init__.py
│   │   ├── enhanced_inputs.py          # STRATEGY_MAP (Layer 1 优先排序)
│   │   ├── feedback_builder.py         # 📦 因果隔离 Prompt (B0-Ours)
│   │   ├── repair_loop.py             # 📦 修复循环 + 双重验证
│   │   └── experience_store.py
│   │
│   ├── mutevolve/                       # 规则演化模块 (实验性)
│   │   ├── __init__.py
│   │   ├── pattern_miner.py
│   │   └── rule_generator.py
│   │
│   └── bridge/
│       ├── __init__.py
│       └── eval_bridge.py              # KernelBench 集成
│
├── scripts/
│   ├── full_block12.py                  # Block 1+2 主实验 (已完成)
│   ├── _mutant_worker.py                # Block 1+2 子进程 worker
│   ├── run_stress_enhance.py            # ★ Phase 1: 4 层增强
│   ├── _stress_worker.py                # ★ 子进程 worker (5 modes 含 llm_verify)
│   ├── _pilot_llm20.py                  # ★ Phase 2+3: LLM 迭代分析 + 聚类
│   ├── _verify_llm_suggestions.py       # ★ GPU 验证 LLM 建议输入
│   ├── _run_full_pipeline.sh            # ★ 完整流水线 (Phase 1 → Phase 2+3)
│   ├── run_baselines.py                 # Baseline 1 + 2
│   ├── run_ablation.py                  # Leave-one-out + 策略曲线
│   ├── run_repair.py                    # RQ4 修复实验 (可选)
│   ├── validate_realism.py              # RealismGuard 验证
│   ├── analyze_results.py               # 结果分析
│   ├── scan_best_kernels.py             # 扫描最佳 kernel
│   └── ...
│
├── tests/
│   ├── test_models.py
│   ├── test_operators.py
│   └── test_parsers.py
│
├── test_data/
│   ├── cuda_kernels/
│   └── l1_smoke20/
│
├── figures/
│   └── method_overview.pdf/png
│
├── full_block12_results/                # Block 1+2 实验结果 (已完成)
│   ├── summary.json
│   ├── details/*.json                   # 90 个 kernel 的详细结果
│   ├── A_survived_analysis/*.md
│   ├── B_survived_analysis/*.md
│   └── C_survived_analysis/*.md
│
├── stress_enhance_results/              # ⏳ Phase 1 输出
│   ├── details/{mutant_id}.json
│   ├── completed.json
│   └── stress_summary.json
│
├── llm_analysis_results/                # ⏳ Phase 2+3 输出
│   ├── details/{mutant_id}.json         # 每个变异体的完整分析
│   ├── summary.json                     # 汇总统计
│   ├── taxonomy.json                    # 数据驱动的分类体系
│   ├── robustness_suggestions.json      # 鲁棒性建议 + 修复代码
│   └── test_construction_rules.json     # 测试用例构造规则
│
├── baseline_results/                    # ⏳ Baseline 输出
└── ablation_results/                    # ⏳ Ablation 输出
```

---

## 六、实验执行流程

### Step 0: 代码准备 (本地) — ✅ 已完成

1. ✅ Phase 1 代码: `differential_tester.py` 简化 + `policy_bank.py` + `_stress_worker.py` (4 modes) + `run_stress_enhance.py` (4 层)
2. ✅ Phase 2+3 代码: `llm_analyzer.py` + `_pilot_llm20.py` + `_verify_llm_suggestions.py` + `config.py` (LLMAttributionConfig)
3. ✅ 流水线: `_run_full_pipeline.sh`
4. ✅ Baselines/Ablation: `run_baselines.py` + `run_ablation.py`
5. ✅ 文档同步: Block4 归档, Block5 合并, PLAN.md 更新

### Step 1: 完整流水线 (GPU 服务器, WSL) — ⏳ 待执行

```bash
# 一键运行 Phase 1 + Phase 2+3
bash /mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/scripts/_run_full_pipeline.sh
```

此命令依次执行:
1. **Phase 1**: `run_stress_enhance.py` — 322 个存活变异体 × 4 层压力测试
2. **Phase 2+3**: `_pilot_llm20.py` — 压力测试后存活的变异体 → LLM 迭代分析 + 聚类

输出:
- `stress_enhance_results/` — Phase 1 杀灭结果
- `llm_analysis_results/` — Phase 2+3 LLM 分析 + taxonomy

### Step 2: Baselines (GPU 服务器, ~1天) — ⏳ 待执行

```bash
python scripts/run_baselines.py --mode random20       # Baseline 1
python scripts/run_baselines.py --mode random_stress   # Baseline 2
```

### Step 3: Ablation (GPU 服务器, ~1天) — ⏳ 待执行

```bash
python scripts/run_ablation.py --mode leave_one_out
python scripts/run_ablation.py --mode strategy_curve
```

### Step 4: 分析与写作 (本地, ~3天) — ⏳ 待执行

1. 统计 mutation score lift (overall / per-category / per-operator)
2. 对比图: Our method vs Baseline 1 vs Baseline 2 (bar chart)
3. 策略贡献热力图: 哪个 policy/layer 杀死了哪类变异体
4. 策略数量 vs 杀灭率曲线 (ablation)
5. LLM 分析有效性: 建议输入杀灭率, 迭代轮次分布
6. RQ4 鲁棒性: LLM 修复代码案例 + 测试用例构造规则
7. 事后聚类 taxonomy 展示 + 普适性讨论

---

## 七、预期结果与风险

### 7.1 预期结果

| 方法 | 预期额外杀死 | 新 MS | 说明 |
|------|-----------|-------|------|
| Baseline 1 (20x random) | 0-5 | 0.750 | 证明"多跑无用" |
| Baseline 2 (random stress) | 40-60 | 0.78-0.79 | stress 有用但不精准 |
| **Our method (guided)** | **80-134** | **0.81-0.85** | 诊断驱动最高效 |

Adjusted MS (排除 LLM 判定的不可杀变异体后): 可能达到 **0.90+**

### 7.2 关键风险

| 风险 | 缓解 |
|------|------|
| Value stress 实际杀灭率低 | Pilot 先验证 |
| dtype 切换导致 model 不兼容 | worker 中 try/except, 记录兼容性 |
| Baseline 2 效果也很好, 削弱诊断价值 | 对比命中率: 我们应远高于随机 |
| LLM 建议输入杀灭率低 | 3 轮迭代 + 输出鲁棒性建议作为备选贡献 |
| 事后聚类结果不稳定 | 报告 taxonomy rationale, 多次运行取共识 |

### 7.3 审稿人预期质疑及回应

**Q1: "这不就是多加了几种测试?"**
A: Baseline 1 证明多跑无效; Baseline 2 证明随机 stress 不如诊断驱动。核心是从存活模式到策略的系统映射。

**Q2: "分类体系是否后验调分?"**
A: 分类体系由 Phase 3 事后聚类自动生成, 不是人工预设。报告三版 MS (baseline / augmented / adjusted)。

**Q3: "策略是人工设计的?"**
A: 每个策略有文献依据 (FPGen/Magneto/Mu2/BVA); 策略本身不新颖, 新颖在于诊断→策略的系统映射。

**Q4: "泛化性如何?"**
A: 90 个异构 kernel, 12 个算子。分类学本身通用, GPU 特有表现已论证。

**Q5: "待测代码有没有问题?"**
A: LLM 对不可杀变异体的鲁棒性分析 + 修复代码建议直接回答了这个问题。

**Q6: "LLM 归因可靠吗?"**
A: 每个 LLM 分析结论都附带 GPU 验证 (最多 3 轮); 事后聚类对比人工分析文档做一致性检查。

---

## 八、关键文献

### 变异测试

- **Mutation Survey** (Papadakis et al., Advances in Computers 2019): 变异测试综述, 基础框架
- **MUTGPU** (Zhu & Zaidman, ICST 2020): GPU 并行编程变异算子
- **Equivalent Mutants in the Wild** (Kushigian et al., ISSTA 2024): 等价变异体分类, 支撑 Type 1 处理
- **LLM for EMD** (Tian et al., ISSTA 2024): LLM 检测等价变异体

### 变异引导测试生成

- **Mu2** (Vikram et al., ISSTA 2023, Distinguished Paper): 变异引导的 greybox fuzzing
- **ACH at Meta** (Harman et al., FSE 2025): 工业级变异引导测试生成

### 数值正确性测试

- **FPGen** (Guo & Rubio-Gonzalez, ICSE 2020): 浮点错误输入生成, 支撑 near_zero/large_magnitude 策略
- **Magneto** (Jeangoudoux et al., ISSTA 2021): 数值规格变异测试, 支撑数值精度测试策略
- **Predoo** (Zhang et al., ISSTA 2021): DL 算子精度测试, 支撑 GPU kernel 测试场景
- **Audee** (Guo et al., ASE 2020): DL 框架差分测试, 支撑三方对比设计
- **FTTN** (Li et al., CCGrid 2024): GPU 矩阵加速器数值行为测试, 支撑 dtype/subnormal 策略

### GPU Kernel 生成与评测

- **KernelBench** (Ouyang et al., ICML 2025): 标准评测 benchmark
- **TritonBench** (Li et al., ACL 2025)
- **CUDA Agent** (Dai et al., 2026)
- **TritonRL** (Woo et al., 2026)

---

## 九、Threats to Validity 预案

| 威胁 | 类型 | 应对 |
|------|------|------|
| C/D 变异体可能不代表真实 bug | Construct | RealismGuard (pattern matching + injection detection) |
| 等价变异体导致可见率失真 | Internal | 统计检测 + 人工抽样 + 三版 MS |
| 仅 KernelBench/DeepSeek-chat | External | 讨论中分析, 可扩展 GPT-4o/TritonBench |
| Stress policies 可能不够全面 | Internal | 报告 residual analysis, 承认局限 |
| GPU 执行非确定性 | Internal | any-divergence 检测 (10 次) |
| LLM 归因可能不一致 | Internal | GPU 验证闭环 + 事后聚类共识 + 人工抽检 |

---

## 十、论文故事线

```
1. LLM 生成 GPU kernel 的评测普遍依赖 torch.allclose + 随机输入
   ↓
2. 变异测试揭示: 322 个变异体存活, baseline MS = 0.746
   ↓
3. Phase 1 (压力测试): 4 层增强自动杀死大部分变异体
   ↓
4. Phase 2 (LLM 分析): 对剩余变异体做自由分析 + 建议输入 + GPU 验证闭环
   ↓
5. Phase 3 (事后聚类): 从分析结果中归纳数据驱动的存活原因分类学
   ↓
6. 评估: Our method >> Random stress >> More random runs
   ↓
7. 消融: 每个策略/层的贡献 + LLM 建议的增量价值
   ↓
8. 鲁棒性: 不可杀变异体 → LLM 修复建议 + 测试用例构造规则
   ↓
9. 启示: benchmark 设计应纳入数值语义维度的 stress 测试,
   诊断驱动 + LLM 辅助的增强是提升测试质量的有效闭环方法
```
