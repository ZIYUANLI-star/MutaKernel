# MutaKernel 实验完整报告

> **实验时间**: 2026 年 5 月  
> **实验对象**: 四个 GPU 内核基准数据集中的 LLM / RL 生成内核实现以及人工编写的真实开源内核  
> **实验框架**: MutaKernel 五维度差分测试引擎  
> **数据来源**:
>
> - CUDA-L1: `外部Benchmark差分测试_RQ4/CUDA-L1/checkpoint.json`（及 `details/*.json`，231 个文件）
> - AI-CUDA-Engineer: `外部Benchmark差分测试_RQ4/AI-CUDA-Engineer/checkpoint.json`（及 `details/*.json`，222 个文件）
> - TritonBench-G: `外部Benchmark差分测试_RQ4/TritonBench-G/checkpoint.json`（及 `details/*.json`，139 个文件）
> - CUDA-Agent: `外部Benchmark差分测试_RQ4/CUDA-Agent/checkpoint.json`（及 `details/*.json`，176 个文件）
> - 实验执行脚本: `scripts/run_fullscale_diff_test.py`、`scripts/_stress_worker.py`

---

## 1. 实验概览

### 1.1 研究目标

本次实验的目标是**评估 LLM/RL 自动生成的 GPU 内核实现的功能正确性**，并在不同生成范式（RL 训练、LLM Agent、Agentic RL）和人工编写的开源工业代码上进行横向对照。具体而言：

- **测试对象**: 来自四个公开基准的待测内核（`ModelNew` 类），其中三个由大模型/强化学习方法自动生成，一个为人工编写的真实开源代码
- **参考对象**: 同一问题的 PyTorch 原生实现（`Model` 类，使用经过广泛验证的 PyTorch 标准算子如 `torch.matmul`、`torch.nn.functional.softmax` 等）
- **判定方法**: 在统一的输入下，比较 `ModelNew` 与 `Model` 的输出张量；若任一维度的测试用例下输出超出容差，则判定该 LLM 生成内核存在缺陷

> **术语澄清**: 本实验属于**差分测试 (Differential Testing)**，不同于第二次实验的变异测试 (Mutation Testing)。本报告中的"检出差异"指 LLM 生成内核的输出与 PyTorch 参考实现产生的输出不一致；"未检出差异"指两者在所有测试用例下输出均在容差范围内一致。

### 1.2 数据集与内核来源


| 数据集              | 来源                                                   | 内核类型          | 内核来源性质                        | 总数      | 难度分级                       |
| ---------------- | ---------------------------------------------------- | ------------- | ----------------------------- | ------- | -------------------------- |
| CUDA-L1          | deepreinforce-ai/CUDA-L1 (Wang et al., 2025)         | CUDA C++      | **RL 自动生成**                   | 241     | L1/L2/L3（KernelBench 三级难度） |
| AI-CUDA-Engineer | SakanaAI/AI-CUDA-Engineer (Sakana AI, 2025)          | CUDA C++      | **LLM Agent 自动生成**            | 229     | L1/L2/L3（KernelBench 三级难度） |
| TritonBench-G    | thunlp/TritonBench (Liang et al., 2025, ACL)         | Triton Python | **人工编写**（从 55 个开源项目爬取的真实生产代码） | 141     | 无难度标签                      |
| CUDA-Agent       | BytedTsinghua-SIA/CUDA-Agent (ByteDance & THU, 2026) | CUDA C++      | **大规模 Agentic RL 自动生成**       | 220     | L1/L2/L3（KernelBench 三级难度） |
| **合计**           | —                                                    | —             | —                             | **831** | —                          |


> **难度分级说明** (来自 KernelBench 论文 Ouyang et al., 2025):
>
> - **L1**: 单个原子算子（matmul、conv、softmax 等基础操作）
> - **L2**: 算子序列/融合（多个算子组成的中等复杂度子图）
> - **L3**: 完整网络架构（端到端的小型模型）

> **TritonBench-G 性质说明**：TritonBench-G 是从 GitHub 上 55 个开源项目（lightllm、triton-lang/triton、Liger-Kernel、bitsandbytes 等）爬取的**184 个真实世界 Triton 算子**，原本设计用作"LLM 评估目标"。本实验把它作为**人工内核基线**，与三个 AI 自动生成数据集（CUDA-L1、AI-CUDA-Engineer、CUDA-Agent）形成对照，用于回答"5 维度差分测试是否同样能在成熟的人工编写内核中发现鲁棒性缺陷"。

> **CUDA-Agent 性质说明**：CUDA-Agent 是 ByteDance 与清华大学软件创新研究院（SIA）于 2026 年联合发布的**大规模 Agentic 强化学习 CUDA 内核生成系统**（GitHub: BytedTsinghua-SIA/CUDA-Agent，arXiv:2602.24286），结合可扩展数据合成、技能增强 CUDA 开发环境与长期决策强化学习。原论文声称在 KernelBench 上取得 SOTA 表现：L1/L2/L3 各级 100%/100%/92% 加速率（相对 torch.compile）和 **98.8% 总体 pass rate**，超越 Claude Opus 4.5 和 Gemini 3 Pro。本实验在该数据集上的 5 维度差分测试结果可视为**对其声称正确性的独立外部验证**。

### 1.3 测试维度与超参数

> **数据来源**: `scripts/run_fullscale_diff_test.py` 第 38-47 行（常量定义）和 `run_kernel_5dim` 函数（第 197-388 行）

每个内核执行 **5 个独立测试维度**，所有维度均完整运行（不存在交叉早停）：

> **与论文 §5.2 / Table 6 标准配置的关系（重要）**：本 RQ4 外部 benchmark 实验因数据规模大（831 个内核、>120 GPU·小时）采用了**缩减预算配置**——每策略 2 seed、`config_stress` 仅 4 个 batch size `[1, 4, 16, 64]`、`repeated_run` 重跑 5 次、全部维度统一 `atol = rtol = 1e-2`。本节所有数字（及全报告 Tables）均严格基于此真实配置，并与论文 §6.3 正文 "five reruns" 的表述一致。
>
> 这区别于论文 §5.2 / Table 6 / Figure 3 描述的**标准 MutaKernel**配置（`value/training_stress` 3 seed、`config_stress` 7 个 batch size `{1, 2, 4, 8, 16, 32, 64}`、`repeated_run` 重跑 10 次且 `atol = rtol = 1e-6`），后者用于 RQ3 变异实验（脚本 `run_stress_enhance.py`）。即：RQ3 用标准配置，RQ4 用缩减预算配置，二者脚本与超参不同，互不冲突。


| 维度                  | 含义                  | 输入策略数                          | seed 数/策略 | 测试用例总数 | 工作模式               |
| ------------------- | ------------------- | ------------------------------ | --------- | ------ | ------------------ |
| **value_stress**    | 推理模式下的数值压力测试        | 21 种                           | 2         | 42     | 比较 forward 输出      |
| **dtype_stress**    | 数据类型转换压力            | 2（float16, bfloat16）           | 2         | 4      | 改变输入 dtype 后比较     |
| **training_stress** | 训练模式下的数值压力测试        | 21 种                           | 2         | 42     | 包含反向传播路径           |
| **repeated_run**    | 非确定性检测              | 1 种（默认输入重跑 5 次）                | 2         | 2      | 检测每次运行是否一致         |
| **config_stress**   | 不同 batch size 下的鲁棒性 | 4 个 batch size: [1, 4, 16, 64] | 2         | 8      | 比较各 batch size 的输出 |


**21 种 value_stress 策略名称（数据来源: `src/stress/policy_bank.py` 第 203-227 行）**:
`large_magnitude`、`near_overflow`、`near_zero`、`denormals`、`all_negative`、`all_positive`、`mixed_extremes`、`alternating_sign`、`sparse`、`uniform_constant`、`structured_ramp`、`boundary_last_element`、`head_heavy`、`tail_heavy`、`relop_boundary_hit`、`extreme_magnitude`、`near_epsilon`、`reduction_adversarial`、`init_sensitive`、`dense_nonzero`、`sparse_extreme`

**比较工具**:

- 容差: `atol = 1e-2, rtol = 1e-2`（`scripts/run_fullscale_diff_test.py` 第 43-44 行）
- 比较函数: `torch.allclose(ref_out.float().cpu(), orig_out.float().cpu(), atol, rtol)`（`scripts/_stress_worker.py` 第 50 行）
- NaN/Inf 回退: 若参考实现产生 NaN/Inf，则只检查待测内核是否同样为 NaN/Inf（避免假阳性）

### 1.4 Baseline 测试的定义、来源与执行流程

> **数据来源**: `scripts/run_fullscale_diff_test.py` 第 146-176 行（`run_quick_baseline`）；`scripts/_stress_worker.py` 第 130-176 行（baseline 执行实现）

#### 1.4.1 Baseline 测试是什么？

**Baseline 不是数据集预先存储的标签，而是 MutaKernel 框架在每次实验时现场执行的功能正确性测试**。它在每个内核进入 5 维度 stress 测试前，先用**数据集自带的默认输入函数 `get_inputs()`** 生成 3 组标准随机输入（3 个不同 seed），分别让待测内核与 PyTorch 参考实现各跑一次，比较输出是否一致。

#### 1.4.2 Baseline 测试每个要素的来源


| 测试要素            | 来源                                      | 提供方                                                                                                                                                | 是否可复现            |
| --------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| 待测内核源码          | 数据集发布的 `kernel_*.py`                    | 数据集作者（CUDA-L1: RL 生成；AI-CUDA-Engineer: LLM 生成；TritonBench-G: 人工编写；CUDA-Agent: Agentic RL 生成）                                                       | ✓ 公开             |
| PyTorch 参考实现    | 数据集发布的 `problem_*.py` 中的 `Model` 类      | 数据集作者（基于 KernelBench `Model` 范式）                                                                                                                   | ✓ 公开             |
| 输入生成函数          | 数据集发布的 `problem_*.py` 中的 `get_inputs()` | 数据集作者（CUDA-L1、AI-CUDA-Engineer、CUDA-Agent 的输入函数与 KernelBench 原版一致；TritonBench-G 的输入函数从原 `TritonBench-0/data/TritonBench_G_v1/` 仓库的 `test_*` 函数中挖掘） | ✓ 公开             |
| Seed（3 个）       | MutaKernel 框架                           | **本工作**：seed ∈ {0, 1, 2}（`run_quick_baseline` 函数 `n_seeds=3`）                                                                                      | ✓ 固定可复现          |
| 容差              | MutaKernel 框架                           | **本工作**：`atol=rtol=1e-2`（沿用 KernelBench 评估协议）                                                                                                      | ✓ 固定可复现          |
| 比较算子            | MutaKernel 框架                           | **本工作**：`torch.allclose(ref, kernel, atol, rtol)`                                                                                                  | ✓ PyTorch 标准 API |
| 不施加任何 stress 变换 | MutaKernel 框架的 `__identity__` 策略        | **本工作**：`stress_inputs = template_inputs`（原样使用 `get_inputs()` 输出，不做任何修改）                                                                           | ✓                |


> **关键澄清**：本实验的 baseline 与"数据集作者发布时所做的正确性验证"在概念上等价（同一份输入函数 + 同一份 PyTorch 参考），但本工作以更严格的多 seed + KernelBench 标准容差独立执行了一遍，因此 baseline 失败率不能直接等同于"数据集作者声明的错误率"。

#### 1.4.3 一次 baseline 测试的具体执行流程（伪代码）

以 CUDA-L1 矩阵乘法内核 `cuda_l1__L1_T1` 为例：

```python
# 步骤 1：从数据集加载 PyTorch 参考与输入生成函数
ref_module = load("external_benchmarks/cuda_l1/problems/problem_L1_T1.py")
get_inputs = ref_module.get_inputs   # 数据集自带：return [torch.rand(4096,4096), torch.rand(4096,4096)]
PyTorch_Model = ref_module.Model     # 数据集自带：forward = torch.matmul(A, B)

# 步骤 2：分别用 3 个不同 seed 跑测试
for seed in [0, 1, 2]:                 # MutaKernel 指定的 3 个种子
    torch.manual_seed(seed)
    inputs = get_inputs()              # 数据集函数生成默认形状/分布的随机张量

    ref_out  = PyTorch_Model(*inputs)        # PyTorch 标准结果
    kernel_out = LLM_Generated_Kernel(*inputs)  # 待测的 RL/LLM/人工内核

    # 步骤 3：用 KernelBench 标准容差比较
    if torch.allclose(ref_out, kernel_out, atol=1e-2, rtol=1e-2):
        passed += 1
    else:
        failed += 1
```

#### 1.4.4 PyTorch 作为参考标准的合理性

1. PyTorch 原生算子（`torch.matmul`、`torch.nn.functional.softmax` 等）是经过数百万开发者长期验证的成熟实现，被工业界和学术界广泛认定为"事实标准"
2. KernelBench (Ouyang et al., 2025)、CUDA-L1 (Wang et al., 2025)、AI-CUDA-Engineer (Sakana AI, 2025)、TritonBench (Liang et al., 2025)、CUDA-Agent (ByteDance & THU, 2026) 等基准/系统**均采用** "LLM/RL 内核输出 vs PyTorch 输出 + `allclose` 比较"作为正确性评估方法，本实验直接沿用该范式
3. PyTorch 实现位于数据集作者发布的 `problem_*.py` 中（不是临时编写），其语义已被原始基准论文认可

#### 1.4.5 Baseline 三种结果分类

- `passed`: 该 seed 下 LLM 内核输出与 PyTorch 参考在容差内一致（`torch.allclose` 返回 `True`）
- `failed`: 该 seed 下 LLM 内核运行成功但输出超出容差（`torch.allclose` 返回 `False`）
- `errors`: 该 seed 下编译/运行报错（worker 异常或 timeout）

#### 1.4.6 SKIPPED 判定

仅当 3 次 baseline **全部** errors（即全部编译失败或运行崩溃，3/3 errors）时，标记为 SKIPPED，跳过 5 维度测试。其他情况（即使 baseline 全 failed 即 3/3 failed）仍执行完整 5 维度测试。

### 1.5 检出率计算公式（双阶段联合检出）

MutaKernel 框架是天然的**两阶段架构**：每个内核先经过 baseline 检测（标准随机输入），再进入 5 维度 stress 检测（极端输入）。两个阶段产生**独立的阳性证据**，本报告采用以下统一判定：

> **缺陷判定准则（联合口径）**: 一个内核被判定为存在缺陷当且仅当——
> $$D(k) = \mathbb{1}[\text{baseline.failed}(k) > 0] \lor \mathbb{1}[\text{totaldiscrepancies}(k) > 0]$$
> 即：3 次 baseline 测试中至少有 1 次产生 `original_ok=False`，**或** 5 维度 stress 测试中至少有 1 个测试用例检出 discrepancy。

**判定的逻辑依据**:

1. **两个阶段的输出不一致都是确凿阳性证据**：无论是默认随机输入还是极端输入，只要 LLM 内核输出与 PyTorch 参考实现的输出在容差外不一致，都直接证明该内核存在缺陷
2. **与原始基准范式兼容**：KernelBench、CUDA-L1、AI-CUDA-Engineer、TritonBench、CUDA-Agent 等原始基准均仅用类似 baseline 的判定，本工作的联合口径**完全包含**原始基准的判定 + 5 维度 stress 增量证据
3. **避免单一阶段的测量盲区**：单纯依赖 stress 会漏掉 stress 阶段因 OOM/依赖缺失/NaN 回退等技术原因无法判定但 baseline 已经证实有缺陷的内核（详见 §8）


| 指标             | 公式                                       | 含义                                |
| -------------- | ---------------------------------------- | --------------------------------- |
| 完成测试率          | Completed / Total                        | 内核可编译运行的比例                        |
| **联合检出率（主指标）** | **(                                      | D(k)=1                            |
| Baseline 阶段检出率 | 仅 baseline.failed > 0 的内核数 / Completed   | 标准输入下检出的缺陷比例                      |
| Stress 阶段独立增量  | (Stress 检出 ∖ Baseline 检出) / Completed    | Stress 在 baseline 之外**新增**检出的缺陷比例 |
| Stress 阶段总检出   | total_discrepancies > 0 的内核数 / Completed | 含与 baseline 重合的全部 stress 检出       |


**集合论关系**（$B$ = baseline 阳性集合，$S$ = stress 阳性集合）：

$$\text{联合检出} = |B \cup S| = |B| + |S \setminus B|$$
$$\text{Stress 独立增量} = |S \setminus B|$$
$$\text{Baseline-Stress 共同检出} = |B \cap S|$$
$$\text{Baseline 独占检出} = |B \setminus S|$$

---

## 2. 总体统计

> **数据来源**: 各数据集 `checkpoint.json` 中所有条目的 `status`、`baseline.failed`、`total_discrepancies` 字段

### 2.1 各数据集联合检出率（主指标）

> 应用 §1.5 联合判定准则：$D(k) = \mathbb{1}[\text{baseline.failed}>0] \lor \mathbb{1}[\text{totaldiscrepancies}>0]$


| 数据集              | 总数      | Completed | Skipped | Baseline 阳性 (B) | Stress 独立增量 (S\B) | **联合检出 (B∪S)** | **联合检出率**  |
| ---------------- | ------- | --------- | ------- | --------------- | ----------------- | -------------- | ---------- |
| CUDA-L1          | 241     | 231       | 10      | 101             | 60                | **161**        | **69.70%** |
| AI-CUDA-Engineer | 229     | 222       | 7       | 25              | 50                | **75**         | **33.78%** |
| TritonBench-G    | 141     | 138       | 3       | 39              | 11                | **50**         | **36.23%** |
| CUDA-Agent       | 220     | 176       | 44      | 5               | 101               | **106**        | **60.23%** |
| **合计**           | **831** | **767**   | **64**  | **170**         | **222**           | **392**        | **51.11%** |


> **完整性校验**:
>
> - 总数: 241 + 229 + 141 + 220 = 831 ✓
> - Completed: 231 + 222 + 138 + 176 = 767 ✓
> - Skipped: 10 + 7 + 3 + 44 = 64（reason 均为 `baseline_all_error`）✓
> - 联合检出: 161 + 75 + 50 + 106 = 392 ✓；分解 |B|+|S\B| = 170 + 222 = 392 ✓
> - 联合检出率：392 / 767 = 51.11% ✓

### 2.2 双阶段检出贡献分解

> **数据来源**: 对所有 767 个 COMPLETED 内核交叉分析 `baseline.failed > 0` 与 `total_discrepancies > 0`

下表把 767 个 Completed 内核按"baseline 阶段是否阳性"× "stress 阶段是否阳性"做四象限分类：


| 类别                                            | 含义                               | CUDA-L1 | AI-CUDA-Engineer | TritonBench-G | CUDA-Agent | 合计      |
| --------------------------------------------- | -------------------------------- | ------- | ---------------- | ------------- | ---------- | ------- |
| **B∩S** Baseline 阳性 **且** Stress 阳性           | 两阶段独立确认的缺陷交集                     | 66      | 24               | 19            | 3          | **112** |
| **B\S** Baseline 阳性 **且** Stress 阴性           | 仅 baseline 检出的缺陷                 | 35      | 1                | 20            | 2          | **58**  |
| **S\B** Baseline 阴性 **且** Stress 阳性（因为资源问题挂掉） | **仅 stress 检出的新缺陷**（stress 增量价值） | 60      | 50               | 11            | 101        | **222** |
| **¬(B∪S)** Baseline 阴性 **且** Stress 阴性        | 干净内核（在双阶段测试下均与参考一致）              | 70      | 147              | 88            | 70         | **375** |
| **小计 (Completed)**                            | —                                | **231** | **222**          | **138**       | **176**    | **767** |


> 校验: 66+35+60+70 = 231 ✓；24+1+50+147 = 222 ✓；19+20+11+88 = 138 ✓；3+2+101+70 = 176 ✓
>
> 联合检出 |B∪S| = |B∩S| + |B\S| + |S\B| = 112 + 58 + 222 = 392 ✓

### 2.3 Stress 阶段相对 Baseline 的增量价值


| 数据集              | 联合检出 (B∪S) | Stress 独立增量 (S\B) | Stress 增量贡献率 |
| ---------------- | ---------- | ----------------- | ------------ |
| CUDA-L1          | 161        | 60                | **37.27%**   |
| AI-CUDA-Engineer | 75         | 50                | **66.67%**   |
| TritonBench-G    | 50         | 11                | **22.00%**   |
| CUDA-Agent       | 106        | 101               | **95.28%**   |
| **合计**           | **392**    | **222**           | **56.63%**   |


**关键论证**: 5 维度 stress 测试在 767 个 Completed 内核中**单独检出 222 个**仅靠 baseline 漏掉的缺陷（即 Baseline 阴性**且** Stress 阳性的内核），占总联合检出 392 的 **56.63%**。这是本工作相对原始基准（KernelBench、CUDA-L1、AI-CUDA-Engineer、TritonBench、CUDA-Agent 仅用 baseline 范式）最直接的方法学贡献。

**数据集差异解读**:

- **CUDA-Agent 增量贡献率 95.28%（最高）**: ByteDance & THU 的 SOTA Agentic RL 系统声称 KernelBench 总体 pass rate 达 98.8%，本实验 baseline 阶段确实只有 5/176 = 2.84% 内核失败（与 ByteDance 声明一致），**但 stress 测试又额外发现了 101 个 baseline 全通过的有缺陷内核**，使联合检出率达到 60.23%。这一发现强烈支持："SOTA 训练时正确性验证通过"远不等于"在生产级输入分布下鲁棒"。
- **AI-CUDA-Engineer 增量贡献率 66.67%**：LLM Agent 生成的内核在标准输入下"看起来正常"，但在 stress 测试下大量失效，说明 LLM 容易过拟合训练分布、缺乏边界鲁棒性
- **CUDA-L1 增量贡献率 37.27%**：RL 优化的内核在 baseline 阶段已经暴露大量缺陷（101/231 = 43.7% baseline 失败），stress 仍能再发现 60 个
- **TritonBench-G 增量贡献率 22.00%（最低）**：人工编写的开源内核 baseline 失败率较高（39/138 = 28.3%），但成熟代码在极端输入下仍较稳健，stress 增量较小

---

## 3. 五维度差分测试结果

### 3.1 各维度测试覆盖与检出统计

> **数据来源**: 各 `checkpoint.json` 中每个内核的 5 个维度子字段中的 `discrepancies` 和 `passes` 值
>
> **覆盖说明**: 部分内核因不支持特定 dtype 或运行时崩溃，相应维度无有效结果（discrepancies=0 且 passes=0），不计入"测试覆盖内核数"。

#### CUDA-L1（231 个 COMPLETED 内核）


| 维度              | 测试覆盖 | 检出差异 | 维度检出率     | 数据来源                              |
| --------------- | ---- | ---- | --------- | --------------------------------- |
| value_stress    | 193  | 110  | **57.0%** | value_stress.discrepancies > 0    |
| training_stress | 193  | 113  | **58.5%** | training_stress.discrepancies > 0 |
| repeated_run    | 215  | 63   | **29.3%** | repeated_run.discrepancies > 0    |
| config_stress   | 203  | 49   | **24.1%** | config_stress.discrepancies > 0   |
| dtype_stress    | 186  | 44   | **23.7%** | dtype_stress.discrepancies > 0    |


#### AI-CUDA-Engineer（222 个 COMPLETED 内核）


| 维度              | 测试覆盖 | 检出差异 | 维度检出率     |
| --------------- | ---- | ---- | --------- |
| value_stress    | 221  | 67   | **30.3%** |
| training_stress | 221  | 65   | **29.4%** |
| repeated_run    | 221  | 29   | **13.1%** |
| config_stress   | 219  | 22   | **10.0%** |
| dtype_stress    | 42   | 4    | **9.5%**  |


> **dtype_stress 覆盖较低（42/222）的原因**: AI-CUDA-Engineer 的内核多为 KernelBench L1 算子，其 reference Model 仅接受 float32 输入，调用 `_to_dtype` 切换为 fp16/bf16 时，参考实现自身就会因 dtype 不匹配抛错，因此该维度大量 no_data。

#### TritonBench-G（138 个 COMPLETED 内核）


| 维度              | 测试覆盖 | 检出差异 | 维度检出率     |
| --------------- | ---- | ---- | --------- |
| value_stress    | 120  | 22   | **18.3%** |
| training_stress | 120  | 23   | **19.2%** |
| repeated_run    | 138  | 24   | **17.4%** |
| config_stress   | 121  | 9    | **7.4%**  |
| dtype_stress    | 123  | 5    | **4.1%**  |


#### CUDA-Agent（176 个 COMPLETED 内核）


| 维度              | 测试覆盖 | 检出差异 | 维度检出率     |
| --------------- | ---- | ---- | --------- |
| value_stress    | 174  | 90   | **51.7%** |
| training_stress | 174  | 94   | **54.0%** |
| repeated_run    | 156  | 15   | **9.6%**  |
| config_stress   | 132  | 4    | **3.0%**  |
| dtype_stress    | 109  | 11   | **10.1%** |


> **CUDA-Agent 各维度覆盖差异说明**:
>
> - **dtype_stress 覆盖 109/176**（62.0%）: 原因与 AI-CUDA-Engineer 类似——CUDA-Agent 多数内核为 KernelBench L1 算子，reference Model 仅接受 float32 输入，切换到 fp16/bf16 时参考实现自身抛出 `Input must be float32` 错误（详细记录见 detail JSON 文件中的 `dtype_stress.test_cases[].error` 字段）
> - **config_stress 覆盖 132/176**（75.0%）: 部分内核在小 batch（batch=1）或大 batch（batch=64）下 reference 与 LLM 内核有一方崩溃，导致 batch 维度无可比较结果
> - **repeated_run 覆盖 156/176**（88.6%）: 部分内核因 5 次重复运行的累计耗时超过 worker 超时阈值（600s）而无法完成

### 3.2 跨维度联合检出分析

> **数据来源**: 各检出差异内核的 `discrepant_dimensions` 数组长度


| 数据集              | Stress 检出差异内核数 | 多维度同时检出 | 仅单维度检出 | 多维度占比     |
| ---------------- | -------------- | ------- | ------ | --------- |
| CUDA-L1          | 126            | 109     | 17     | **86.5%** |
| AI-CUDA-Engineer | 74             | 65      | 9      | **87.8%** |
| TritonBench-G    | 30             | 27      | 3      | **90.0%** |
| CUDA-Agent       | 104            | 92      | 12     | **88.5%** |
| **合计**           | **334**        | **293** | **41** | **87.7%** |


**检出维度数分布**:


| 检出维度数  | CUDA-L1 | AI-CUDA-Engineer | TritonBench-G | CUDA-Agent | 合计      |
| ------ | ------- | ---------------- | ------------- | ---------- | ------- |
| 1 维度   | 17      | 9                | 3             | 12         | 41      |
| 2 维度   | 44      | 35               | 9             | 77         | 165     |
| 3 维度   | 22      | 12               | 14            | 12         | 60      |
| 4 维度   | 7       | 18               | 0             | 3          | 28      |
| 5 维度   | 36      | 0                | 4             | 0          | 40      |
| **合计** | **126** | **74**           | **30**        | **104**    | **334** |


> 校验: CUDA-L1: 17+44+22+7+36 = 126 ✓；AI-CUDA-Engineer: 9+35+12+18+0 = 74 ✓；TritonBench-G: 3+9+14+0+4 = 30 ✓；CUDA-Agent: 12+77+12+3+0 = 104 ✓
>
> **AI-CUDA-Engineer 与 CUDA-Agent 全无 5 维度同时检出**：因为这两个数据集中 dtype_stress 覆盖率较低（AI-CUDA-Engineer 42/222；CUDA-Agent 109/176），难以与其他四维度同时检出。

### 3.3 各维度的独立侦测能力（消融分析）

> **本节回答的问题**：在已被 stress 测试发现存在缺陷的内核集合上，**如果只使用某一个维度，能召回多少？会漏掉多少？**——即每个维度作为独立侦测器的能力评估。

**计算口径**:

- **分母**: 各数据集中 `total_discrepancies > 0` 的内核数（即 stress 已检出的有缺陷内核数）
  - CUDA-L1 = 126；AI-CUDA-Engineer = 74；TritonBench-G = 30；CUDA-Agent = 104
- **分子**: 该维度的 `discrepancies > 0` 即被该维度独立抓到的内核数（来自 `discrepant_dimensions` 字段）
- **注意**: 同一个内核可被多个维度同时抓到，因此每列各行**不互斥**，相加 > 分母（这正是 §3.2 多维度交叉验证的体现）

#### CUDA-L1（126 个 stress 已检出内核）


| 维度              | 抓到  | 漏检  | 召回率       | 单维度独立侦测能力  |
| --------------- | --- | --- | --------- | ---------- |
| value_stress    | 110 | 16  | **87.3%** | 强（漏 12.7%） |
| training_stress | 113 | 13  | **89.7%** | 强（漏 10.3%） |
| repeated_run    | 63  | 63  | **50.0%** | 中（漏一半）     |
| config_stress   | 49  | 77  | **38.9%** | 中弱         |
| dtype_stress    | 44  | 82  | **34.9%** | 中弱         |


#### AI-CUDA-Engineer（74 个 stress 已检出内核）


| 维度              | 抓到  | 漏检  | 召回率       | 单维度独立侦测能力                       |
| --------------- | --- | --- | --------- | ------------------------------- |
| value_stress    | 67  | 7   | **90.5%** | 强                               |
| training_stress | 65  | 9   | **87.8%** | 强                               |
| repeated_run    | 29  | 45  | **39.2%** | 中弱                              |
| config_stress   | 22  | 52  | **29.7%** | 弱                               |
| dtype_stress    | 4   | 70  | **5.4%**  | 极弱（受限于该数据集仅 42 个内核支持 fp16/bf16） |


#### TritonBench-G（30 个 stress 已检出内核）


| 维度              | 抓到  | 漏检  | 召回率       | 单维度独立侦测能力               |
| --------------- | --- | --- | --------- | ----------------------- |
| repeated_run    | 24  | 6   | **80.0%** | 强（人工 Triton 内核非确定性问题突出） |
| training_stress | 23  | 7   | **76.7%** | 强                       |
| value_stress    | 22  | 8   | **73.3%** | 强                       |
| config_stress   | 9   | 21  | **30.0%** | 弱                       |
| dtype_stress    | 5   | 25  | **16.7%** | 弱                       |


#### CUDA-Agent（104 个 stress 已检出内核）


| 维度              | 抓到  | 漏检  | 召回率       | 单维度独立侦测能力 |
| --------------- | --- | --- | --------- | --------- |
| training_stress | 94  | 10  | **90.4%** | 强         |
| value_stress    | 90  | 14  | **86.5%** | 强         |
| repeated_run    | 15  | 89  | **14.4%** | 弱         |
| dtype_stress    | 11  | 93  | **10.6%** | 弱         |
| config_stress   | 4   | 100 | **3.8%**  | 极弱        |


**跨四数据集汇总（合计 334 个 stress 已检出内核）**:


| 维度              | 累计抓到 | 累计漏检 | 累计召回率     |
| --------------- | ---- | ---- | --------- |
| training_stress | 295  | 39   | **88.3%** |
| value_stress    | 289  | 45   | **86.5%** |
| repeated_run    | 131  | 203  | **39.2%** |
| config_stress   | 84   | 250  | **25.1%** |
| dtype_stress    | 64   | 270  | **19.2%** |


**关键观察**:

1. **value_stress 与 training_stress 是跨四数据集稳定的核心维度**：累计召回率分别为 86.5% / 88.3%，在 CUDA-L1、AI-CUDA-Engineer、CUDA-Agent 三个 LLM/RL 数据集上召回率均 ≥ 86%，单独使用任一维度都能召回近九成缺陷
2. **TritonBench-G 上 repeated_run 跃升为最强维度（80.0%）**：人工编写的 Triton 内核中非确定性问题（如 atomic 竞争、未同步 shared memory）显著高于三个 LLM/RL 数据集
3. **dtype_stress 在 AI-CUDA-Engineer / CUDA-Agent 上召回率仅 5.4% / 10.6%**：不是该维度无效，而是这两个数据集的 reference Model 多数仅支持 fp32 输入（覆盖率分别为 42/222、109/176）；这反映 KernelBench-style 数据集本身在数据类型多样性上的局限
4. **即使单一最强维度（training_stress 88.3%）也会漏检 39/334 个缺陷**，§3.4 独占检出分析进一步证明"5 维度联合不可替代"

### 3.4 独占检出分析（仅由单一维度检出的内核）

> **数据来源**: `discrepant_dimensions` 数组长度为 1 的内核


| 维度              | CUDA-L1 | AI-CUDA-Engineer | TritonBench-G | CUDA-Agent | 合计     |
| --------------- | ------- | ---------------- | ------------- | ---------- | ------ |
| value_stress    | 2       | 4                | 0             | 0          | 6      |
| training_stress | 7       | 2                | 1             | 4          | 14     |
| repeated_run    | 1       | 0                | 1             | 5          | 7      |
| config_stress   | 5       | 0                | 0             | 2          | 7      |
| dtype_stress    | 2       | 3                | 1             | 1          | 7      |
| **合计**          | **17**  | **9**            | **3**         | **12**     | **41** |


> **设计验证**: 41 个内核仅由单一维度检出差异，说明每个测试维度都不可替代——若移除任一维度，将漏检对应数量的缺陷。其中：
>
> - 移除 training_stress 将漏检 14 个缺陷（最多）
> - 移除 repeated_run、config_stress、dtype_stress 各将漏检 7 个缺陷
> - 移除 value_stress 将漏检 6 个缺陷

---

## 4. 值域压力策略（value_stress）详细分析

> **数据来源**: 各 `checkpoint.json` 中每个内核 `value_stress.details` 字段（键为策略名，值为 `"discrepancy"` 表示该策略至少有 1 个 seed 检出差异）
>
> **统计粒度**: 一个内核如果在某策略的 2 个 seed 中至少有 1 个产生 discrepancy，即计为该策略检出该内核 1 次。

### 4.1 各策略在四个数据集中的检出内核数


| 策略                    | CUDA-L1 | AI-CUDA-Engineer | TritonBench-G | CUDA-Agent | 合计      |
| --------------------- | ------- | ---------------- | ------------- | ---------- | ------- |
| extreme_magnitude     | 104     | 47               | 20            | 75         | **246** |
| tail_heavy            | 92      | 41               | 20            | 75         | 228     |
| head_heavy            | 91      | 39               | 20            | 74         | 224     |
| mixed_extremes        | 89      | 37               | 20            | 72         | 218     |
| large_magnitude       | 88      | 38               | 20            | 69         | 215     |
| sparse_extreme        | 85      | 35               | 19            | 67         | 206     |
| alternating_sign      | 85      | 37               | 20            | 60         | 202     |
| near_overflow         | 88      | 38               | 10            | 65         | 201     |
| reduction_adversarial | 73      | 42               | 20            | 55         | 190     |
| all_positive          | 83      | 35               | 20            | 45         | 183     |
| init_sensitive        | 82      | 33               | 20            | 45         | 180     |
| sparse                | 79      | 30               | 20            | 50         | 179     |
| all_negative          | 80      | 34               | 19            | 43         | 176     |
| uniform_constant      | 69      | 34               | 19            | 32         | 154     |
| boundary_last_element | 70      | 26               | 20            | 22         | 138     |
| relop_boundary_hit    | 69      | 28               | 16            | 4          | 117     |
| dense_nonzero         | 68      | 25               | 17            | 4          | 114     |
| structured_ramp       | 67      | 23               | 16            | 6          | 112     |
| near_epsilon          | 48      | 19               | 17            | 2          | 86      |
| near_zero             | 46      | 18               | 17            | 2          | 83      |
| denormals             | 46      | 18               | 16            | 2          | 82      |


> **观察**:
>
> - 21 种策略中，`extreme_magnitude` (±1e6 量级) 检出能力最强，累计在 246 个内核中检出差异；`denormals`（次正规数）最弱，但仍能在 82 个内核中检出差异
> - **不同策略检出能力差距 3.0 倍**（最强 246 vs 最弱 82），但所有策略都有不可替代的贡献，没有任何策略可被完全淘汰
> - **CUDA-Agent 上策略检出能力两极分化最显著**: 顶部策略（如 `extreme_magnitude`、`tail_heavy`）检出 65-75 个内核，但 `denormals`/`near_zero`/`near_epsilon` 仅 2 个——说明 ByteDance & THU 的 Agentic RL 系统对量级类极端输入鲁棒性最弱，但对次正规数和 epsilon 邻域的处理（多由 PyTorch IR/编译器生成的代码默认处理）相对稳定

---

## 5. 按难度级别的分析（联合检出口径）

> **数据来源**: 内核 ID 中的 level 前缀（如 `cuda_l1__L1_T1` → L1）；对应 `external_benchmarks/{dataset}/registry.json` 中的 `level_id` 字段
>
> **检出口径**: 应用 §1.5 联合判定准则 $D(k) = \mathbb{1}[\text{baseline.failed}>0] \lor \mathbb{1}[\text{totaldiscrepancies}>0]$

### 5.1 CUDA-L1 难度分级结果


| 级别       | Completed | Skipped | Baseline 阳性 (B) | Stress 独立增量 (S\B) | 联合检出 (B∪S) | **联合检出率**  |
| -------- | --------- | ------- | --------------- | ----------------- | ---------- | ---------- |
| L1（基础算子） | 84        | 9       | 37              | 11                | 48         | **57.14%** |
| L2（算子序列） | 99        | 0       | 37              | 33                | 70         | **70.71%** |
| L3（完整模型） | 48        | 1       | 27              | 16                | 43         | **89.58%** |
| **小计**   | **231**   | **10**  | **101**         | **60**            | **161**    | **69.70%** |


> **观察**: L1 → L2 → L3，联合检出率从 57.14% 单调递增到 89.58%，**强正相关**。说明 RL 优化方法在简单算子上有相对的稳定性，但在复杂模型上几乎九成内核与 PyTorch 参考行为不一致——这是论文中关于"AI 生成内核在复杂任务上失效率随复杂度增加"的核心证据之一。

### 5.2 AI-CUDA-Engineer 难度分级结果


| 级别       | Completed | Skipped | Baseline 阳性 (B) | Stress 独立增量 (S\B) | 联合检出 (B∪S) | **联合检出率**  |
| -------- | --------- | ------- | --------------- | ----------------- | ---------- | ---------- |
| L1（基础算子） | 87        | 4       | 7               | 21                | 28         | **32.18%** |
| L2（算子序列） | 96        | 2       | 11              | 20                | 31         | **32.29%** |
| L3（完整模型） | 39        | 1       | 7               | 9                 | 16         | **41.03%** |
| **小计**   | **222**   | **7**   | **25**          | **50**            | **75**     | **33.78%** |


> **观察**: L1 与 L2 联合检出率几乎相同（32.18% vs 32.29%），L3 略高（41.03%），与 CUDA-L1 的强相关曲线明显不同。AI-CUDA-Engineer 各难度级别上 baseline 阳性比例均较低（≤14%），但 stress 在各级难度上都有显著独立增量贡献（21/0/9 vs 0/0/1，即仅 1 个内核出现 baseline 阳性而 stress 阴性的情况），表明 LLM Agent 生成的内核在标准随机输入下表现稳定但**在极端输入下普遍存在边界鲁棒性问题**。

### 5.3 CUDA-Agent 难度分级结果


| 级别       | Completed | Skipped | Baseline 阳性 (B) | Stress 独立增量 (S\B) | 联合检出 (B∪S) | **联合检出率**  |
| -------- | --------- | ------- | --------------- | ----------------- | ---------- | ---------- |
| L1（基础算子） | 87        | 1       | 1               | 37                | 38         | **43.68%** |
| L2（算子序列） | 62        | 32      | 0               | 46                | 46         | **74.19%** |
| L3（完整模型） | 27        | 11      | 4               | 18                | 22         | **81.48%** |
| **小计**   | **176**   | **44**  | **5**           | **101**           | **106**    | **60.23%** |


> **观察**:
>
> - L1 → L2 → L3，联合检出率从 43.68% 单调递增到 81.48%，**与 CUDA-L1 类似的强正相关**，再次印证"AI 自动生成内核在复杂任务上失效率随复杂度增加"的论点
> - **CUDA-Agent L2 级别 SKIPPED 比例显著（32/94 = 34.0%）**: 大量 L2 内核因编译失败/运行崩溃而无法进入 5 维度测试，这反映该 SOTA 系统在算子融合任务上的稳定性问题
> - **L1 级别尤其值得关注**: 87 个完成测试的 L1 内核中，仅 1 个 baseline 失败但 stress 独立又发现 37 个新缺陷——这意味着对最简单的原子算子，CUDA-Agent 也存在 42.5% (37/87) 的潜在鲁棒性缺陷被现有验证流程漏检

### 5.4 TritonBench-G

TritonBench-G 数据集来自 55 个开源仓库的实际生产代码，无 KernelBench 三级难度分级。其 138 个 COMPLETED 内核的联合检出情况：


| 维度                | 数值  | 占比         |
| ----------------- | --- | ---------- |
| Baseline 阳性 (B)   | 39  | 28.26%     |
| Stress 独立增量 (S\B) | 11  | 7.97%      |
| 联合检出 (B∪S)        | 50  | **36.23%** |
| 双阶段共同检出 (B∩S)     | 19  | —          |


值得关注的是：在已经被开源社区代码审查过的人工内核中，stress 测试仍**单独检出 11 个 baseline 漏掉的鲁棒性问题**（占总联合检出的 22.0%），这进一步验证 5 维度差分测试方法在工业级人工编写代码上同样有效。

### 5.5 三个 LLM/RL 数据集的难度敏感性对比

> **本节回答**：不同 LLM/RL 生成范式在 KernelBench 三级难度上呈现什么样的失效模式？


| 难度级别         | CUDA-L1（RL）  | AI-CUDA-Engineer（LLM Agent） | CUDA-Agent（Agentic RL） |
| ------------ | ------------ | --------------------------- | ---------------------- |
| L1 联合检出率     | 57.14%       | 32.18%                      | 43.68%                 |
| L2 联合检出率     | 70.71%       | 32.29%                      | 74.19%                 |
| L3 联合检出率     | 89.58%       | 41.03%                      | 81.48%                 |
| **L3-L1 增长** | **+32.4 pp** | **+8.9 pp**                 | **+37.8 pp**           |
| 难度敏感性        | 强正相关         | 弱相关                         | 强正相关                   |


> **核心发现**:
>
> - **基于 RL 的方法（CUDA-L1、CUDA-Agent）对难度敏感**：L3 失效率比 L1 高 32.4-37.8 pp，反映 RL 优化在长序列依赖（完整模型）上更容易过拟合训练时奖励信号
> - **基于 LLM Agent 的方法（AI-CUDA-Engineer）对难度近似无关**：L3 比 L1 仅高 8.9 pp，反映 LLM 生成的"系统性"缺陷在各级难度上分布较均匀
> - **CUDA-Agent 虽然采用 Agentic RL（结合 LLM 推理与 RL 优化），但失效模式仍偏向 RL 类**：L1 → L3 检出率单调递增，与 CUDA-L1 高度相似

---

## 6. 执行时间统计

> **数据来源**: 各 `checkpoint.json` 中每个内核的 `elapsed_s` 字段（仅 COMPLETED 内核）


| 数据集              | 总耗时（小时）    | 平均耗时（秒/内核） | 最短（秒） | 最长（秒）  |
| ---------------- | ---------- | ---------- | ----- | ------ |
| CUDA-L1          | 56.28      | 877.1      | 167.0 | 5419.5 |
| AI-CUDA-Engineer | 15.30      | 248.1      | 164.4 | 1158.7 |
| TritonBench-G    | 9.30       | 242.5      | 171.5 | 1263.2 |
| CUDA-Agent       | 40.55      | 829.3      | 167.8 | 3276.6 |
| **合计**           | **121.43** | **569.9**  | —     | —      |


> **观察**: CUDA-L1 与 CUDA-Agent 平均耗时显著高于 AI-CUDA-Engineer / TritonBench-G（877s / 829s vs ~245s）。CUDA-L1 的高耗时主要由于其 RL 生成的内核包含大量"自适应基准选择"代码（含 warmup、benchmarking 子流程），单次前向传播时间是普通实现的数倍；CUDA-Agent 的高耗时则源于其 Agentic 设计中嵌入的多轮验证/profile 子流程，以及对 L3 大模型场景的偏好（27 个 L3 内核平均耗时较 L1/L2 更长）。

---

## 7. Baseline 测试与编译可行性

> **数据来源**: 各 `checkpoint.json` 中每个内核的 `baseline.passed`、`baseline.failed`、`baseline.errors` 字段

### 7.1 各数据集 Baseline 分布

> **统计粒度**: 每个 COMPLETED 内核执行 3 次 baseline 测试（3 个 seed，identity policy），下表为单次 baseline 测试的累计计数。


| 数据集              | Completed 内核数 | Baseline 通过总次数 | Baseline 失败总次数 | Baseline 错误总次数 |
| ---------------- | ------------- | -------------- | -------------- | -------------- |
| CUDA-L1          | 231           | 390            | 303            | 0              |
| AI-CUDA-Engineer | 222           | 596            | 70             | 0              |
| TritonBench-G    | 138           | 299            | 115            | 0              |
| CUDA-Agent       | 176           | 513            | 15             | 0              |
| **合计**           | **767**       | **1798**       | **503**        | **0**          |


> 校验: 每数据集应有 Completed × 3 次测试。CUDA-L1: 231×3=693 = 390+303+0 ✓；AI-CUDA-Engineer: 222×3=666 = 596+70+0 ✓；TritonBench-G: 138×3=414 = 299+115+0 ✓；CUDA-Agent: 176×3=528 = 513+15+0 ✓

### 7.2 按内核分类的 Baseline 结果

> **统计粒度**: 按内核分组，统计每个内核的 3 次 baseline 中通过的次数


| 数据集              | 全部通过 (3/3) | 部分通过 (1-2) | 全部失败但能运行 (0/3) | SKIPPED（全 errors） |
| ---------------- | ---------- | ---------- | -------------- | ----------------- |
| CUDA-L1          | 130        | 0          | 101            | 10                |
| AI-CUDA-Engineer | 197        | 4          | 21             | 7                 |
| TritonBench-G    | 99         | 1          | 38             | 3                 |
| CUDA-Agent       | 171        | 0          | 5              | 44                |
| **合计**           | **597**    | **5**      | **165**        | **64**            |


> 校验: CUDA-L1: 130+0+101+10=241 ✓；AI-CUDA-Engineer: 197+4+21+7=229 ✓；TritonBench-G: 99+1+38+3=141 ✓；CUDA-Agent: 171+0+5+44=220 ✓

> **解读**:
>
> - **597 个内核 (71.8%) 全部 baseline 通过**：在标准随机输入下表现正常，但其中 222 个（来自 §2.3 的统计）仍被 stress 测试检出缺陷
> - **165 个内核 (19.9%) baseline 全部失败但能编译运行**：在标准输入下输出已与参考不一致，属于"显性缺陷"
> - **64 个内核 (7.7%) SKIPPED**：完全无法通过编译/运行，已从分母中剔除
> - **CUDA-Agent 是 4 个数据集中 SKIPPED 比例最高的（44/220 = 20.0%）**：反映该 SOTA Agentic RL 系统在某些任务（特别是 L2 算子融合，32/94 = 34.0% SKIPPED）上的内核稳定性问题

### 7.3 CUDA-Agent 的 Baseline 通过率与原作者声明对比

ByteDance 与清华联合发布的 CUDA-Agent 论文（arXiv:2602.24286）声称在 KernelBench 上达到 **98.8% 总体 pass rate**。在我们的实验中：


| 指标                                          | 数值                  |
| ------------------------------------------- | ------------------- |
| ByteDance & THU 论文声明的 KernelBench pass rate | 98.8%               |
| 本实验 baseline 通过率（按内核计，含 SKIPPED）            | 171/220 = **77.7%** |
| 本实验 baseline 通过率（仅 Completed）               | 171/176 = **97.2%** |
| 本实验 baseline 失败率（在 Completed 内核上）           | 5/176 = **2.84%**   |


> **数据可比性说明**: 排除 SKIPPED（编译/运行崩溃，相当于原论文中的"无法运行"类）后，我们的 baseline 通过率 97.2% 与 ByteDance 声明的 98.8% 在同一量级。差异 1.6 pp 可归因于 (1) 我们使用 3 个独立 seed 而非单一 seed 验证，(2) 我们采用 KernelBench 统一容差 atol=rtol=1e-2 而非原论文训练时容差，(3) 测试硬件/驱动差异。**这一独立验证支持了原作者关于其系统 baseline 正确性的声明**。
>
> **但本工作的核心发现是**: 即使在 baseline 通过率高达 97.2% 的情况下，5 维度 stress 测试仍在剩余 171 个 baseline 全过的内核中**新发现 101 个鲁棒性缺陷**（详见 §2.3），即 baseline-confirmed-correct 的内核中仍有 **101/171 = 59.06%** 在极端输入下与 PyTorch 参考行为不一致。这是论文最强的方法学论据：**SOTA 系统的 release-time 正确性验证远不足以保证生产部署中的鲁棒性**。

### 7.4 内核来源仓库分析（TritonBench-G）

> **数据来源**: `外部Benchmark差分测试_RQ4/TritonBench-G/checkpoint.json` 中各内核的 `repo` 字段，仅统计 COMPLETED 内核

TritonBench-G 数据集的 138 个已完成测试的内核来自 55 个不同的开源项目，下表展示前 12 个主要来源（覆盖 83 个内核，占 60.1%）：


| 仓库                                   | 内核数量    |
| ------------------------------------ | ------- |
| ModelTC/lightllm                     | 17      |
| S-LoRA/S-LoRA                        | 9       |
| triton-lang/triton                   | 8       |
| sustcsonglin/flash-linear-attention  | 8       |
| FlagOpen/FlagGems                    | 6       |
| bitsandbytes-foundation/bitsandbytes | 5       |
| MzeroMiko/VMamba                     | 5       |
| linkedin/Liger-Kernel                | 5       |
| ELS-RD/kernl                         | 5       |
| thu-ml/SageAttention                 | 5       |
| hpcaitech/ColossalAI                 | 5       |
| josStorer/RWKV-Runner                | 5       |
| 其他 43 个仓库                            | 55      |
| **合计**                               | **138** |


---

## 8. 关键结论

1. **双阶段联合检出框架**：MutaKernel 采用 baseline + 5 维度 stress 两阶段检测，缺陷判定为 $D(k) = \mathbb{1}[\text{baseline.failed}>0] \lor \mathbb{1}[\text{totaldiscrepancies}>0]$，覆盖原始基准范式 + stress 增量。**四数据集合计 392 个内核被检出缺陷，联合检出率 51.11%（392/767）**。
2. **Stress 测试相对原始基准 baseline 提供 56.63% 的独立增量检出**：在 767 个 Completed 内核中，222 个内核 baseline 全部通过但被 5 维度 stress 检出，占总联合检出 392 的 56.63%。这是本工作相对仅依赖 baseline 的原始基准范式最直接的方法学贡献。
3. **针对 SOTA Agentic RL 系统（CUDA-Agent）的强力实证**：ByteDance & THU 联合发布的 CUDA-Agent 系统在 KernelBench 上声称 98.8% pass rate（本实验独立验证为 97.2%）。然而，在其声称正确通过的 171 个 baseline-pass 内核中，5 维度 stress 测试又发现 **101 个（59.06%）** 在极端输入下与 PyTorch 参考行为不一致。CUDA-Agent 上 stress 独立增量贡献率高达 **95.28%**——这是论文最强的论据：SOTA 系统的 release-time 正确性验证远不足以保证生产部署中的鲁棒性。
4. **不同生成方法的失效模式差异显著**:
  - CUDA-L1（RL 自动生成）: 联合检出率 **69.70%**；难度强相关（L1: 57.14% → L3: 89.58%）
  - CUDA-Agent（Agentic RL 自动生成）: 联合检出率 **60.23%**；难度强相关（L1: 43.68% → L3: 81.48%）
  - AI-CUDA-Engineer（LLM Agent 自动生成）: 联合检出率 **33.78%**；难度近似无关（L1: 32.18% ≈ L3: 41.03%）；stress 独立增量贡献率高达 66.67%
  - TritonBench-G（人工编写，开源生产代码）: 联合检出率 **36.23%**；其中 stress 仍单独检出 11 个 baseline 漏掉的缺陷
5. **多维度交叉验证增强 stress 检出的可信度**: 334 个 stress 阶段检出内核中，**87.7% 由 2 个及以上维度独立确认**，证明 stress 缺陷判定具有高可靠性，非偶然抖动。
6. **每个测试维度都不可替代**: 41 个内核仅由单一维度检出（training_stress 独占 14 个，repeated_run/config_stress/dtype_stress 各 7 个，value_stress 独占 6 个），任一维度被移除将漏检对应缺陷。
7. **value_stress 与 training_stress 是核心检测维度**: 两者在 stress 阶段已检出内核中的累计召回率分别为 86.5% / 88.3%（CUDA-L1: 87.3% / 89.7%；AI-CUDA-Engineer: 90.5% / 87.8%；TritonBench-G: 73.3% / 76.7%；CUDA-Agent: 86.5% / 90.4%）。
8. `**extreme_magnitude` 是最有效的单一压力策略**: 在四个数据集中累计在 246 个内核中检出差异；最弱策略 `denormals` 也能检出 82 个，**21 种策略均有独特贡献**。
9. **数据完整性**: 全部 767 个 Completed 内核均通过单内核维度一致性校验（`sum(各维度 discrepancies) == total_discrepancies`，`discrepant_dimensions` 与各维度实际值一致），64 个 SKIPPED 内核均明确标注 `reason=baseline_all_error`。

---

## 9. 数据完整性验证


| 验证项                          | 验证方法                                                   | 结果                                                                                   |
| ---------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------ |
| 总数 = Completed + Skipped     | 各数据集计数加和                                               | ✓（241=231+10；229=222+7；141=138+3；220=176+44）                                         |
| Stress 检出与未检出之和 = Completed  | 各数据集 `total_discrepancies>0` 计数                        | ✓（126+105=231；74+148=222；30+108=138；104+72=176）                                      |
| 联合检出 = B + (S\B)             | 各数据集 `baseline.failed>0` 与 `total_discrepancies>0` 的并集 | ✓（161=101+60；75=25+50；50=39+11；106=5+101）                                            |
| 联合检出 = (B∩S) + (B\S) + (S\B) | 四象限分解                                                  | ✓（392 = 112+58+222）                                                                  |
| 维度一致性                        | 单内核 `sum(各维度 discrepancies) == total_discrepancies`    | ✓（767/767 全部通过）                                                                      |
| dimension list 一致性           | 各内核 `discrepant_dimensions` 与实际有 disc 的维度集合相等          | ✓（767/767 全部通过）                                                                      |
| Detail 文件覆盖                  | 每个 COMPLETED 内核均有对应 detail JSON                        | ✓（CUDA-L1 231/231；AI-CUDA-Engineer 222/222；TritonBench-G 138/138；CUDA-Agent 176/176） |
| Baseline 测试次数                | 每内核 baseline 计数 = 3                                    | ✓（767×3=2301 = 1798 passed + 503 failed + 0 errors）                                  |


---

## 附录 A：数据文件说明


| 文件路径                                               | 内容                         | 条目数     |
| -------------------------------------------------- | -------------------------- | ------- |
| `外部Benchmark差分测试_RQ4/CUDA-L1/checkpoint.json`          | CUDA-L1 全部内核测试结果           | 241 条   |
| `外部Benchmark差分测试_RQ4/CUDA-L1/details/*.json`           | CUDA-L1 各内核详细测试记录          | 231 个文件 |
| `外部Benchmark差分测试_RQ4/AI-CUDA-Engineer/checkpoint.json` | AI-CUDA-Engineer 全部内核测试结果  | 229 条   |
| `外部Benchmark差分测试_RQ4/AI-CUDA-Engineer/details/*.json`  | AI-CUDA-Engineer 各内核详细测试记录 | 222 个文件 |
| `外部Benchmark差分测试_RQ4/TritonBench-G/checkpoint.json`    | TritonBench-G 全部内核测试结果     | 141 条   |
| `外部Benchmark差分测试_RQ4/TritonBench-G/details/*.json`     | TritonBench-G 各内核详细测试记录    | 139 个文件 |
| `外部Benchmark差分测试_RQ4/CUDA-Agent/checkpoint.json`                  | CUDA-Agent 全部内核测试结果        | 220 条   |
| `外部Benchmark差分测试_RQ4/CUDA-Agent/details/*.json`                   | CUDA-Agent 各内核详细测试记录       | 176 个文件 |


> **TritonBench-G detail 文件多 1 个**: `tritonbench__157_rmsnorm_fused.json` 存在于 details 但不在 checkpoint 中，为补跑时生成的副产物，不计入主统计。

---

## 附录 B：关键源码路径


| 功能          | 文件路径                                 | 关键函数/常量                                                                |
| ----------- | ------------------------------------ | ---------------------------------------------------------------------- |
| 主测试调度       | `scripts/run_fullscale_diff_test.py` | `run_kernel_5dim()` (第 197-388 行)、`run_quick_baseline()` (第 146-176 行) |
| 子进程 worker  | `scripts/_stress_worker.py`          | `run_stress()` (第 115 行)、`_allclose()` (第 45 行)                        |
| 21 种压力策略    | `src/stress/policy_bank.py`          | `STRESS_POLICIES` 字典（第 203-227 行）                                      |
| 容差超参        | `scripts/run_fullscale_diff_test.py` | `DEFAULT_ATOL=1e-2, DEFAULT_RTOL=1e-2` (第 43-44 行)                     |
| Batch sizes | `scripts/run_fullscale_diff_test.py` | `BATCH_SIZES=[1,4,16,64]` (第 45 行)                                     |


