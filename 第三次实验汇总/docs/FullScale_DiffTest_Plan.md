# 全量差分测试计划

> 基于 MutaKernel 5 维度增强测试方法，对多个外部 CUDA kernel 数据集进行全量差分测试

---

## 一、数据集调研结果

### 1.1 数据集总览

| 数据集 | 来源 | kernel 数量 | 格式 | PyTorch 参考 | 可测性评估 |
|--------|------|-----------|------|-------------|----------|
| **NVIDIA/apex** | 官方仓库 | ~10 个可测模块 | Python API 包装 CUDA C++ | 需手写对应 nn.Module | ✅ 直接可测 |
| **TritonBench-G** | thunlp/TritonBench | **184** 个 GitHub Triton 算子 | Python + Triton kernel | 部分有 PyTorch ref | ⚠️ 需适配（Triton 而非 CUDA C++） |
| **AI CUDA Engineer Archive** | SakanaAI (HuggingFace) | **~30,000** 个 CUDA kernel（250 task × ~120 变体） | Parquet: CUDA_Code + PyTorch_Code_Module | ✅ 已包含 PyTorch ref | ✅ **最佳候选** |
| **CUDA-L1** | deepreinforce-ai | **250** 个优化 CUDA kernel（基于 KernelBench） | JSON: ref_code + custom_code | ✅ KernelBench 的 PyTorch ref | ✅ 直接可测 |
| **KernelBench** (原始) | ScalingIntelligence | **250** 个 PyTorch → CUDA 任务 | Python: Model + ModelNew + get_inputs | ✅ 原生 KernelBench 格式 | ✅ 已兼容（第一/二次实验来源） |

### 1.2 各数据集详细分析

#### TritonBench (thunlp/TritonBench)

- **论文**: Li et al., ACL Findings 2025
- **规模**: TritonBench-G = 184 个 GitHub 实际 Triton 算子；TritonBench-T = PyTorch 对齐算子集
- **格式**: 每个任务有 Triton kernel 代码 + 自然语言描述 + 测试代码
- **关键特性**:
  - 真实 GitHub 仓库代码（>100 stars 仓库筛选）
  - 覆盖 Attention(20%), MatMul(10.9%), LayerNorm(6.5%), SoftMax(3.8%) 等
  - 有 execution accuracy 评估机制
- **可测性评估**: ⚠️ **中等难度**
  - Triton kernel 不是 CUDA C++，而是 Python DSL → 不需要 `load_inline` 编译
  - 每个 kernel 需要有对应的 PyTorch reference
  - 部分有现成 PyTorch ref，部分需要从语义描述中推导
  - 我们的 `_stress_worker.py` 需要适配 Triton kernel 加载方式

#### AI CUDA Engineer Archive (SakanaAI)

- **论文**: Lange et al., 2025 (Sakana AI)
- **规模**: ~30,615 个 CUDA kernel（level_1: 12,157; level_2: 12,938; level_3: 5,520）
- **基础任务数**: 250 个 KernelBench 任务 × 多个变体
- **格式** (Parquet 字段):
  - `CUDA_Code`: 生成的 CUDA kernel 代码（已含 `load_inline`）
  - `PyTorch_Code_Module`: PyTorch 参考实现（KernelBench 的 `class Model`）
  - `PyTorch_Code_Functional`: 函数式参考实现
  - `Correct`: bool，原始正确性判定
  - `Max_Diff`: float，与参考实现的最大差异
  - `CUDA_Speedup_Native`: 相对 PyTorch 的加速比
- **可测性评估**: ✅ **最佳候选**
  - 天然 KernelBench 格式，与我们的 `_stress_worker.py` 完全兼容
  - 已有 `Correct` 和 `Max_Diff` 字段作为基线对比
  - 可以筛选 `Correct=True` 的 kernel 作为测试集（验证我们的 stress 策略是否能暴露其通过简单正确性测试但在极端输入下失败的情况）
  - 也可以测试 `Correct=False` 的 kernel（验证我们的策略检出率 vs 简单随机测试检出率）

#### CUDA-L1 (deepreinforce-ai)

- **论文**: Li et al., arXiv 2507.14111, 2025
- **规模**: 250 个 KernelBench 任务 × 1 个最优 RL 优化 kernel
- **格式** (JSON 字段):
  - `ref_code`: KernelBench 原始参考 CUDA 代码
  - `custom_code`: CUDA-L1 RL 优化后的 CUDA 代码
  - `level_id`, `task_id`: 对应 KernelBench 的层级和任务编号
  - `score_default`: 加速比
- **可测性评估**: ✅ **直接可测**
  - 与 KernelBench 格式一致，每个 kernel 都有对应的 PyTorch ref
  - 250 个 kernel 规模适中，可全量测试
  - 特别有意义：这些是 RL 优化后的 kernel，验证"追求加速是否牺牲了数值正确性"

#### NVIDIA/apex (全量)

- **可测模块枚举**:

| # | 模块 | API | PyTorch 参考 | 参数类型 | 输入约束 |
|---|------|-----|-------------|---------|---------|
| 1 | `apex.normalization.FusedLayerNorm` | ✅ 已测 | `nn.LayerNorm` | 确定性 | 3D |
| 2 | `apex.normalization.FusedRMSNorm` | ✅ 已测 | 手写 RMSNorm | 确定性 | 3D |
| 3 | `apex.fused_dense.FusedDense` | ✅ 已测 | `nn.Linear` | 随机（需同步） | 2D |
| 4 | `apex.fused_dense.FusedDenseGeluDense` | ✅ 已测 | Linear+GELU+Linear | 随机（需映射） | 2D |
| 5 | `apex.contrib.group_norm` | 新增 | `nn.GroupNorm` | 确定性 | 4D (NCHW) |
| 6 | `apex.contrib.layer_norm` (FastLayerNorm) | 新增 | `nn.LayerNorm` | 确定性 | 3D |
| 7 | `apex.contrib.xentropy` (LabelSmoothing) | 新增 | `F.cross_entropy` | 无参数 | (N,C) logits |
| 8 | `apex.contrib.conv_bias_relu` | 新增 | Conv2d+Bias+ReLU | 随机（需同步） | 4D (NCHW) |
| 9 | `apex.contrib.fmha` (FlashMHA) | 新增 | `F.multi_head_attention_forward` | 随机 | 需特殊输入 |
| 10 | `apex.optimizers.FusedAdam` | 新增 | `torch.optim.Adam` | 状态 kernel | 梯度输入 |

---

## 二、全量测试方案

### 2.1 测试策略

采用**分层递进**策略，按优先级排列：

| 优先级 | 数据集 | 计划测试数量 | 理由 |
|--------|--------|-----------|------|
| P0 | **NVIDIA/apex** 全量 | ~10 个 | 完善当前实验，补充 5 个新 kernel |
| P1 | **CUDA-L1** 全量 | **250 个** | KernelBench 格式，直接兼容，RL 优化 kernel 最有研究价值 |
| P2 | **AI CUDA Engineer** 筛选 | **250 个**（每 task 选最优 1 个） | 30k 中选 `Correct=True` 且 `Speedup>1.5x` 的最优变体 |
| P3 | **TritonBench-G** 可测子集 | **~50-80 个**（有 PyTorch ref 的） | 需要适配 Triton 加载，选有明确 ref 的子集 |

### 2.2 五维度增强测试配置

> 与当前实验一致，不含 LLM 维度

| 维度 | 描述 | 轮次/kernel |
|------|------|-----------|
| Baseline | 50 seeds × identity policy | 50 |
| value_stress | 21 策略 × 3 seeds | 63 |
| dtype_stress | float16 + bfloat16 × 3 seeds | 6 |
| training_stress | .train() 模式 × 21 策略 × 3 seeds | 63 |
| repeated_run | 3 seeds × 10 trials | 3 |
| config_stress | 7 batch_sizes × 3 seeds（allclose） | 7（1 call） |
| **总计** | | **192 轮/kernel** |

### 2.3 预估时间

基于当前实验：每个 kernel 约 120-140 秒（6 维度），按 RTX 4070 Laptop 单 GPU：

| 数据集 | kernel 数 | 预估时间 |
|--------|----------|---------|
| Apex 全量（新增 5 个） | 5 | ~12 分钟 |
| CUDA-L1 全量 | 250 | ~9 小时 |
| AI CUDA Engineer 筛选 | 250 | ~9 小时 |
| TritonBench 子集 | ~60 | ~2.3 小时 |
| **总计** | **~565 个 kernel** | **~20.5 小时** |

---

## 三、各数据集接入方案

### 3.1 NVIDIA/apex 全量（P0）

**当前状态**: 已测 4 个，需新增 ~5 个

**新增 kernel 清单**:

| kernel | 包装方式 | 需要的适配 |
|--------|---------|----------|
| `apex.contrib.group_norm` | `from apex.contrib.group_norm import GroupNorm` | 需确认编译标志 |
| `apex.contrib.layer_norm` (Fast) | `from apex.contrib.layer_norm import FastLayerNorm` | 需确认可用性 |
| `apex.contrib.xentropy` | `from apex.contrib.xentropy import SoftmaxCrossEntropyLoss` | 特殊输入 (logits, labels) |
| `apex.contrib.conv_bias_relu` | `from apex.contrib.conv_bias_relu import ConvBiasReLU` | 需确认模块存在 |
| `apex.optimizers.FusedAdam` | `from apex.optimizers import FusedAdam` | 需特殊测试流程（optimizer step） |

**实现方式**: 在现有 `external_benchmarks/registry.py` 中追加条目

### 3.2 CUDA-L1 全量（P1）

**数据获取**:
```bash
# 从 HuggingFace 下载
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('deepreinforce-ai/CUDA-L1'); ds.save_to_disk('cuda_l1_data')"
# 或直接下载 JSON
wget https://github.com/deepreinforce-ai/CUDA-L1/raw/main/optimized_cuda_code/a100.json
```

**格式适配**:
- `ref_code` → 作为 `kernel_code`（KernelBench 的 ModelNew 实现）
- `custom_code` → 作为 `mutated_code`（CUDA-L1 优化后的实现）
- PyTorch 参考从 KernelBench 的 `level{X}/problem_{Y}.py` 获取
- **关键**: 这里不是 "外部 kernel vs PyTorch"，而是 **"CUDA-L1 优化 kernel vs KernelBench 原始 kernel"**
  - 如果两者都与 PyTorch ref 一致 → 优化未引入 Bug
  - 如果优化 kernel 偏离但原始 kernel 不偏离 → **RL 优化引入了数值 Bug**

**也可以做**: "CUDA-L1 优化 kernel vs PyTorch ref" 的差分测试（与当前 FlashAttention 实验一致）

### 3.3 AI CUDA Engineer Archive（P2）

**数据获取**:
```python
from datasets import load_dataset
dataset = load_dataset("SakanaAI/AI-CUDA-Engineer-Archive")
df_l1 = dataset["level_1"].to_pandas()  # 12,157 rows
df_l2 = dataset["level_2"].to_pandas()  # 12,938 rows
df_l3 = dataset["level_3"].to_pandas()  # 5,520 rows
```

**筛选策略**: 每个 task（250 个）选取 `Correct=True` 且 `CUDA_Speedup_Native` 最高的 1 个变体

**格式适配**:
- `CUDA_Code` → 作为 `kernel_code`（含 `load_inline`，即 KernelBench 的 ModelNew）
- `PyTorch_Code_Module` → 作为 `problem_file`（含 `class Model` + `get_inputs`）
- **天然兼容** `_stress_worker.py` 的协议

**研究价值**:
- 已标记 `Correct=True` 的 kernel 在简单测试下通过了，我们验证它们在 **stress 输入下是否仍然正确**
- `Max_Diff` 字段作为基线对比：我们的多容差方法 vs 原始 Max_Diff 判定

### 3.4 TritonBench（P3）

**数据获取**:
```bash
git clone https://github.com/thunlp/TritonBench.git
# TritonBench-G: TritonBench_G_v1/ 目录含 184 个可执行 Triton kernel
# TritonBench-T: TritonBench_T_v1/ 目录含 PyTorch 对齐的 Triton kernel
```

**格式适配**（需要较多工作）:
- Triton kernel 不使用 `load_inline`，而是直接 `import triton` + `@triton.jit`
- 需要修改 `_stress_worker.py` 或编写专用 worker 支持 Triton kernel 加载
- 每个 kernel 需要明确的 PyTorch 参考实现和输入规范
- **建议先从 TritonBench-T（PyTorch 对齐）子集入手**，因为它有明确的 PyTorch 接口

**可测性筛选标准**:
1. 有明确的 PyTorch 参考实现
2. 输入为标准 tensor（非自定义数据结构）
3. 输出可直接用 `allclose` 比较

---

## 四、实现步骤

### Phase 1: Apex 全量补充（预计 1 小时）

1. 在 WSL 中检测哪些 apex.contrib 模块实际可用
2. 为可用模块编写 reference + registry 条目
3. 运行 5 维度测试
4. 更新结果

### Phase 2: CUDA-L1 全量（预计 10 小时）

1. 下载 CUDA-L1 的 `optimized_cuda_code/a100.json`（或 HuggingFace 数据）
2. 下载 KernelBench 数据集获取 PyTorch ref 文件
3. 编写 `cuda_l1_registry.py`：自动从 JSON 生成 250 个测试条目
4. 编写 `run_cuda_l1_diff_test.py`：批量运行 5 维度测试
5. 监控运行，修复任何编译/运行时错误
6. 生成结果报告

### Phase 3: AI CUDA Engineer Archive（预计 10 小时）

1. 从 HuggingFace 下载数据集
2. 筛选 250 个最优变体
3. 编写 `sakana_registry.py`：自动从 Parquet 生成测试条目
4. 复用 Phase 2 的运行框架
5. 对比分析：我们的 stress 检出率 vs 原始 `Correct` 字段

### Phase 4: TritonBench 子集（预计 3 小时）

1. Clone TritonBench 仓库
2. 筛选有 PyTorch ref 的 kernel
3. 编写 Triton kernel 加载适配器
4. 运行 5 维度测试
5. 分析 Triton kernel 的数值特性

---

## 五、预期研究贡献

### 5.1 论文中可支撑的 Claims

| Claim | 数据支撑 |
|-------|---------|
| MutaKernel 的 stress 策略可迁移到外部 kernel 差分测试 | 5 个 Apex + 1 FlashAttention 已验证 |
| stress 策略可检测出简单随机测试无法发现的数值 Bug | AI CUDA Engineer 的 `Correct=True` kernel 中发现新偏差 |
| RL 优化 CUDA kernel 可能引入数值退化 | CUDA-L1 的 250 个优化 kernel vs 原始 kernel |
| 方法具备规模化能力（>500 kernel） | 全量测试覆盖 |
| 不同来源 kernel 的数值质量分布差异 | 跨数据集对比 |

### 5.2 对比实验设计

| 对比维度 | 实验组 | 对照组 |
|---------|--------|--------|
| stress vs random | 21 策略的检出率 | 纯 random 输入的检出率 |
| 5 维度 vs 单维度 | 全部 5 维度 | 仅 value_stress |
| 优化 kernel vs 原始 | CUDA-L1 优化版 | KernelBench 原始版 |
| 人类写 vs LLM 写 | Apex (人类) | AI CUDA Engineer (LLM) |

---

## 六、风险与缓解

| 风险 | 缓解措施 |
|------|---------|
| CUDA-L1/AI-CUDA kernel 编译失败 | 跳过编译失败的 kernel，统计可测率 |
| 运行时间过长（>20h） | 先跑 Level 1（100 个），再扩展 |
| GPU 显存不足（某些大 kernel） | 设置 per-kernel 超时，OOM 时跳过 |
| TritonBench Triton 版本不兼容 | 优先级最低，可选跳过 |
| AI CUDA Engineer 数据集过大（30k） | 只取每 task 最优 1 个变体（250 个） |

---

## 七、参考文献

| # | 引用 | 与本实验的关系 |
|---|------|-------------|
| 1 | Li et al. "TritonBench" ACL Findings 2025 | TritonBench 数据集来源 |
| 2 | Lange et al. "The AI CUDA Engineer" arXiv 2025 | AI CUDA Engineer Archive 来源 |
| 3 | Li et al. "CUDA-L1" arXiv 2507.14111, 2025 | CUDA-L1 数据集来源 |
| 4 | Ouyang et al. "KernelBench" ICML 2025 | 基础任务定义，250 个 PyTorch→CUDA 任务 |
| 5 | Sakana AI "robust-kbench" NeurIPS 2025 | 健壮性 benchmark 设计参考 |
| 6 | Lin et al. "SOL-ExecBench" NVIDIA 2026 | 新一代 kernel benchmark（参考） |
