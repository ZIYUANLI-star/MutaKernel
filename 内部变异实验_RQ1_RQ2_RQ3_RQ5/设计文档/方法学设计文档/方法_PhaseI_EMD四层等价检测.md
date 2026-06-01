# MutaKernel EMD: 面向 GPU CUDA Kernel 的多层等价变异体检测方法

> **定位**: 本文档是 MutaKernel 框架 **Phase 1 流水线中等价变异体检测（Equivalent Mutant Detection, EMD）模块**的完整方法说明。EMD 在 Phase 1 内位于 Block 2（变异 kill/survive 初判）之后、Phase 2（增强测试）之前，仅对 Block 2 中标记为 SURVIVED 的变异体执行。文档内容严格对齐代码实现，包含充分的设计原理、算法伪代码和理论定位，可直接用于撰写 CCF-A 期刊论文的方法（Method）章节。

---

## 0. EMD 在整体流水线中的位置

```
Block 1: MutantRunner.generate_mutants()          ← 主进程
   │   纯 Python AST 变异生成（16 类算子）
   │   每类算子采样 SAMPLE_PER_OP=3 个变异体
   ▼
Block 2: 子进程 _mutant_worker.py (mode=run)      ← 每变异体独立子进程
   │   JIT 编译变异体 → 用 reference(PyTorch) 输出
   │   与 mutant(CUDA) 输出做 torch.allclose(atol=1e-2, rtol=1e-2) 比较
   ▼
   ├─ KILLED      → 终止（测试检测到行为差异）
   ├─ STILLBORN   → 终止（编译失败 / 运行崩溃）
   └─ SURVIVED ──┐
                 ▼
EMD 四层流水线（本文档的范围）
   Layer 0 (主进程) → Layer 1 (主进程) → Layer 2 (子进程) → Layer 3 (主进程)
                 │
                 ▼
   {STRICT_EQUIVALENT, CANDIDATE_EQUIVALENT, SURVIVED}
                 │
                 ▼
Phase 2: 增强测试（按 Tier 1/2/3 分级）
```

EMD **仅对 SURVIVED 变异体执行**。KILLED、STILLBORN 变异体不进入 EMD。Phase 1 的完整入口为 `scripts/full_block12.py: _process_one_kernel()`，Block 1 变异生成、Block 2 kill/survive 初判和 EMD 四层流水线全部在该函数内串联执行。

---

## 1. 问题定义

### 1.1 等价变异体问题

等价变异体问题（Equivalent Mutant Problem, EMP）是变异测试中最根本的挑战之一。Budd 和 Angluin (1982) 证明了**程序等价性在一般情况下是不可判定的**（undecidable），不存在通用算法能判断任意两个程序是否语义等价。

在 MutaKernel 的实验流水线中，Block 2 对每个 CUDA GPU kernel 变异体用 `torch.allclose(atol=1e-2, rtol=1e-2)` 判定 kill/survive。存活的变异体（survived mutants）中包含两种情况：

1. **真正可杀但当前测试套件未能杀死的变异体**：后续 Phase 2 增强测试的目标。
2. **语义等价的变异体**：变异未改变程序在任何输入下的输出行为，不可能被任何测试杀死。

由于程序等价性在一般情况下不可判定，EMD 模块**不试图对每个 survived 变异体给出"等价/不等价"的二元确定性判定**，而是将其分级为两种置信度的等价标签：

- **STRICT_EQUIVALENT**（强证据等价）：由文本归一化或静态规则可证明——这类变异体从存活集合中移除，因为**任何 validator 都不应被期待杀死它们**。
- **CANDIDATE_EQUIVALENT**（统计性启发式等价）：112 轮动态 bitwise 检测未观测到差异——这类变异体**保留为不确定样本**，并在 Phase 2 中接受进一步挑战。

这一分级设计服务于 mutation score 的校准：STRICT_EQUIVALENT 从分母中剔除，防止度量夸大 validator 的薄弱程度；CANDIDATE_EQUIVALENT 按保守口径保留在分母中，防止隐藏 validator 未能揭示的真实缺陷。两种口径（Conservative / Optimistic）分别报告，让读者自行判断结果的上下界（详见 §5.2）。

### 1.2 GPU/CUDA 变异测试中的特殊困难（与intro部分提出的问题应该表述一致）

现有等价变异体检测研究几乎全部针对顺序程序（Java、C）。GPU CUDA kernel 引入了多重额外复杂性：

| 复杂性维度 | 说明 |
|---|---|
| 并行语义等价 | 一个 kernel 可能发射百万线程，grid/block 配置的变异可能改变线程总数但不影响最终输出（多余线程因 bounds check 不执行写操作）——这种"grid size 膨胀等价"在顺序程序中不存在 |
| host-device 分离 | 变异可能发生在 Python 宿主代码或 CUDA device 代码中，两者的等价判定逻辑完全不同 |
| 嵌入式 CUDA 字符串 | CUDA 代码以字符串形式嵌入 Python 文件（用于 `torch.utils.cpp_extension.load_inline()` JIT 编译），传统 AST 分析工具无法解析 |
| JIT 编译开销 | `load_inline` 编译一个 CUDA kernel 可能需要 30-120 秒，使动态检测的时间成本远高于 Java/C |
| GPU 专有操作 | `__syncthreads()`、warp shuffle、shared memory bank conflict 等 GPU 特有并行原语，其等价性判定需要 GPU 领域知识 |
| fixed-shape 契约 | 在 KernelBench 等基准测试中，输入 shape 固定不变，某些变异仅在特定 shape 下等价——这种**配置依赖的等价性**是传统方法未考虑的 |

### 1.3 现有方法的局限性

| 方法 | 核心思路 | 局限性 |
|---|---|---|
| **TCE** (Papadakis et al., ICSE 2015; TSE 2017) | 利用编译器优化将程序编译为 object code 后比较 | 依赖 gcc/javac 的规范化能力，nvcc 的优化策略与之差异显著；CUDA kernel 嵌入在 Python 宿主代码中，TCE 的"整个程序编译后比较"范式无法直接适用 |
| **EMS** (Kushigian et al., ISSTA 2024) | 基于 10 种 Java AST 静态分析模式检测等价变异体 | 规则完全针对 Java AST 设计，不适用于 CUDA C++ 的并行编程范式（threadIdx、blockDim 等） |
| **LLM 方法** (Tian et al., ISSTA 2024) | 使用 LLM 进行等价检测，F1 提升 35.69% | 仅针对方法级 Java 代码，未涉及 GPU kernel 的并行语义 |
| **约束求解** (Holling et al., ICST 2016, Nequivack) | 符号执行 + 约束求解检测非等价性 | GPU 的大规模并行执行模型（数万线程同时执行）使路径爆炸问题更加严重，支持几乎为零 |

### 1.4 EMD 模块的贡献

现有等价变异体检测工作（TCE、EMS、LLM 方法等）均针对顺序程序（Java、C），**GPU CUDA kernel 源码级变异体的等价检测尚未被系统性地研究**。MutateNN（Chatzikonstantinou et al., FSE 2024）关注 DNN 模型层级变异（权重/激活函数等），不涉及 CUDA kernel 源码的并行语义。MutaKernel EMD 直接作用于嵌入式 CUDA kernel 源码，设计了面向 GPU 并行编程模式的多层证据链检测流水线。它解决了以下关键问题：

1. **CUDA-aware 源码归一化**：从嵌入 Python 的 CUDA 字符串中提取 kernel 源码，进行 C++ 级归一化，同时分析 Python 宿主代码差异——这是对 TCE 在 host-device 混合编程范式下的适配。
2. **GPU 算子静态规则**：受 EMS 的 AST pattern matching 启发，设计了 4 条面向 CUDA 并行编程模式的静态规则，能识别 GPU 特有的等价模式。
3. **算子定向动态 bitwise 检测**：受 state infection condition (Offutt & Lee, 1996) 启发，为 CUDA 变异算子设计了定向输入生成策略，最大化触发变异效果。
4. **LLM 等价验证（证据链审查）**：与 Tian et al. (ISSTA 2024) 的 LLM 独立判断不同，我们的 LLM 在自动化层的完整证据基础上做二次审查。

---

## 2. 设计原则

| # | 原则 | 含义 |
|---|---|---|
| P1 | **精度优先于召回** | 误标等价（false equivalent）会终止后续增强测试，代价远高于漏检等价（false non-equivalent），后者仅浪费增强测试资源 |
| P2 | **分层递进** | 计算代价从低到高：字符串比较（毫秒级）→ 静态规则匹配（毫秒级）→ 动态 GPU 运行（分钟级）→ LLM API 调用（秒级 + token 费用），仅对前一层未能判定的变异体执行后一层 |
| P3 | **证据分级** | 不同强度的等价证据对应不同标签：STRICT_EQUIVALENT（有可证明的强证据）vs CANDIDATE_EQUIVALENT（统计性启发式证据），不混为一谈 |
| P4 | **单向安全** | LLM（Layer 3）只能推翻等价判定（等价→SURVIVED），不能反向确认（SURVIVED→等价），保证 false-equivalent 方向的安全性 |
| P5 | **固定 shape、变值** | 输入 tensor 的 shape/dtype 固定来自 reference 模块的 `get_inputs()`，只有数值可变 |
| P6 | **不短路** | 当 CUDA kernel 代码相同但 Python 宿主代码不同时，不直接判定等价，必须继续经过 Layer 1 静态规则和 Layer 2 动态验证 |

---

## 3. EMD 输入/输出契约

### 3.1 输入

| 项 | 类型 | 说明 |
|---|---|---|
| **survived_mutants** | `List[Mutant]` | Block 2 标记为 `SURVIVED` 的变异体列表。每个变异体包含 `original_code`（best original 完整源码）、`mutated_code`（变异后完整源码）、`operator_name`（变异算子类型）、`site.line_start`（变异行号）、`site.original_code`（变异行原始文本片段） |
| **kernel** | `KernelInfo` | 包含 `kernel_code`（best original CUDA 源码，来自 `best_kernels.json`，已通过 KernelBench harness 验证与 reference allclose 一致）、`reference_module_path`（KernelBench problem 文件路径，提供 `get_inputs()` 和 `get_init_inputs()`） |
| **llm_caller** | `Optional[Callable[[str], str]]` | LLM 调用器，接收 prompt 返回 response 字符串。可为 None（跳过 Layer 3） |

### 3.2 输出

| 项 | 说明 |
|---|---|
| **mutant.status** | 由 `SURVIVED` 可能转移到 `STRICT_EQUIVALENT` / `CANDIDATE_EQUIVALENT`，或保持 `SURVIVED` |
| **mutant.equiv_detail** | dict，包含 `layer0` / `layer1` / `layer2` / `layer3` 四层证据链 + `decided_at`（在哪一层定型）+ `mutant_id`、`input_spec` 等元数据 |

### 3.3 不变量

- LLM 永远不会把 `SURVIVED` 改为 `STRICT_/CANDIDATE_EQUIVALENT`（单向安全，原则 P4）
- `STRICT_EQUIVALENT` 仅由 Layer 0 或 Layer 1 产出（有可证明的强证据）
- `CANDIDATE_EQUIVALENT` 仅由 Layer 2 产出（统计性启发式证据）
- Layer 3 只能将已判定的等价状态**回退**为 `SURVIVED`

---

## 4. 实验参数清单

以下参数控制 EMD 流水线的行为，所有值均以代码中 `scripts/full_block12.py` 和 `scripts/_mutant_worker.py` 的默认设定为准：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `EQUIV_RUNS` | 100 | Layer 2 随机阶段轮数 |
| `EQUIV_TIMEOUT` | 600s (10 min) | Layer 2 子进程硬超时（含 2 次 JIT 编译 + 112 轮 GPU forward） |
| `MUTANT_TIMEOUT` | 180s (3 min) | Block 2 单变异体子进程硬超时 |
| `ATOL` / `RTOL` | 1e-2 | Block 2 allclose kill 判定容差（EMD 不使用，仅作上下文参照） |
| `base_seed` | 10000 | Layer 2 种子起点 |
| `SEED` | 42 | Block 2 全局随机种子 |
| `SAMPLE_PER_OP` | 3 | Block 1 每类算子采样数 |
| `LLM_MODEL` | `deepseek-chat` | Layer 3 LLM 模型（可被环境变量 `LLM_MODEL` 覆盖） |
| `LLM_TEMPERATURE` | 0.0 | Layer 3 LLM 采样温度（确定性输出） |
| `LLM_MAX_TOKENS` | 4096 | Layer 3 LLM 输出 token 上限 |
| LLM revoke threshold | 0.7 | Layer 3 confidence 阈值，超过才推翻等价判定 |

> **`base_seed = 10000` 的设计理由**：Block 2 初判使用 `SEED=42` 及其衍生序列作为 `get_inputs()` 的种子。Layer 2 选择远离此区间的 10000 作为起点，确保 EMD 动态检测看到的 100 轮随机输入（seed 10000~10099）与 Block 2 初判使用的输入序列不重叠，避免"Block 2 没杀掉的输入在 Layer 2 也看不到差异"的冗余覆盖。

---

## 5. 状态模型

### 5.1 MutantStatus 枚举

EMD 模块使用以下 8 个互斥状态标记每个变异体的最终分类：

```python
class MutantStatus(Enum):
    PENDING              = "pending"                # 尚未测试
    KILLED               = "killed"                 # 被 Block 2 allclose 测试杀死
    SURVIVED             = "survived"               # 存活（未被杀死，也未被判定等价）
    STILLBORN            = "stillborn"              # 编译失败 / 子进程崩溃
    STRICT_EQUIVALENT    = "strict_equivalent"      # 强证据等价（文本归一化 / 静态规则可证明）
    CANDIDATE_EQUIVALENT = "candidate_equivalent"   # 动态检测未观测到差异（启发式）
    UNKNOWN              = "unknown"                # 异常 / infra 故障
    TIMEOUT              = "timeout"                # 超时
```

**向后兼容**: `from_dict` 读到旧的 `"equivalent"` 字符串时自动映射为 `CANDIDATE_EQUIVALENT`。

### 5.2 Mutation Score 计算

EMD 使用两种口径报告 mutation score，让读者自行判断结果的上下界：

| 口径 | 公式 | 排除项 |
|---|---|---|
| **Conservative** (保守) | killed / (total − stillborn − strict_eq) | 仅排除有强证据的 STRICT_EQUIVALENT |
| **Optimistic** (乐观) | killed / (total − stillborn − strict_eq − candidate_eq) | 额外排除启发式判定的 CANDIDATE_EQUIVALENT |

**代码实现**（`src/models.py: MutationTestResult`）：

```python
@property
def mutation_score(self) -> float:             # 保守口径
    denom = self.total - self.stillborn - self.strict_equivalent
    return self.killed / denom if denom > 0 else 0.0

@property
def mutation_score_optimistic(self) -> float:  # 乐观口径
    denom = self.total - self.stillborn - self.strict_equivalent - self.candidate_equivalent
    return self.killed / denom if denom > 0 else 0.0
```

按类别（A/B/C/D）和按算子的分 mutation score 使用保守口径（分母排除 STILLBORN + STRICT_EQUIVALENT）。

---

## 6. 四层流水线架构

### 6.1 架构总览

```
     Survived Mutant
           │
     ┌─────▼─────┐
     │  Layer 0   │  CUDA 源码归一化 + 宿主代码差异分析
     │  (主进程)   │  耗时: <1ms / mutant
     └─────┬─────┘
           │
    ┌──────┼──────────────────┐
    │      │                  │
  cuda_eq  │  cuda_eq=T       │  cuda_eq=F
  AND      │  py_eq=F         │  (或 py_eq=T/F)
  py_eq=T  │                  │
    │      │  记录证据         │
    │      │  继续 ↓           │  继续 ↓
    ▼      ▼                  ▼
 STRICT  ┌─────────────┐
 EQUIV   │   Layer 1    │  算子静态等价规则 (4 条)
         │   (主进程)    │  耗时: <10ms / mutant
         └──────┬──────┘
           ┌────┼────┐
           │         │
        rule_hit   no_hit
           │         │
           ▼         ▼
        STRICT   ┌─────────────┐
        EQUIV    │   Layer 2    │  动态 bitwise 检测
                 │  (子进程)    │  100 随机 + 6策略×2 = 112 轮
                 │              │  耗时: 30s-600s / mutant
                 └──────┬──────┘
                   ┌────┼────────────┐
                   │                 │
               112轮全部             找到反例 / 异常
               bitwise一致           │
                   │                 ▼
                   ▼              SURVIVED
            CANDIDATE_EQUIV      (保持存活)
                   │
           ┌───────┼───────┐
           │               │
           ▼               ▼
     ┌─────────────┐
     │   Layer 3    │  LLM 等价验证 (二次审查)
     │  (主进程)    │  对 STRICT 和 CANDIDATE 均审查
     └──────┬──────┘
       ┌────┼────┐
       │         │
    confirmed   possibly_killable
       │         AND confidence>0.7
       │         │
       ▼         ▼
    保持原      回退为
    等价判定    SURVIVED
```

### 6.2 关键设计：比较对象选择

**EMD Layer 2 比较的是 original_kernel vs mutant_kernel（两个 CUDA kernel），而非 reference(PyTorch) vs mutant。**

这一选择的核心原因是：GPU kernel 与 PyTorch 参考实现之间本身存在浮点差异（来自不同的计算路径、不同的归约顺序等），用 reference vs mutant 做 bitwise 比较会产生大量 false-negative（两者本来就不完全一致，bitwise 几乎总是不同）。通过比较"同一份源码的 best 版本 vs 变异版本"，二者共享相同的 JIT 编译路径和浮点运算顺序，bitwise 比较才具有判别力。

具体地：两个模型均通过 `torch.utils.cpp_extension.load_inline()` 从源码 JIT 编译为 `.so`，加载后在 GPU 上执行，用相同的输入和 seed 分别调用 `forward()`，然后对输出 tensor 做 NaN-aware bitwise 比较。

> **关于 original kernel 的正确性保证**：original 使用的是 `best_kernels.json` 中的 best generation（已被 KernelBench harness 验证与 reference allclose 一致）。因此"original 自己就是 buggy 的"情况理论上不会发生。

### 6.3 执行位置说明

**生产路径**中，EMD 的四层流水线在 `scripts/full_block12.py: _process_one_kernel()` 函数中实现：
- Layer 0 和 Layer 1 在**主进程**中执行（纯 Python AST 分析和字符串比较，无 CUDA 操作）
- Layer 2 在**隔离子进程**中执行（`scripts/_mutant_worker.py: _equiv_mode()`，使用 `subprocess.Popen` + 硬超时保护）
- Layer 3 在**主进程**中调用 LLM API（`src/stress/llm_analyzer.py: verify_equivalent_with_llm()`）

这一进程隔离设计确保 CUDA 编译挂起、非法内存访问、GPU driver 崩溃等只影响 Layer 2 的子进程，主进程标记为超时后继续处理下一个变异体。

**每个变异体的 Layer 2 子进程是完全独立的进程**：每次都通过 `subprocess.Popen` 重新启动 `_mutant_worker.py`，重新 import torch，重新 JIT 编译 original 和 mutant 两个 CUDA kernel（无跨变异体的编译缓存复用）。这是导致 Layer 2 耗时下限较高（2 × CUDA 编译时间 ≈ 60-240s）的根源，也是 §14 描述的超时偏差的直接原因。

---

## 7. Layer 0: CUDA-aware 源码归一化 + 宿主代码差异分析

### 7.1 目标

判断变异是否改变了程序的文本表示。如果归一化后完全相同，则变异在语法层面无效（如变异仅影响注释、空白、格式），可判定为 STRICT_EQUIVALENT。同时，Layer 0 对每个变异体标注**变异域**（mutation_domain），将变异归类到 `python_host` / `cuda_kernel` / `both`，为后续层提供上下文。

### 7.2 双层归一化

由于 CUDA kernel 以字符串形式嵌入在 Python 宿主代码中，Layer 0 采用双层归一化：

#### 7.2.1 CUDA 字符串提取与 C++ 级归一化

1. **CUDA 字符串提取**（`_extract_cuda_strings`）：使用 `CudaParser` 对 Python 源码进行 AST 解析，定位所有赋值给变量（如 `cuda_source = """..."""`）的字符串块，检测是否包含 CUDA 标识符（`__global__`、`__device__`、`load_inline` 等）。如果是 CUDA 文件，返回所有 CUDA 字符串块的拼接内容。

2. **C++ 级归一化**（`_normalize_cuda_source`）：
   - 删除 C++ 块注释 `/* ... */`（含跨行）
   - 删除 C++ 行注释 `// ...`
   - 删除空行
   - 将连续空白折叠为单个空格
   - 行首尾 strip

#### 7.2.2 Python 宿主代码归一化

（`_normalize_python_source`）：
- 删除空行
- 删除以 `#` 开头的注释行
- 行首尾 strip

> **重要说明**: 这两层归一化都是**轻量字符串级处理**（去注释、空白折叠），不做语义归一化（如变量重命名、常量折叠、AST 重排）。这是有意为之的——在 GPU kernel 变异测试场景中，变异算子产生的修改都是语义级的（改操作符、改常量值等），纯格式级的变异极少，因此轻量归一化已足以捕获"变异只影响注释/空白"的情况，同时避免引入语义归一化可能产生的 false positive。

### 7.3 变异域分类（mutation_domain）

Layer 0 根据 CUDA 和 Python 两层归一化比较的结果，为每个变异体标注变异域：

```
mutation_domain ∈ {"python_host", "cuda_kernel", "both"}

判定规则:
  - cuda_eq=True               → "python_host"  (CUDA 字符串相同，变异在 Python 宿主)
  - cuda_eq=False AND py_eq=T  → "cuda_kernel"  (Python 相同，变异在 CUDA 字符串内)
  - cuda_eq=False AND py_eq=F  → "both"         (两边都不同，常见于跨域变异)
```

该字段记录到 `detail["layer0"]["mutation_domain"]` 中，供后续分析和 LLM prompt 使用。

### 7.4 CUDA 差异行记录（cuda_diff_lines）

当 `cuda_eq=False`（CUDA 归一化后不同）时，Layer 0 逐行 diff CUDA 归一化结果，记录**前 10 行差异**到 `detail["layer0"]["cuda_diff_lines"]`，每行格式为：

```json
{"cuda_line": <行号>, "original": "<归一化后原始行>", "mutated": "<归一化后变异行>"}
```

如果两边行数不同，还会追加一条长度差异记录：

```json
{"cuda_line": "length_diff", "original": "<n> lines", "mutated": "<m> lines"}
```

该字段传递给 Layer 3 LLM（通过 `_format_layer_evidence`），帮助 LLM 精确定位变异在 CUDA 源码中的位置。

### 7.5 宿主代码差异分析

当 CUDA 字符串归一化后相同但 Python 宿主代码不同时（`cuda_eq=True, py_eq=False`），调用 `_analyze_host_diff()` 对变异在宿主代码中的位置和影响进行 AST 级分析。该函数返回以下 4 个字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `mutation_location` | str | `"module_level"` / `"inside class <Name>"` / `"inside <Class>.<Method>()"` / `"inside function <Name>()"` |
| `mutated_variable` | str \| None | 被赋值的变量名（仅在变异位于模块级赋值语句时提取） |
| `used_in_model` | bool \| None | 该变量是否被 `ModelNew` 或 `Model` 类的任何方法引用（AST 遍历 `ast.Name` 节点） |
| `used_in_get_inputs` | bool \| None | 该变量是否被 `get_inputs()` 或 `get_init_inputs()` 函数引用 |

**实现逻辑**（`src/mutengine/equivalent_detector.py: _analyze_host_diff()`）：

1. 用 `ast.parse()` 解析变异后的完整源码
2. 遍历 AST，检查变异行号是否在某个 class 或 function 内部，确定 `mutation_location`
3. 如果是模块级赋值（`ast.Assign`），提取左侧变量名
4. AST 遍历 `ModelNew`/`Model` 类体，检查是否引用该变量
5. AST 遍历 `get_inputs`/`get_init_inputs` 函数体，检查是否引用该变量

### 7.6 Layer 0 判定逻辑

```
输入: 变异体 m (包含 original_code 和 mutated_code)

cuda_orig ← ExtractCudaStrings(m.original_code)
cuda_mut  ← ExtractCudaStrings(m.mutated_code)
cuda_eq   ← 两者均非空 AND NormalizeCuda(cuda_orig) == NormalizeCuda(cuda_mut)

py_orig   ← NormalizePython(m.original_code)
py_mut    ← NormalizePython(m.mutated_code)
py_eq     ← py_orig == py_mut
```

| 条件 | Layer 0 verdict | 判定 | 后续 |
|---|---|---|---|
| `cuda_eq=True AND py_eq=True` | `STRICT_EQUIVALENT` | 整个程序归一化后完全相同 | **直接跳到 Layer 3** LLM 审查 |
| `cuda_eq=True AND py_eq=False` | `cuda_identical_host_differs` | CUDA 不变，变异在宿主代码 | **不判定**；记录 `host_diff_analysis` 证据；**继续到 Layer 1 → Layer 2** |
| `cuda_eq=False` (py_eq 任意) | `not_equivalent` | CUDA kernel 代码不同 | **不判定**；记录 `cuda_diff_lines`（前 10 行差异）；继续到 Layer 1 → Layer 2 |

**设计决策**: 当 `cuda_eq=True, py_eq=False` 时**不短路判定为等价**。这修复了早期版本的一个 bug：如 `const_perturb` 修改了模块级常量 `N=2048→2049`，如果 `forward()` 使用了 `N` 来 launch kernel grid，Layer 2 的动态测试可能发现输出差异；而之前直接跳过了动态测试。

### 7.7 Layer 0 伪代码

```
Algorithm 1: Layer0_SourceNormalization(m)
────────────────────────────────────────
Input:  mutant m with m.original_code, m.mutated_code, m.site.line_start
Output: verdict ∈ {STRICT, CONTINUE}, evidence dict

 1  cuda_orig ← CudaParser.parse(m.original_code).all_cuda_source
 2  cuda_mut  ← CudaParser.parse(m.mutated_code).all_cuda_source
 3  cuda_eq   ← (cuda_orig ≠ ∅) ∧ (cuda_mut ≠ ∅) ∧
                 (NormCuda(cuda_orig) = NormCuda(cuda_mut))
 4  py_eq     ← NormPython(m.original_code) = NormPython(m.mutated_code)
 5
 6  mutation_domain ← DetermineDomain(cuda_eq, py_eq)
 7  evidence ← {cuda_eq, py_eq, mutation_domain}
 8
 9  if ¬cuda_eq then
10      evidence.cuda_diff_lines ← DiffLines(NormCuda(cuda_orig), NormCuda(cuda_mut))[:10]
11  end if
12
13  if cuda_eq ∧ py_eq then
14      return (STRICT, evidence)          ▷ 文本完全相同
15  end if
16
17  if cuda_eq ∧ ¬py_eq then
18      hda ← AnalyzeHostDiff(m.mutated_code, m.site.line_start)
19      evidence.host_diff_analysis ← hda  ▷ 记录宿主差异分析
20  end if
21
22  return (CONTINUE, evidence)            ▷ 继续 Layer 1
```

---

## 8. Layer 1: 算子静态等价规则

### 8.1 目标

基于 CUDA 并行编程语义的 pattern matching 规则，在不执行 kernel 的情况下判定 STRICT_EQUIVALENT。受 EMS (Kushigian et al., ISSTA 2024) 的 Java AST 模式匹配启发，我们设计了 4 条面向 GPU 特有编程模式的静态规则。

### 8.2 四条规则

#### 规则 1: `boundary_unreachable`

**适用算子**: `relop_replace`, `mask_boundary`

**原理**: 在 CUDA 中，`threadIdx.x` 的取值范围是 `[0, blockDim.x - 1]`。如果变异将 `threadIdx.x < blockDim.x` 改为 `threadIdx.x <= blockDim.x`，由于 `threadIdx.x` 的值永远到不了 `blockDim.x`，两个条件在所有合法线程上的求值结果完全相同。

**匹配模式**: 使用正则表达式检测 `threadIdx.<dim> <op> blockDim.<dim>` 形式的表达式，判断变异是否仅将 `<` 换成了 `<=`（或 `>` 换成 `>=`），且两侧的维度标识符（x/y/z）相同。

**形式化**:

```
设 d ∈ {x, y, z}, t = threadIdx.d, B = blockDim.d。

由 CUDA 执行模型定义:
    t ∈ [0, B-1]

因此:
    (t < B)  ≡ True   ≡ (t <= B)        ⇒  (t < B) 与 (t <= B) 在合法线程上恒等
    (t > B)  ≡ False  ≡ (t >= B)        ⇒  (t > B) 与 (t >= B) 在合法线程上恒等

故变异 < ↔ <= (或 > ↔ >=) 不改变守卫表达式的真值，构成文本-语义等价。
```

**实现**（`src/mutengine/static_equiv_rules.py: _boundary_unreachable()`）:

1. 用正则 `threadIdx\s*\.\s*([xyz])\s*(<|<=|>|>=)\s*blockDim\s*\.\s*([xyz])` 匹配原始片段（`m.site.original_code`），提取维度 `dim_orig`、操作符 `op_orig`、右侧维度 `dim_rhs_orig`
2. 检查左右维度标识符相同（`dim_orig == dim_rhs_orig`）
3. 在变异后完整代码中搜索相同正则模式
4. 确认维度不变，操作符构成 `{<, <=}` 或 `{>, >=}` 对

#### 规则 2: `dead_write`

**适用算子**: `arith_replace`, `const_perturb`, `scale_modify`, `init_modify`

**原理**: 如果变异修改了一个赋值语句的右侧值，但该变量在下一次被读取之前就被无条件重新赋值，则变异的效果被覆写，不影响后续计算。

**匹配逻辑**:
1. 用正则 `^\s*(\w+)\s*[+\-*/&|^%]?=\s` 匹配赋值语句左侧变量名（排除 CUDA 关键字如 `if`, `for`, `__global__` 等）
2. 通过 `CudaParser` 提取整个 CUDA kernel 源码
3. 从变异行开始向下逐行扫描：如果在该变量被读取（`\b<var>\b`）之前先遇到对它的重新赋值（`^\s*<var>\s*[op]=`），则返回 True；若先遇到读取则返回 False；跳过空行和注释行

#### 规则 3: `mask_noreach`

**适用算子**: `mask_boundary`

**原理**: `mask_boundary` 算子将边界守卫条件收紧（如 `idx < n` → `idx < n-1`），这只影响 padding 区域的线程——那些线程的写操作不在有效输出范围内。

**匹配模式**: 用正则 `(?:idx|tid|index|i|gid)\s*(<|<=)\s*(\w+)` 检测守卫模式，判断变异是否将 `<=` 收紧为 `<`，或将右侧边界减 1（如 `n` → `n-1`）。

#### 规则 4: `dead_host_constant`

**适用算子**: `const_perturb`

**原理**: 在 MutaKernel 的 fixed-shape 测试框架中，`get_inputs()` 和 `get_init_inputs()` **始终来自 reference（参考）模块**，而非变异体模块。因此，如果 `const_perturb` 修改了一个模块级常量（如 `N = 2048 → 2049`），且该常量**不被 `ModelNew`（或 `Model`）类的任何方法引用**，则这个变异是死代码——它永远不会影响 CUDA kernel 的执行路径。

> **关键细节**: 规则的触发条件是"变量不被 ModelNew/Model 引用"，而**不**要求"变量必须被 get_inputs 引用"。即使一个模块级常量谁都不引用（无用变量），变异它同样是死代码。

**实现**（`_dead_host_constant()`），按保守短路优先级：

1. 算子不是 `const_perturb` → 直接返回 False（不适用）
2. `ast.parse()` 解析失败（语法错误）→ 返回 False（保守拒绝）
3. 变异行落在任何 `ast.ClassDef` 体内 → 返回 False（不是模块级）
4. 变异行落在任何 `ast.FunctionDef` / `ast.AsyncFunctionDef` 体内 → 返回 False
5. 在模块级顶层子节点中查找变异行对应的 `ast.Assign`，提取左侧 `ast.Name`；如果不是简单赋值或找不到 → 返回 False
6. AST 遍历 `ModelNew` 或 `Model` 类，发现被赋值变量名出现在任何 `ast.Name` 节点中 → 返回 False
7. 以上全部不命中 → 返回 True（STRICT_EQUIVALENT）

> **注意**：步骤 6 是 *存在性* 检查，不区分 `ast.Store`（赋值左侧）还是 `ast.Load`（读取）。如果 `ModelNew` 内有 `N = some_value`（重新赋值同名变量但与模块级常量无关），规则也会保守拒绝判定为死代码——这是正确的安全行为。

**典型触发场景**:
```python
N = 2048          # ← const_perturb 改为 2049
def get_inputs():
    return [torch.randn(N, N)]    # 只有 get_inputs 用了 N，但 get_inputs 来自 reference
class ModelNew(nn.Module):
    def forward(self, A, B):
        return self.ext.cuda_func(A, B)  # forward 不用 N → 死代码
```

### 8.3 Layer 1 伪代码

```
Algorithm 2: Layer1_StaticRules(m)
────────────────────────────────────────
Input:  mutant m
Output: verdict ∈ {STRICT, CONTINUE}, matched_rule ∈ {str, None}

 1  rules ← [boundary_unreachable, dead_write, mask_noreach, dead_host_constant]
 2  for each rule in rules do
 3      try:
 4          if rule(m) then
 5              return (STRICT, rule.name)
 6          end if
 7      catch Exception:
 8          continue                       ▷ 规则匹配异常时跳过，保守处理
 9  end for
10  return (CONTINUE, None)
```

---

## 9. Layer 2: 动态 Bitwise 检测

### 9.1 目标

对 Layer 0/1 未能判定的 survived 变异体，通过在 GPU 上实际执行 original kernel 和 mutant kernel、比较其输出是否 bitwise 一致，来判定 CANDIDATE_EQUIVALENT。

### 9.2 模型构造与 init_args 语义

Layer 2 子进程启动时，从 reference 模块获取 `get_init_inputs()` 的返回值作为 `init_args`，**original 和 mutant 两个模型均使用同一份 init_args 初始化**：

```python
ref_mod = LoadModule(kernel.reference_module_path)
init_args = ref_mod.get_init_inputs()

orig_model = JIT_Compile(kernel.kernel_code).ModelNew(*init_args).cuda().eval()
mut_model  = JIT_Compile(m.mutated_code).ModelNew(*init_args).cuda().eval()
```

这意味着：
- 构造参数相同，但 `mut_model` 执行的是**变异后的 `__init__` 方法体**，因此 `init_modify` 类算子（如修改 reduction identity 值）**会被 Layer 2 检测到**——虽然两个模型收到相同的构造参数，但变异改变了 `__init__` 内部的行为。
- 如果变异修改了 reference 的 `get_init_inputs()` 函数体本身（使其返回不同的 init args），Layer 2 **不会检测到此差异**——因为 init_args 始终来自 reference 而非 mutant。但在实际实验中，`get_init_inputs()` 位于 reference 模块（problem file）中，不属于变异对象。

### 9.3 NaN-aware Bitwise 比较函数

标准的 `torch.equal()` 在遇到 NaN 时会返回 False（因为 `NaN ≠ NaN`），但在变异测试中，如果 original 和 mutant 在相同位置产生了 NaN，我们认为它们行为一致。因此 EMD 使用自定义的 NaN-aware bitwise 比较：

```
Algorithm 3: BitwiseIdentical(a, b)
────────────────────────────────────────
Input:  两个输出 a, b (可以是 Tensor、tuple、list 或标量)
Output: bool

 1  if a 和 b 都是 Tensor then
 2      if a.shape ≠ b.shape or a.dtype ≠ b.dtype then
 3          return False
 4      if a.is_floating_point() then
 5          nan_a ← isnan(a),  nan_b ← isnan(b)
 6          if nan_a ≠ nan_b then           ▷ NaN 位置不同
 7              return False
 8          finite_mask ← ¬nan_a
 9          if finite_mask.any() then
10              return torch.equal(a[finite_mask], b[finite_mask])
11          return True                      ▷ 全部是 NaN 且位置相同
12      return torch.equal(a, b)             ▷ 整数类型直接比较
13  if a 和 b 都是 tuple/list then
14      return len 相同 ∧ 逐元素递归 BitwiseIdentical
15  return a == b                            ▷ 标量
```

### 9.4 随机阶段

使用 N = 100 轮随机输入（由 `get_inputs()` 配合不同的 `torch.manual_seed(base_seed + i)` 生成，seed 范围 10000~10099），逐轮比较 original 和 mutant 的输出。任意一轮 bitwise 不同即中止并报告反例。

**随机阶段的异常处理**：随机阶段**未**对单边异常做结构化区分。`_run_pair()` 在同一个 `with torch.no_grad()` 上下文中依次调用 `orig_model.forward()` 和 `mut_model.forward()`，如果任一调用抛出异常（CUDA 错误、illegal memory access 等），该异常直接传播到子进程最外层的 `try/except`，以 `EquivCrash` 形式返回 `{is_equivalent=False, error="EquivCrash: ..."}`。设计考虑：随机阶段使用 reference 标准 `get_inputs()` 生成的合法 shape/dtype 输入，正常情况下不应触发运行时异常，因此异常本身即为变异引入 GPU 错误的强证据，无需细分。

### 9.5 定向压力阶段

随机阶段全部通过后，使用算子定向的压力策略生成特殊输入继续检测。每个策略执行 2 个子轮次（sub-trial，使用不同 seed），共 6 策略 × 2 = 12 轮。合计 **100 + 12 = 112 轮**。

策略选择依据变异算子类型，从 `OPERATOR_DIRECTED_POLICIES` 映射中获取。

> **轮次参数选择 rationale**：100 轮随机取自实验经验平衡——在 `EQUIV_TIMEOUT=600s` 约束下，100 轮 + 编译时间已接近上限。定向阶段选 6 策略 × 2 sub-trial，是为了每条策略至少有 2 次机会（1 次失败时另 1 次仍可观测），同时不使总轮数过大。

### 9.6 算子定向策略选择

**生产 worker**（`scripts/_mutant_worker.py`）中，以下 8 类算子有专属定向策略（每类 6 条），其余算子使用 6 条通用策略兜底：

| 算子 | 定向策略 |
|---|---|
| `relop_replace` | relop_boundary_hit, boundary_last_element, structured_ramp, near_zero, sparse, large_magnitude |
| `arith_replace` | extreme_magnitude, large_magnitude, near_zero, all_negative, sparse, boundary_last_element |
| `epsilon_modify` | near_epsilon, near_zero, denormals, large_magnitude, sparse, boundary_last_element |
| `mask_boundary` | boundary_last_element, structured_ramp, head_heavy, tail_heavy, sparse, large_magnitude |
| `index_replace` | head_heavy, tail_heavy, structured_ramp, large_magnitude, sparse, boundary_last_element |
| `sync_remove` | structured_ramp, head_heavy, tail_heavy, large_magnitude, sparse, boundary_last_element |
| `const_perturb` | near_zero, boundary_last_element, sparse, large_magnitude, structured_ramp, all_negative |
| `launch_config_mutate` | structured_ramp, head_heavy, tail_heavy, large_magnitude, sparse, boundary_last_element |

**通用兜底策略**（适用于上述 8 类之外的算子）：
`large_magnitude, near_zero, structured_ramp, all_negative, sparse, boundary_last_element`

> **说明**: `src/mutengine/equivalent_detector.py` 中的 `OPERATOR_DIRECTED_POLICIES` 覆盖了全部 16 类算子（每类 6 条策略），但该模块仅被 smoke 测试脚本调用。第二次全量实验的生产路径使用的是 `scripts/_mutant_worker.py` 中的策略映射，覆盖上述 8 类。未来工作可将全部 16 类策略同步到生产 worker 中。

### 9.7 压力策略库

策略库（`src/stress/policy_bank.py`）共包含 **21 条策略**，所有策略保持输入 tensor 的 shape 和 dtype 不变，仅改变数值分布：

| 类型 | 策略名 | 生成逻辑 | 设计目标 |
|---|---|---|---|
| **通用 (14 条)** | `large_magnitude` | `randn * 1000` | 大数值累积/溢出 |
| | `near_overflow` | `randn * overflow_threshold(dtype)` | 接近类型极限 |
| | `near_zero` | `randn * 1e-7` | 近零值敏感性 |
| | `denormals` | `randn * 1e-38` | 次正规浮点数 |
| | `all_negative` | `-|randn| * 100` | 全负值 |
| | `all_positive` | `|randn| * 100` | 全正值 |
| | `mixed_extremes` | 50% 值 ×10000, 50% 值 ×0.0001 | 极端值混合 |
| | `alternating_sign` | `[+big, -big, +big, ...]` | 交替符号 |
| | `sparse` | 90% 零 + 10% `randn * 100` | 稀疏输入 |
| | `uniform_constant` | 全部元素 = 88.0 | 均匀常数 |
| | `structured_ramp` | `arange(0, N) / N` | 线性递增结构 |
| | `boundary_last_element` | `randn` + 最后元素 = 1e4 | 最后元素边界 |
| | `head_heavy` | 前 25% 极端值，其余近零 | 头部集中 |
| | `tail_heavy` | 后 25% 极端值，其余近零 | 尾部集中 |
| **算子定向 (5 条)** | `relop_boundary_hit` | `arange(N) % 10`（整数值） | 关系运算符边界命中 |
| | `extreme_magnitude` | `randn * 1e6` | 极端数量级溢出路径 |
| | `near_epsilon` | `rand * 9e-6 + 1e-7` | epsilon 敏感分支 |
| | `reduction_adversarial` | 交替 `+1e4 / -1e4 + noise` | 最大化浮点归约误差 |
| | `init_sensitive` | 随机选全正或全负 | 初始值敏感（min/max） |
| **稀疏梯度 (2 条)** | `dense_nonzero` | `|randn| + 1.0` | 消除零值掩码等价 |
| | `sparse_extreme` | 99% 零 + 1% `randn * 1e4` | 极端稀疏 + 边界传播 |

> **注**：`head_heavy` 和 `tail_heavy` 策略的实际代码实现是取前/后 **25%** 元素（`n_head = numel // 4`）为极端值，其余近零。

### 9.8 压力阶段的结构化异常处理

在压力阶段（定向策略轮次），分别对 original 和 mutant 执行并捕获异常，根据异常组合做出不同判定：

| 场景 | original | mutant | 判定 |
|---|---|---|---|
| 策略输入生成阶段抛异常 | (尚未运行模型) | (尚未运行模型) | **跳过**本轮，记录 `status="generation_failed"`，**不**作为反例 |
| 两边都成功，输出相同 | OK | OK, bitwise 一致 | **通过**（本轮等价） |
| 两边都成功，输出不同 | OK | OK, bitwise 不同 | **反例**（判非等价） |
| 一边异常一边成功 | 异常/OK | OK/异常 | 若为 OOM → **跳过**；否则 → **反例** |
| 两边都异常且类型相同 | 异常 A | 异常 A | **通过**（行为一致） |
| 两边都异常但类型不同 | 异常 A | 异常 B | **反例** |

> 策略输入生成失败（`generation_failed`）不计入反例的原因：某些策略可能与特定 tensor 的 dtype/shape 不兼容（如 `init_sensitive` 对零维 tensor 调用 `.item()` 失败），这是策略适用性问题而非变异行为差异。

### 9.9 反例的子分类

Layer 2 发现的反例在 `divergence` 字段中记录了 3 种细粒度分类，用于后续分析和 Phase 2 的 `tier1_replay`：

| `divergence.round_type` | `divergence.detail` | 触发场景 | 语义解释 |
|---|---|---|---|
| `random` | (未设置) | 随机阶段 bitwise 不同 | 数值层差异 |
| `stress` | `output_diverged` | 压力阶段 bitwise 不同 | 数值层差异（被定向输入触发） |
| `stress` | `diff_exception` | 双边都异常但类型不同 | 控制流层差异 |
| `stress` | `one_side_exception` | 一边成功一边异常（非 OOM） | 单边崩溃（GPU 错误强证据） |

每条反例还记录 `seed`、`policy`（压力阶段）、`sub_index`、`input_summary`（每个 tensor 的 shape/dtype/min/max/mean/has_nan/has_inf），确保反例可精确复现。

### 9.10 Layer 2 结果分类

Layer 2 只产出两种输出状态（**不产出 KILLED**）：

| Layer 2 结果 | 条件 | 输出状态 | 后续 |
|---|---|---|---|
| **CANDIDATE_EQUIVALENT** | 112 轮全部 bitwise 一致 | `CANDIDATE_EQUIVALENT` | 进入 Layer 3 LLM 审查 |
| **SURVIVED（不等价）** | 任意一轮发现反例（bitwise 不同 / 单边异常 / 异常类型不同） | 保持 `SURVIVED` | Phase 2 增强测试 |
| **SURVIVED（超时/异常）** | 子进程超时（600s）或 infra 故障 | 保持 `SURVIVED`（保守默认 `is_equivalent=false`） | Phase 2 增强测试 |

**为什么 Layer 2 找到差异 ≠ KILLED**: Layer 2 使用 **bitwise 比较**（精确到每一位），而 Block 2 的 kill 判定使用 `torch.allclose(atol=1e-2, rtol=1e-2)`（容差比较）。因此存在灰色地带：变异体输出与 original 有微小的位级差异（如 1e-6 级浮点误差传播），Layer 2 能检测到差异，但 Block 2/Phase 2 的 allclose 容差会容忍它。这类差异可能只是 GPU 浮点运算的非确定性噪声，不代表语义层面的错误，因此不能直接判定为 KILLED。

### 9.11 Layer 2 完整伪代码

```
Algorithm 4: Layer2_DynamicBitwise(m, kernel)
─────────────────────────────────────────────
Input:  mutant m, kernel info (含 kernel_code, reference_module_path)
        超时限制 T_equiv = 600 秒
Output: (is_equiv: bool, detail: dict)

▷ 以下在隔离子进程中执行 ──────────────────────

 1  ref_mod ← LoadModule(kernel.reference_module_path)
 2  get_inputs ← ref_mod.get_inputs           ▷ 输入生成器来自 reference
 3  init_args  ← ref_mod.get_init_inputs()    ▷ 构造参数来自 reference

 4  orig_model ← JIT_Compile(kernel.kernel_code).ModelNew(*init_args).cuda().eval()
 5  mut_model  ← JIT_Compile(m.mutated_code).ModelNew(*init_args).cuda().eval()
    ▷ 如果任一编译失败 → return (False, "CompileError")
    ▷ 注: 两模型使用相同 init_args，但 mut_model 执行变异后的 __init__ 体

    ▷ ── 随机阶段 ──
 6  for i ← 0 to N_random - 1 do             ▷ N_random = 100
 7      seed ← base_seed + i                  ▷ base_seed = 10000
 8      torch.manual_seed(seed)
 9      inputs ← get_inputs() → move to CUDA
10      with torch.no_grad():
11          orig_out ← orig_model(*inputs)
12          mut_out  ← mut_model(*inputs)
13      if ¬ BitwiseIdentical(orig_out, mut_out) then
14          return (False, {divergence: {round_type: "random", round_index: i, seed}})
        ▷ 任何未捕获异常 → 传播到最外层 → return (False, "EquivCrash: ...")
15  end for

    ▷ ── 定向压力阶段 ──
16  policies ← GetDirectedPolicies(m.operator_name)  ▷ 6 条
17  for each policy_name in policies do
18      policy_fn ← STRESS_POLICIES[policy_name]
19      for si ← 0 to 1 do                    ▷ 每策略 2 个 sub-trial
20          seed ← base_seed + N_random + si
21          torch.manual_seed(seed)
22          try:
23              template ← get_inputs()
24              stress_inputs ← policy_fn(template, seed)
25          catch Exception:
26              记录 status="generation_failed"; continue  ▷ 生成失败跳过
27          inputs ← stress_inputs → move to CUDA

28          orig_exc, mut_exc ← None
29          try: orig_out ← orig_model(*inputs)  except: orig_exc ← e
30          try: mut_out  ← mut_model(*inputs)   except: mut_exc ← e

31          if orig_exc ≠ None ∧ mut_exc ≠ None then
32              if type(orig_exc) = type(mut_exc) then continue  ▷ 同类异常=一致
33              else return (False, {divergence: {detail: "diff_exception"}})
34          if orig_exc ≠ None ∨ mut_exc ≠ None then
35              if isOOM(orig_exc ∨ mut_exc) then continue       ▷ OOM 跳过
36              else return (False, {divergence: {detail: "one_side_exception"}})
37          if ¬ BitwiseIdentical(orig_out, mut_out) then
38              return (False, {divergence: {detail: "output_diverged", policy}})
39      end for
40  end for

41  return (True, {total_rounds: N_random + |policies| × 2})
```

---

## 10. Layer 3: LLM 等价验证

### 10.1 设计动机

Layer 0-2 完成后，所有被标记为 `STRICT_EQUIVALENT` 或 `CANDIDATE_EQUIVALENT` 的变异体都经过一次 LLM 二次审查。目标是**捕获 false-equivalent**（被误判为等价的非等价变异体），因为这类误判会终止后续增强测试，代价最高。

### 10.2 传递给 LLM 的信息

| 信息段 | 来源 | 说明 |
|---|---|---|
| 完整原始源码 + 完整变异源码 | 变异体的 `original_code` 和 `mutated_code` | 完整 Python + CUDA 代码（超过 6000 字符时截断） |
| 变异位置上下文（±12 行，标记变异行） | `_extract_context()` | 用 `>>>` 标记变异行 |
| 变异算子名称和语义描述 | `OPERATOR_DESCRIPTIONS` 字典 | 16 类算子的简要说明 |
| **Layer 0 证据**: CUDA/Python 归一化比较结果、变异域 | `equiv_detail["layer0"]` | 含 `cuda_strings_equal`, `python_host_equal`, `verdict`, `mutation_domain` |
| **Layer 0 宿主差异分析** | `host_diff_analysis` | 含 `mutation_location`, `mutated_variable`, `used_in_model`, `used_in_get_inputs` |
| **Layer 0 CUDA 差异行** | `cuda_diff_lines` | 前 10 行 CUDA 归一化差异（仅 `cuda_eq=False` 时存在） |
| **Layer 1 证据**: 命中了哪条规则 / 未命中 | `equiv_detail["layer1"]` | 以 `rule_hit` 为准；其他字段如 `rules_checked` / `rule_description` / `rule_details` 等仅供调试 |
| **Layer 2 证据**: 测试轮数、bitwise 结果 | `equiv_detail["layer2"]` | 含 `is_equivalent`, `total_rounds`, `cuda_was_identical`, `tested_policies` |
| **实际 input 规格** | `_extract_input_spec()` 从 reference 模块获取 | `forward()` 参数数量、每个 tensor 的 shape/dtype |
| **Testing Principle 声明** | Prompt 正文 | 明确告知 LLM "shape 固定、只能变值、bitwise 比较" |

**`_extract_input_spec()` 的实现方式**：在主进程中通过 `importlib` 加载 reference 模块并调用一次 `get_inputs()`，直接读取每个 tensor 的 `shape` 和 `dtype` 字段后字符串化注入 prompt。如果 reference 的 `get_inputs()` 在导入或调用阶段抛异常（如引用了不可用的 CUDA 设备），则注入一个占位说明字符串 `"(could not extract input spec: <err>)"` 而不阻塞 LLM 调用。

### 10.3 Prompt 核心设计

LLM prompt（`EQUIV_VERIFY_PROMPT`）包含以下关键段落：

1. **Testing Principle 声明**：明确"shape 固定，只能变值；batch_size 也固定"
2. **5 步强制推理流程**（Mandatory Reasoning Steps）：
   - Step 1 — Kernel dispatch analysis：确定哪个 kernel 在当前 input shape 下实际被调用
   - Step 2 — Reachability analysis：推导线程索引、循环变量的具体值域范围
   - Step 3 — Semantic distinguishability：在推导范围内判断变异是否产生不同结果
   - Step 4 — Value-level kill feasibility：如果有差异，是否能通过改变输入值触发
   - Step 5 — Conclusion：只有 Step 1-4 均找到具体可达场景才输出 `possibly_killable`
3. **明确禁止**："Do NOT suggest changing input shapes, dimensions, or batch size"
4. **区分"执行路径上的变异"与"死代码变异"**
5. **输出格式**：严格 JSON，包含 `verdict`, `confidence`, `reason_category`, `proof_sketch`, `reasoning`, `kill_strategy`, `suggested_test`

### 10.4 审查逻辑

```
Algorithm 5: Layer3_LLMVerification(m, kernel, equiv_detail, input_spec, call_llm)
──────────────────────────────────────────────────────────────────────────────────
Input:  已标记为 STRICT/CANDIDATE 的变异体 m
Output: 可能修改 m.status

 1  prompt ← BuildEquivVerifyPrompt(
        kernel.kernel_code, m.mutated_code, m.operator_name,
        m.site, m.status.value, m.error_message, input_spec, equiv_detail)

 2  raw_response ← call_llm(prompt)
 3  result ← ParseJSON(raw_response)

 4  if result is None then                    ▷ JSON 解析失败
 5      记录 verdict="error"，保持原判定
 6      return

 7  verdict    ← result["verdict"]            ▷ "confirmed_equivalent" 或 "possibly_killable"
 8  confidence ← result["confidence"]         ▷ 0.0 ~ 1.0

 9  if verdict = "possibly_killable" ∧ confidence > 0.7 then
10      old_status ← m.status
11      m.status ← SURVIVED                   ▷ 回退为存活
12      m.error_message ← "LLM rejected equiv ..."
13      记录 layer3.action = "reverted to SURVIVED (was <old_status>)"
14  else if verdict = "possibly_killable" then
15      保持原等价判定                         ▷ LLM 怀疑可杀但置信度不足以推翻
16      记录 layer3.action = "confirmed_low_confidence"
17  else
18      保持原等价判定                         ▷ LLM 主动确认等价
19      记录 layer3.action = "confirmed"
20  end if
```

> **注意**：当前生产代码（`full_block12.py:738`）对行 14-16 和行 17-19 两种场景统一记录为 `action="confirmed"`。论文实验数据分析时应区分"LLM 主动确认等价"（`verdict=confirmed_equivalent`）与"LLM 怀疑但不敢推翻"（`verdict=possibly_killable` 且 `confidence ≤ 0.7`），后者在严格意义上不算等价证据。

### 10.5 关键设计决策

1. **LLM 只能推翻等价判定（等价→SURVIVED），不能反向确认（SURVIVED→等价）**。这保证了 false-equivalent 方向的安全性（遵循设计原则 P4）。

2. **对 STRICT_EQUIVALENT 和 CANDIDATE_EQUIVALENT 均进行审查**，但 STRICT 有更强的先验证据（文本/静态规则可证明），被推翻的概率较低。

3. **置信度阈值 0.7**：LLM 不够确信时不推翻，避免 LLM 自身的 false positive。

4. **LLM 审查结果完整记录**在 `mutant.equiv_detail["layer3"]` 中，包含 `model`、`verdict`、`confidence`、`reason_category`、`proof_sketch`、`reasoning`、`kill_strategy`、`suggested_test`、`raw_response` 等字段。

> **关于"LLM 建议输入实际验证"**：当前代码中，Layer 3 **不会**用 LLM 建议的测试输入实际执行验证。LLM 返回的 `suggested_test` 仅做安全性校验（`validate_suggested_code`），结果记录在 metadata 中供分析使用，但不执行。实际的 kill 判定留给 Phase 2 增强测试。

---

## 11. 完整流水线伪代码

```
Algorithm 6: EMD_Pipeline(survived_mutants, kernel, llm_caller)
──────────────────────────────────────────────────────────────
Input:  Block 2 存活变异体列表 survived_mutants
        kernel 信息（含 kernel_code, reference_module_path）
        LLM 调用函数 llm_caller (可为 None)
Output: 更新每个变异体的 status 和 equiv_detail

 1  input_spec ← ExtractInputSpec(kernel.reference_module_path)

 2  for each m in survived_mutants do
 3      detail ← InitializeDetailDict(m, input_spec)
 4      decided ← False

        ▷ ─── Layer 0 ───
 5      (l0_verdict, l0_evidence) ← Layer0_SourceNormalization(m)
 6      detail["layer0"] ← l0_evidence

 7      if l0_verdict = STRICT then
 8          m.status ← STRICT_EQUIVALENT
 9          m.error_message ← "Textually equivalent (full program normalization)"
10          decided ← True
11      else if l0_evidence.cuda_eq ∧ ¬l0_evidence.py_eq then
12          hda ← AnalyzeHostDiff(m.mutated_code, m.site.line_start)
13          detail["layer0"]["host_diff_analysis"] ← hda
14          ▷ 不判定，继续
15      end if

        ▷ ─── Layer 1 ───
16      if ¬decided then
17          (l1_verdict, rule_hit) ← Layer1_StaticRules(m)
18          detail["layer1"] ← {rules_checked, rule_hit, rule_description, ...}
19          if l1_verdict = STRICT then
20              m.status ← STRICT_EQUIVALENT
21              m.error_message ← "Static rule: " + rule_hit
22              decided ← True
23          end if
24      end if

        ▷ ─── Layer 2 (子进程) ───
25      if ¬decided then
26          (is_equiv, l2_detail) ← Layer2_DynamicBitwise(m, kernel)  ▷ 隔离子进程, 超时 600s
27          detail["layer2"] ← l2_detail

28          if is_equiv then
29              m.status ← CANDIDATE_EQUIVALENT
30              m.error_message ← "Candidate equivalent (112 rounds, bitwise identical)"
31          else
32              ▷ 保持 SURVIVED
33              detail["layer2"]["divergence"] ← l2_detail.divergence
34          end if
35      end if

36      m.equiv_detail ← detail
37  end for

    ▷ ─── Layer 3 (LLM) ───
38  equiv_mutants ← {m ∈ survived_mutants | m.status ∈ {STRICT_EQ, CANDIDATE_EQ}}
39  if llm_caller ≠ None ∧ equiv_mutants ≠ ∅ then
40      for each m in equiv_mutants do
41          Layer3_LLMVerification(m, kernel, m.equiv_detail, input_spec, llm_caller)
42      end for
43  end if
```

---

## 12. 子进程隔离与超时机制

### 12.1 隔离架构

MutaKernel 的一条核心设计原则是**主进程永远不执行 CUDA 操作**。所有 CUDA 编译（`load_inline`）和 GPU 执行都在独立子进程中完成（`subprocess.Popen` + `start_new_session=True`）。

| 组件 | 执行位置 | CUDA 操作 |
|---|---|---|
| Block 1: 变异生成 | 主进程 | 无（纯 Python AST） |
| Block 2: 变异体执行 (kill/survive) | 子进程 (`_mutant_worker.py`, mode=run) | JIT 编译 + GPU forward |
| EMD Layer 0 | 主进程 | 无（字符串比较 + AST 分析） |
| EMD Layer 1 | 主进程 | 无（正则匹配 + AST 分析） |
| EMD Layer 2 | 子进程 (`_mutant_worker.py`, mode=equiv) | JIT 编译 2 个 kernel + 112 轮 GPU forward |
| EMD Layer 3 | 主进程 | 无（LLM API 调用） |

### 12.2 超时参数

| 参数 | 值 | 含义 |
|---|---|---|
| `MUTANT_TIMEOUT` | 180 秒 (3 min) | 单个变异体的编译 + 执行超时 |
| `EQUIV_TIMEOUT` | 600 秒 (10 min) | 等价检测的编译 + 112 轮动态测试超时 |

等价检测超时设为 10 分钟的原因：需要 JIT 编译 2 个 CUDA kernel（original + mutant，可能各需 30-60 秒）+ 112 轮 GPU forward。由于每个变异体的子进程完全独立（无编译缓存复用），编译开销不可摊销。

### 12.3 超时处理策略

当子进程超时时，主进程通过 `os.killpg(SIGKILL)` 强制终止子进程组，然后对该变异体返回 `{"is_equivalent": False, "error": "worker_timeout_or_crash"}`。即**保守地将超时视为"非等价"**（保持 SURVIVED），这遵循设计原则 P1（精度优先于召回）——宁可让一个等价变异体留在增强测试队列中浪费资源，也不将其误标为等价而终止后续测试。

---

## 13. EMD 结果与 Phase 2 增强测试的衔接

### 13.1 Tier 分级策略

EMD 的结果直接驱动 Phase 2 增强测试的优先级分级（`scripts/run_stress_enhance.py: classify_tier()`）：

```python
def classify_tier(mutant_meta):
    status = mutant_meta.get("status", "survived")
    ed = mutant_meta.get("equiv_detail", {})

    if status == "candidate_equivalent":
        return 3     # Layer 2 全通过 → 最可能等价，优先级最低

    l2 = ed.get("layer2", {})
    if l2 and l2.get("is_equivalent") is False:
        return 1     # Layer 2 找到差异 → 最可能可杀，优先级最高

    l3 = ed.get("layer3", {})
    if l3 and l3.get("verdict") == "possibly_killable":
        return 2     # LLM 认为可杀

    return 2         # 默认 Tier 2
```

| Tier | 含义 | 来源 | 实测 Kill Rate |
|---|---|---|---|
| Tier 1 | Layer 2 拒绝等价（bitwise 有差异） | `layer2.is_equivalent = false` | **84.8%** (128/151) |
| Tier 2 | LLM 拒绝等价 / 默认 | `layer3.verdict = "possibly_killable"` 或兜底 | **16.0%** (19/119) |
| Tier 3 | Candidate Equivalent | `status = "candidate_equivalent"` | **8.3%** (22/264) |

### 13.2 Layer 2 反例的价值传递

Phase 2 的 `tier1_replay` 维度会尝试**重放 Layer 2 的反例输入**（使用相同的 seed 和 policy），在 allclose 容差下重新执行以尝试杀死变异体。Tier 1 的高杀死率（84.8%）验证了 Layer 2 反例信息的价值。

```
Layer 2 发现 bitwise 差异
  ├─ 差异在 allclose 容差内 → Phase 2 tier1_replay 无法杀死
  │   └─ 但 Phase 2 其他维度 (value_stress 等) 可能放大差异 → 杀死
  └─ 差异超出 allclose 容差 → Phase 2 tier1_replay 直接杀死
```

### 13.3 比较标准差异总结

| 阶段 | 比较方法 | 比较对象 | 精度 |
|---|---|---|---|
| Block 2 初始测试（杀死判定） | `torch.allclose(atol=1e-2, rtol=1e-2)` | reference (PyTorch) vs mutant (CUDA) | 宽松 |
| EMD Layer 2（等价检测） | `BitwiseIdentical`（NaN-aware 位级精确） | original (CUDA) vs mutant (CUDA) | 严格 |
| Phase 2 增强测试（杀死判定） | `torch.allclose(atol=1e-2, rtol=1e-2)` | reference (PyTorch) vs mutant (CUDA) | 宽松 |

---

## 14. Layer 2 超时偏差分析

### 14.1 发现

在 Phase 2 增强测试阶段完成对 Tier 1 变异体的测试后，对 23 个 survived mutant 做深入分析时发现：**所有 23 个 mutant 的 Layer 2 都记录为 `is_equivalent=false`，但 `total_rounds=0`、`divergence={}`——即 Layer 2 根本没有执行任何测试轮次。**

### 14.2 根因

Layer 2 的 worker 子进程在 600 秒超时限制内**未完成 CUDA JIT 编译**（甚至未开始任何一轮测试）。超时后，主进程返回 `{"is_equivalent": false, "error": "worker_timeout_or_crash"}`。

```json
{
  "is_equivalent": false,
  "tested_random_seeds": [],
  "tested_policies": [],
  "total_rounds": 0,
  "error": "worker_timeout_or_crash",
  "divergence": {},
  "time_ms": 600149
}
```

超时原因推测：某些变异引入了更多 grid blocks（如 `grid_x = (N+63)*64 = 1,052,608`），可能触发 nvcc 更长的编译时间或运行时 GPU 资源分配延迟。这与 §6.3 中描述的"每个变异体子进程完全独立、无编译缓存"的架构直接相关。

### 14.3 影响链条

```
Layer 2 超时 → is_equivalent=false (保守默认值)
  → classify_tier() 看到 layer2.is_equivalent==False → 归类为 Tier 1
    → tier1_replay 尝试重放 L2 divergence → 无数据可重放 (divergence={})
      → 增强测试所有其他维度正常运行 → 全部未能杀死
        → LLM 分析 → 判定 unkillable
```

### 14.4 对实验结果的影响

**对最终杀死/存活判定无影响**：虽然 Tier 分类有误（应归为 Tier 2/3 而非 Tier 1），但增强测试对所有 Tier 都执行了完整的全维度测试（value_stress、dtype_stress、training_stress、repeated_run、config_stress + LLM 迭代分析），不存在因 Tier 分类错误而遗漏任何测试维度的情况。唯一无效的步骤是 `tier1_replay`（因无 divergence 数据），但其他所有维度均正常执行。

**对 Tier 统计的影响**：报告中的 "Tier 1: 151 tested" 实际包含了 23 个本应归入更高 Tier 的 mutant。论文写作时需注意此偏差并如实报告。

### 14.5 这些 survived mutant 是否真正等价

LLM（DeepSeek）在第 1 轮即对全部 23 个 mutant 判定为 `killable=false`，给出的 `reason_category` 分布为：

| reason_category | 数量 | 含义 |
|---|---|---|
| `predicate_unreachable` | 11 | 变异代码路径在 fixed-shape 下不可达 |
| `value_insensitive` | 10 | 变异的值差异不影响最终输出 |
| `infection_no_propagation` | 2 | 感染发生但不传播到输出 |

典型等价模式包括：
- **Grid size 膨胀**：`(N+63)/64 → (N+63)*64`，多余 block 因 bounds check 不写入
- **分支条件松弛**：`idx < K → idx <= K`，但 thread index 永远达不到边界值
- **死代码变异**：变异的 kernel specialization（如 `gelu_kernel<half>`）在 float32 输入下从未被调用
- **注释变异**：`const_perturb` 变异了注释文本

---

## 15. 报告系统

### 15.1 MutationReporter 输出

报告系统（`src/mutengine/report.py`）输出：

| 输出 | 格式 | 内容 |
|---|---|---|
| 每个 kernel 的详细结果 | JSON | 含每个变异体的 `equiv_detail`（Layer 0-3 完整证据链） |
| 汇总统计 | JSON + Markdown | 按 kernel / category / operator 分拆的统计 |
| 等价体详情表 | Markdown | 每个等价体的 Kernel、Mutant ID、算子、等价级别、判定证据 |

### 15.2 汇总统计拆分

- `total_strict_equivalent` 和 `total_candidate_equivalent` 分别计数
- 保守口径 / 乐观口径两种 mutation score
- 按 category（A/B/C/D）和按 operator 的分口径 score

---

## 16. 有效性威胁与未来工作

### 16.1 当前有效性威胁

**1. 等价检测的不完备性 (Internal Validity)**

等价判定本质上不可判定 (Budd & Angluin, 1982)。我们的方法存在两类错误：
- **False Equivalent (漏杀)**：将非等价变异体误判为等价，终止后续测试。Layer 3 LLM 审查旨在缓解此问题。
- **False Non-Equivalent (误标)**：将等价变异体保留在测试队列中，浪费增强测试资源。如 §14 中的 Layer 2 超时案例。

**2. Layer 2 的超时偏差 (Construct Validity)**

Layer 2 的 worker 超时导致 `is_equivalent=false` 的保守默认判定，使部分等价变异体被错误归入 Tier 1。最终不影响杀死/存活判定，但影响 Tier 统计分布的准确性。

**3. Fixed-shape 依赖的等价性 (External Validity)**

我们的等价判定在 **fixed-shape 契约** 下进行。某些变异体在当前 shape 下等价，但在不同 shape 下可能可杀（如 grid size 计算变异在 `size=128` 下等价但在 `size=257` 下不等价）。严格来说应称为"在特定配置下的功能等价"而非"语义等价"。

**4. 动态检测的统计局限性 (Construct Validity)**

Layer 2 的 112 轮检测是统计性的，不是证明性的。未观测到差异不等于不存在差异。

**5. LLM 判断的可靠性 (Internal Validity)**

Layer 3 的 LLM 判断可能存在数学推理错误、遗漏执行路径、过度自信等问题。目前 LLM 对 survived mutant 的判定未经系统化人工验证。

**6. GPU 并行行为的覆盖不足 (Construct Validity)**

当前的动态检测仅比较 kernel 输出（值等价），不覆盖 data race、warp divergence、shared memory bank conflict 等 GPU 特有的并行行为属性。

**7. 算子定向策略的不完全覆盖**

生产 worker 中仅 8 类算子有专属定向策略，其余 8 类（`stab_remove`, `scale_modify`, `acc_downgrade`, `reduction_reorder`, `init_modify`, `cast_remove`, `broadcast_unsafe`, `layout_assume`）使用通用兜底策略。未来可将 `equivalent_detector.py` 中的完整 16 类策略同步到生产路径。

### 16.2 未来工作

1. **GPU kernel 的形式化等价验证**：探索 GPUVerify (Betts et al., OOPSLA 2012) 对 candidate equivalent 变异体的 data race freedom 和 barrier divergence freedom 验证。

2. **PTX/SASS 级 TCE**：比较 nvcc 编译产出的 PTX 中间代码或 SASS 机器码，利用 GPU 编译器的优化能力检测等价变异体。

3. **配置感知的等价检测**：在标准 fixed-shape 之外，额外测试若干关键 shape 变体（如 `size=blockDim±1`），识别配置依赖的等价性。

4. **LLM 判断的人工验证**：对 LLM 判定为 unkillable 的 survived mutant 进行系统化人工审核，建立 ground-truth 数据集。

5. **并行拓扑感知策略**：根据 kernel 的 grid/block 配置生成恰好触发边界线程的输入。

---

## 17. 论文各章节对应表述建议

| 章节 | 建议表述 |
|---|---|
| **Method: EMD 模块** | "We propose a four-layer equivalent mutant detection (EMD) pipeline for GPU CUDA kernel mutations: (L0) CUDA-aware source normalization with host-code AST analysis and mutation domain classification, (L1) four static equivalence rules targeting GPU-specific patterns (thread index boundary unreachability, dead write, mask boundary no-reach, dead host constant under fixed-shape testing), (L2) NaN-aware bitwise dynamic comparison driven by operator-directed stress policies, totaling 100 random rounds plus 6 directed policies × 2 sub-trials = 112 rounds per mutant, and (L3) LLM-based equivalence verification with full evidence chain. L0/L1 produce STRICT_EQUIVALENT (provable); L2 produces CANDIDATE_EQUIVALENT (heuristic); L3 can only revoke equivalence (equiv→survived), never promote." |
| **Experiment: Score 报告** | 报告保守口径（excl. strict_eq）和乐观口径（excl. strict_eq + candidate_eq）两种 mutation score；主表使用保守口径 |
| **Threats to Validity** | "Equivalent mutant detection is fundamentally undecidable. Our STRICT_EQUIVALENT is limited to textually/statically provable cases; CANDIDATE_EQUIVALENT is heuristic. Dynamic detection is based on output equivalence over 112 rounds, not a proof. LLM verification is subject to model capability limits." |

---

## 18. 涉及文件清单

| 文件 | 角色 | 在 EMD 中的功能 |
|---|---|---|
| `src/models.py` | 数据模型 | `MutantStatus` 枚举（8 状态）、`MutationTestResult`（含 `mutation_score` / `mutation_score_optimistic`）、`Mutant.equiv_detail` 字段 |
| `src/mutengine/equivalent_detector.py` | 核心库 | `_bitwise_identical`（NaN-aware 比较）、`_normalize_cuda_source` / `_normalize_python_source`（归一化）、`_extract_cuda_strings`（CUDA 字符串提取）、`_analyze_host_diff`（宿主差异分析）、`CompareResult` 枚举、`EquivalentDetector` 类（smoke 脚本用） |
| `src/mutengine/static_equiv_rules.py` | 静态规则 | 4 条规则实现 + `check_all_rules()` 公共 API |
| `src/mutengine/parser/cuda_parser.py` | CUDA 解析器 | `CudaParser`（从 Python 提取嵌入式 CUDA 字符串） |
| `src/stress/policy_bank.py` | 策略库 | 21 条压力策略 + `STRESS_POLICIES` 字典 |
| `src/stress/llm_analyzer.py` | LLM 模块 | `EQUIV_VERIFY_PROMPT`（含 Testing Principle + 5 步推理）、`verify_equivalent_with_llm()`、`build_equiv_verify_prompt()`、`_format_layer_evidence()`、`_extract_context()` |
| `src/mutengine/report.py` | 报告系统 | `MutationReporter`（汇总统计拆分、双口径 score、等价体详情表） |
| `scripts/full_block12.py` | **生产入口** | `_process_one_kernel()` — EMD 四层流水线的实际执行逻辑（Layer 0/1 在主进程、Layer 2 子进程、Layer 3 LLM）、`check_equiv_isolated()`、`_extract_input_spec()` |
| `scripts/_mutant_worker.py` | 子进程 worker | `_equiv_mode()` — Layer 2 的 GPU 执行逻辑（编译、112 轮比较、结构化异常处理）；含独立的 `_bitwise_identical` 和 `OPERATOR_DIRECTED_POLICIES`（8 类算子） |
| `scripts/run_stress_enhance.py` | Phase 2 入口 | `classify_tier()` — 基于 EMD 结果的 Tier 分级策略 |

---

## 19. 参考文献

| # | 引用 | 出处 |
|---|------|------|
| 1 | Budd, T.A. & Angluin, D. "Two notions of correctness and their relation to testing" | *Acta Informatica*, 18(1), 1982 |
| 2 | Madeyski, L. et al. "Overcoming the Equivalent Mutant Problem: A Systematic Literature Review and a Comparative Experiment of Second Order Mutation" | *IEEE TSE*, 40(1), 2014 |
| 3 | Papadakis, M. et al. "Trivial Compiler Equivalence: A Large Scale Empirical Study of a Simple, Fast and Effective Equivalent Mutant Detection Technique" | *IEEE ICSE*, 2015; *IEEE TSE*, 43(10), 2017 |
| 4 | Offutt, A.J. & Craft, W.M. "Using compiler optimization techniques to detect equivalent mutants" | *STVR*, 4(3), 1994 |
| 5 | Offutt, A.J. & Lee, S.D. "An Empirical Evaluation of Weak Mutation" | *IEEE TSE*, 22(5), 1996 |
| 6 | Kushigian, B. et al. "Equivalent Mutants in the Wild: Identifying and Efficiently Suppressing Equivalent Mutants for Java Programs" | *ISSTA*, 2024 |
| 7 | Tian, Z. et al. "Large Language Models for Equivalent Mutant Detection: How Far Are We?" | *ISSTA*, 2024 |
| 8 | Holling, D. et al. "Nequivack: Assessing Mutation Score Confidence" | *IEEE ICST*, 2016 |
| 9 | Chatzikonstantinou, G. et al. "MutateNN: Mutation Testing of Image Recognition Models Deployed on Hardware Accelerators" | *ACM FSE*, 2024; *IEEE ICST Mutation Workshop*, 2025 |
| 10 | Betts, A. et al. "GPUVerify: A Verifier for GPU Kernels" | *ACM OOPSLA*, 2012 |
| 11 | Jia, Y. & Harman, M. "An Analysis and Survey of the Development of Mutation Testing" | *IEEE TSE*, 37(5), 2011 |
