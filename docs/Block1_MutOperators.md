# Block 1: MutOperators — ML 算子数值语义变异算子族

> **对应文件**: `src/mutengine/operators/base.py`, `arithmetic.py`, `gpu_parallel.py`, `ml_semantic.py`, `llm_pattern.py`
>
> **论文位置**: Section 3.2（变异算子设计）— 核心贡献 1 的智力内核

---

## 一、问题背景

### 1.1 现有变异测试算子的局限

传统变异测试的算子设计（如 PIT、µJava、LittleDarwin 等）面向通用程序，仅覆盖：
- 算术运算符替换 (`+` → `-`)
- 关系运算符替换 (`<` → `<=`)
- 常量扰动 (`1024` → `1023`)

这些算子对 GPU kernel 的两类核心缺陷**完全无能为力**->传统通用变异算子主要覆盖表层语法与通用控制流错误，难以系统性建模 ML 数值语义缺陷和 GPU 并行语义缺陷。

| 缺陷类型 | 具体表现 | 为什么通用算子覆盖不了 |
|---------|---------|-------------------|
| **数值语义缺陷** | 缺少 max-subtraction 导致 softmax 溢出、FP16 累加器精度不足、epsilon 缺失导致除零 | 这些是 ML 领域特有的编程范式，不是"运算符错误" |
| **GPU 并行语义缺陷** | 线程索引维度混淆、同步屏障缺失、mask 边界条件 off-by-one | 并行编程范式通用变异测试未设计 |



### 1.2 已有 GPU 变异测试的不足

MUTGPU (ICST 2020) 提出了 9 个 GPU 并行编程变异算子，但：
- **不涉及 ML 数值语义**：MUTGPU 面向传统 HPC（高性能计算）程序（矩阵乘法、排序等），不理解 softmax、LayerNorm、Attention 等 ML 算子的数值约束
- **6 年前的工作**：LLM 生成 kernel 的错误模式（如遗忘 `.contiguous()`、忽略 `expand`）完全不在其考虑范围内
- **仅 6 个 CUDA SDK 程序**：样本量极小，未在工业级 ML 算子上验证

### 1.3 LLM 生成 kernel 的特殊错误模式

通过分析 KernelBench 和 TritonBench 的失败样本，LLM 生成 GPU kernel 存在高频、独特的错误模式：

（放到实验节里面报告：
1. 总共分析了多少失败样本
2. 有多少可归因
3. 每类错误有多少例
4. 多标注者一致性如何
5. 这些类别如何映射到 C/D operators）->用deepseek R1+V3；GPT的两个系列的api；

1. **遗忘数值稳定化**：LLM 不理解为什么 softmax 要减 max，直接 `exp(x)` 导致溢出
2. **精度处理缺失**：用 FP16 做累加，小规模测试看不出差异，但大规模下精度崩塌
3. **epsilon 遗漏**：LayerNorm 的分母不加 epsilon，大多数随机输入不触发，但接近零时除零
4. **scale factor 缺失**：Attention 的 `1/√d` 被遗忘，容差内可能不被检出
5. **显式类型转换缺失**：隐式精度降级
6. **shape 对齐假设错误**：LLM 忘记 `.expand()` 或 `.contiguous()`

**这些错误恰恰是 `torch.allclose` + 随机输入最容易漏掉的。**

---

## 二、本块解决了什么问题

### 2.1 核心贡献

MutOperators 设计了 **16 个变异算子**，分为 4 个类别，形成了**首个面向 ML 算子数值语义的变异算子族**：

| 类别 | 算子数 | 定位 | 对应论文位置 |
|------|--------|------|------------|
| **A: 通用算术变异** | 3 | Baseline（对照组） | 3.2.1 |
| **B: GPU 并行语义** | 4 | 适配 MUTGPU 到 LLM kernel 场景 | 3.2.2 |
| **C: ML 数值语义** | 7 | **核心创新**，本文独有 | 3.2.3 |
| **D: LLM 错误模式** | 2 | 数据驱动，从真实 LLM 错误中提取 | 3.2.4 |

### 2.2 为什么有意义

1. **C 类算子是全新的**：学术界首次系统设计面向 ML 数值语义的变异算子，7 个算子分别模拟 7 种不同的数值缺陷
2. **每个算子有明确的"逃逸机制"**：每个 C 类算子都附带"为什么 `torch.allclose` 可能检不出"的理论解释，这不是随意构造
3. **D 类算子是数据驱动的**：直接从 KernelBench/TritonBench 的 LLM 失败日志中提炼，不是想象出来的
4. **A/B 作为对照组**：A（通用变异）和 B（GPU 并行）的存在不是填数，而是为了实验中对比——"数值语义类缺陷 vs 传统缺陷的检出率差异"

### 2.3 做到了什么别人做不到的事情

| 能力 | MUTGPU (2020) | 通用变异工具 | **MutOperators** |
|------|:---:|:---:|:---:|
| 通用算术变异 | ✗ | ✓ | ✓ (A类) |
| GPU 并行语义 | ✓ | ✗ | ✓ (B类, 适配 Triton/CUDA 双语法) |
| ML 数值稳定性 | ✗ | ✗ | **✓ (C1-C7)** |
| LLM 特有错误模式 | ✗ | ✗ | **✓ (D1-D2)** |
| Triton + CUDA 双语言 | ✗ | ✗ | **✓** |

---

## 三、实现流程

### 3.1 架构概览

```
MutationOperator (抽象基类)
├── find_sites(source, tree) → List[MutationSite]   # 在源码中找变异位点
├── apply(source, site) → str                         # 对某个位点施加变异
└── generate_mutants(source, kernel_id, tree) → List[Mutant]  # 便捷方法

注册机制: __init_subclass__ 自动注册到 _OPERATOR_REGISTRY
```

### 3.2 基类设计 (`base.py`)

```
MutationOperator
  │
  ├── 属性: name, category, description
  ├── 抽象方法: find_sites(), apply()
  ├── 便捷方法: generate_mutants()
  │     └── 自动调用 find_sites → apply → 构造 Mutant 对象
  │
  └── 辅助工具:
        ├── _replace_in_source(): 按行范围精确替换
        ├── _find_pattern_sites(): 正则搜索位点
        ├── _split_lines(): 按行分割保留换行
        └── get_all_operators() / get_operators_by_category(): 注册表查询
```

**关键设计**：算子通过 `__init_subclass__` 在定义时自动注册到全局字典 `_OPERATOR_REGISTRY`，无需手动管理。`generate_mutants()` 方法是模板方法，子类只需实现 `find_sites` 和 `apply`。

### 3.3 Category A: 通用算术变异 (`arithmetic.py`)

#### A1 `ArithReplace` — 算术运算符替换

```
输入: 完整源码 (Python wrapper + CUDA 字符串)

Python/Triton 路径 (AST + tokenizer):
  ├── [Step 1] ast.parse() 解析为 AST
  ├── [Step 2] ast.walk() 遍历所有 BinOp 节点
  ├── [Step 3] 对每个 +/-/*// 运算:
  │     ├── 从 AST 获取左右操作数的精确位置
  │     ├── tokenize 在操作数 span 之间定位运算符 token
  │     └── 确认 token 位置 → 生成 MutationSite
  └── 特点: 排除 decorator 内的运算, 注释/字符串不误匹配

CUDA C++ 路径 (字符扫描):
  ├── 逐行扫描, 先 _strip_cuda_comment 去除 // 注释
  ├── 对每个 +/-/*/ 字符:
  │     ├── 排除复合运算: ++, --, ->, +=, -=, *=, /=, //, /*
  │     └── 二元判定: 向前跳过空白, 前一个字符是 \w / ) / ] / digit
  │           → 则为二元运算符 → 生成 MutationSite
  └── 允许少量误判 (如 float *ptr), 产生 STILLBORN 自动过滤

合并: AST sites ∪ CUDA sites, 按 (line, col) 去重
apply(): + ↔ -, * ↔ / 互换 (两条路径共用同一替换逻辑)
```

#### A2 `RelOpReplace` — 关系运算符替换

```
Python/Triton 路径:
  同 A1 的 AST 方法, 遍历 Compare 节点
  替换映射: < ↔ <=, > ↔ >=, == ↔ !=

CUDA C++ 路径 (字符扫描):
  ├── 优先匹配 2 字符运算: <=, >=, ==, !=
  ├── 单字符 < / >:
  │     ├── 排除 << (左移) 和 >> (右移)
  │     ├── _is_template_open(): 检查 < 前是否为 static_cast / data_ptr 等 → 跳过
  │     └── _is_template_close(): 检查 > 后是否紧跟 '(' 或前为类型名 → 跳过
  └── 替换映射与 Python 路径相同

合并: AST sites ∪ CUDA sites, 按 (line, col) 去重
```

#### A3 `ConstPerturb` — 常量微扰

```
两步搜索:
  ├── 整数: 正则 \b(?!0\b|1\b)\d+\b(?!\.)
  │   排除 0 和 1 (改 0±1 或改 1±1 语义变化太大)
  │   排除 GPU 调度配置行: _GPU_CONFIG_RE 检测行内是否包含
  │     BLOCK_SIZE / blockDim / gridDim / dim3 / num_warps /
  │     threadIdx / blockIdx 等 GPU 配置关键词,
  │     避免 block_size=256 → 255 这类导致 STILLBORN 的无意义变异
  │   每个整数生成两个变异: +1 和 -1
  │
  └── 浮点数: 复杂正则处理 科学计数法/后缀(f,F,l,L)
      排除 0.0 和 1.0
      每个浮点数生成两个变异: ×1.01 和 ×0.99
```

### 3.4 Category B: GPU 并行语义变异 (`gpu_parallel.py`)

#### B1 `IndexReplace` — 线程/块索引替换

```
Triton 路径:
  正则匹配 tl.program_id(N) → 替换为 program_id(M), M≠N, M∈{0,1,2}
  每个位点生成 2 个变异 (其余两个轴)

CUDA 路径:
  正则匹配 threadIdx.x/y/z, blockIdx.x/y/z 等
  每个位点生成 2 个变异 (其余两个维度)

关键: _is_in_comment_or_string() 检测避免注释/字符串内误匹配
```

#### B2 `SyncRemove` — 同步屏障删除

```
搜索: tl.debug_barrier() 或 __syncthreads()
变异: 删除整行（如果行上只有该语句）或仅删除调用

冗余检测 (reduction_tail 标记):
  当 __syncthreads() 后的第一条非空语句为 if (tid == 0) / if (threadIdx.x == 0) 时,
  表明 sync 出现在归约尾部、后续仅 tid=0 读取 shared[0], 删除不产生竞态。
  此类变异位点标记 node_type = "cuda_syncthreads:reduction_tail",
  归因时自动归为 Type 1 (Algorithmically Absorbed)。

  同理, 如果此 sync 是 kernel 中最后一个 __syncthreads, 且后续所有
  shared memory 访问仅使用 shared[0], 也标记为 reduction_tail。
```

#### B3 `MaskBoundary` — mask 边界条件变异

```
触发条件:
  Triton: 行内包含 "mask=" 或 "tl.where"
  CUDA:   C-style if/elif 语句满足以下任一:
    ├── 行内直接出现 threadIdx / blockIdx
    └── 行以 '{' 结尾或含 'return;', 且行内有比较运算符
        (覆盖用派生变量 row/col/idx 做的边界守卫)

搜索两种比较模式:
  ├── word < word  (原有)
  └── word >= word (新增, 匹配 `if (i >= n) return;` 等早返回模式)

变异 (< 路径, 仅保留收紧方向):
  └── x < y   → x < y - 1 (边界收紧, 遗漏末尾元素)
  注: 宽松方向 x < y → x <= y 已排除——
      对多数 GPU kernel 而言宽松 mask 几乎必然越界 → STILLBORN,
      不产生有意义的存活变异体

变异 (>= 路径):
  ├── x >= y  → x > y     (边界收紧, 不再跳过恰好等于 y 的线程)
  └── x >= y  → x >= y+1  (边界收紧, 多跳过一个线程)
```

#### B4 `LaunchConfigMutate` — 启动配置扰动

```
搜索 4 种模式:
  Triton 路径:
    ├── triton.cdiv(...)     # Triton ceiling-division 调用
    └── X // Y               # Python 整数除法 (常用于 grid 计算)

  CUDA C++ 路径:
    ├── cdiv(...)            # C++ ceiling-division helper 函数
    └── (expr - 1) / Y      # (N + BLOCK - 1) / BLOCK 向上取整惯用法

变异 (仅保留 -1 方向):
  └── expr → expr - 1   (grid 少分配一个 block, 边缘数据遗漏)
  注: +1 方向已排除——增大 grid/block 通常只多启动冗余线程,
      kernel 内的 mask 守卫会过滤掉越界线程, 变异体大概率存活 (等价)

影响: grid size 减少，边缘数据可能被遗漏
```

### 3.5 Category C: ML 数值语义变异 (`ml_semantic.py`) — 核心创新

这是整个项目最关键的部分。7 个算子的共同特点是：**模拟的缺陷在常规随机输入下可能不被 `torch.allclose` 检出，但在特定数值条件下会导致严重错误。**

#### C1 `StabRemove` — 移除数值稳定化

```
目标: softmax 中的 x - max(x) 技巧

搜索逻辑:

  Triton / PyTorch 路径 (三种模式):
  ├── var - tl.max(var, ...)     # Triton
  ├── var - torch.max(var, ...)  # PyTorch
  └── var - var.max(...)          # 方法调用

    精确匹配流程:
    1. 正则匹配 "X - tl.max(" 头部
    2. _balanced_paren_span() 找到完整括号范围
    3. _first_call_arg() 提取第一个参数
    4. 验证: 第一个参数 == 减号前的变量名 (确认是 x - max(x) 而非其他减法)
    5. 变异: 删除 "- tl.max(...)" 部分，只保留 x

  CUDA C++ 路径 (两种模式):
  ├── expf(val - row_max)        # exp 参数内包含减法 (单行写法)
  │   匹配正则: \bexpf?\s*\(\s*([\w.\[\]]+)\s*-\s*([\w.\[\]]+)\s*\)
  │   → 变异: expf(val)           # 移除 max-subtraction
  │
  └── val - max_val              # 减去含 "max/MAX" 的变量 (分行写法)
      匹配正则: ([\w.\[\]]+)\s*-\s*([\w]*(?:max|MAX)[\w]*)
      → 变异: val                 # 仅保留左操作数
      去重: 若该位置已被 expf(...) 模式覆盖，则跳过

逃逸机制: 小值输入时 exp(x) 不溢出，两个版本输出一致
```

#### C2 `AccDowngrade` — 累加器精度降级

```
目标: FP32 累加器被降为 FP16

Triton / PyTorch 路径 (6 种模式):
  ├── tl.zeros(..., dtype=tl.float32)  → dtype=tl.float16
  ├── torch.zeros(..., dtype=torch.float32)  → dtype=torch.float16
  ├── .to(torch.float32)  → .to(torch.float16)
  ├── .to(tl.float32)  → .to(tl.float16)
  ├── .float()  → .half()
  └── .double()  → .float()

CUDA C++ 路径 (3 种模式):
  ├── static_cast<float>(expr)  → static_cast<__half>(expr)
  ├── (float)expr               → (__half)expr     # C-style cast
  └── __half2float(expr)        → 提取 expr 本身   # 移除精度提升

辅助函数 _zeros_fp32_call_sites():
  专门处理 zeros(..., dtype=float32) 的完整匹配
  用 _balanced_paren_span 确保括号完整

逃逸机制: 小规模 reduction 时 FP16 精度足够, 差异在 atol=1e-2 内
```

#### C3 `EpsilonModify` — epsilon 常量变异

```
目标: LayerNorm / safe_div / log 中的 epsilon

搜索 (两类正则, 取并集):
  1. 科学计数法: 1e-5, 1e-6f, 1e-8F 等
     正则 _EPS_SCI: [+-]?(\d+)?(\.\d+)?[eE][+-]?\d+[fFlL]?
  2. 小数表示:   0.00001f, 0.000001 等 (小数点后 ≥4 个前导零)
     正则 _EPS_DEC: 0\.0{4,}\d+[fFlL]?

  筛选: 绝对值 ≤ 1e-3 (排除 1e-1, 0.01 等非 epsilon 常量)

  同时在 Python 代码和 CUDA 字符串中搜索，
  自动处理 CUDA 的 f 后缀 (apply 时保留原后缀)

每个 epsilon 生成 2 个变异:
  ├── eps → 0 / 0f        # 完全移除保护 (保留后缀)
  └── eps → 1e-2 / 1e-2f  # 大幅增大 (保留后缀)

逃逸机制: 大多数随机输入不触发接近零的情况
```

#### C4 `ScaleModify` — scale factor 变异

```
目标: Attention / Normalization 中的 1/√d 缩放

Triton / PyTorch 路径:
  ├── 1 / math.sqrt(x)  或 1.0 / torch.sqrt(x)
  │   → 变异: 删除 "1/" 前缀, 变成 math.sqrt(x) (含义反转)
  │
  └── math.rsqrt(x) / torch.rsqrt(x) / tl.rsqrt(x)
      → 变异: 替换为参数本身 (删除 rsqrt 调用)

CUDA C++ 路径 (2 种模式):
  ├── 1.0f / sqrtf(x) 或 1.0 / sqrt(x)
  │   匹配正则 _CUDA_INV_SQRT: (?:\b1\.0f?\b|\b1\b)\s*/\s*sqrtf?\s*\(
  │   括号定位: 使用 m.end()-1 精确找到 sqrtf 的左括号
  │   → 变异: 删除 "1.0f/" 前缀, 变成 sqrtf(x)
  │
  └── rsqrtf(expr) / rsqrt(expr)
      匹配正则 _CUDA_RSQRT: \brsqrtf?\s*\(
      → 变异: 替换为参数本身 (1.0f)

  典型命中: BatchNorm 中 rsqrt(var + eps), GroupNorm 中 rsqrtf(variance + eps)

逃逸机制: 如果 d 较小, scale 差异在 tolerance 内可能不被检出
```

#### C5 `CastRemove` — 删除显式类型转换

```
Triton / PyTorch 路径 (4 种模式):
  ├── .to(torch.float32/float16/bfloat16/int32/int64)  → 删除
  ├── .to(tl.float32/float16/bfloat16/int32)           → 删除
  ├── .float() / .half() / .bfloat16()                  → 删除
  └── tl.cast(x, dtype)                                 → 替换为 x

CUDA C++ 路径 (3 种模式):
  ├── static_cast<T>(expr)  → 提取 expr (删除 cast 包装)
  │   匹配 T ∈ {float, double, __half, at::Half, at::BFloat16, int, ...}
  │
  ├── (float)expr / (double)expr / (__half)expr  → 删除 C-style cast
  │
  └── __float2half(expr) / __half2float(expr)    → 提取 expr
      __double2float(expr) / __float2half_rn(expr)

逃逸机制: 如果输入 dtype 恰好与目标 dtype 一致, 删除 cast 无影响

冗余检测 (redundant 标记, 方案 C):
  对 static_cast<T>(expr), 检查所在行的上下文类型:
    ├── float var = static_cast<float>(expr);  → LHS 类型与 T 匹配 → 冗余
    ├── const float var = ...                  → 同上
    └── var += static_cast<float>(expr);       → 行中含 float 关键字/字面量 → 冗余
  冗余 cast 标记 node_type = "cast:cuda_static_cast:redundant",
  C++ 隐式类型转换保证删除该 cast 无语义变化。
  归因时自动归为 Type 1 (Algorithmically Absorbed)。
```

#### C6 `ReductionReorder` — reduction 归约顺序变异

```
目标: 浮点结合律非精确性

Triton / PyTorch 路径:
  ├── tl.sum(tensor, axis=N)    → tl.sum(tensor[::-1], axis=N)
  └── torch.sum(tensor, dim=N)  → torch.sum(torch.flip(tensor, (N,)), dim=N)

CUDA C++ 路径: 暂不支持
  CUDA 归约通常是手写 for 循环 + shared memory 并行归约，
  无法用单行正则匹配。未来可通过 AST 级分析或 LLM 辅助变异实现。

含义: 反转输入顺序后求和，利用浮点加法不满足结合律的性质

逃逸机制: 差异通常在 1e-7 量级，远小于 atol=1e-2
```

#### C7 `InitModify` — identity element 变异

```
目标: min/max reduction 的初始值

Python 路径:
  ├── float('-inf')           → float('-1e10') 或 0.0
  ├── float('inf')            → float('1e10') 或 0.0
  └── tl.full(..., float('-inf'), ...)  → 替换 -inf 为 0.0 或 -1e10

CUDA C++ 路径 (6 种模式, 每种生成 2 个变异):
  ├── -INFINITY    → 0.0f 或 -1e10f   # <math.h>
  ├── INFINITY     → 0.0f 或 1e10f
  ├── -FLT_MAX     → 0.0f 或 -1e10f   # <cfloat>
  ├── FLT_MAX      → 0.0f 或 1e10f
  ├── -HUGE_VALF   → 0.0f 或 -1e10f
  └── HUGE_VALF    → 0.0f 或 1e10f

  典型命中: max reduction 中 `float thread_max = -INFINITY;`
           或 `scalar_t thread_max = -FLT_MAX;`

逃逸机制: 如果实际数据范围在 [-1e10, 1e10] 内，-1e10 和 -inf 行为一致
```

### 3.6 Category D: LLM 错误模式变异 (`llm_pattern.py`)

D 类算子作用于 **Python 层**的 PyTorch 张量操作（不作用于 CUDA 内核字符串内部），
因为 `.expand()` / `.contiguous()` 等是 Python wrapper 中的调用。

#### D1 `BroadcastUnsafe` — 移除显式形状对齐

```
搜索 (5 种模式):
  ├── .expand(...)         # PyTorch 显式广播
  ├── .expand_as(...)      # 按目标张量广播
  ├── .broadcast_to(...)   # 显式广播到指定 shape
  ├── tl.broadcast_to(...) # Triton 广播
  └── .unsqueeze(...)      # 添加维度

  每个模式: 正则定位方法名 + 括号平衡找到完整调用范围
  过滤: _outside_comment_or_string() 排除注释/字符串内的误匹配
        (已支持三引号字符串 ''' / """)

  不包含 .reshape() / .view():
    删除必需的形状变换几乎必定触发 shape mismatch → STILLBORN,
    不产生有意义的变异体

变异:
  ├── tensor.expand(...)       → tensor          # 删除 .expand(...) 部分
  ├── tensor.expand_as(...)    → tensor          # 删除 .expand_as(...) 部分
  ├── tensor.broadcast_to(...) → tensor          # 删除 .broadcast_to(...) 部分
  ├── tl.broadcast_to(x, ...) → x               # 提取第一参数
  └── tensor.unsqueeze(...)    → tensor          # 删除 .unsqueeze(...) 部分

  替换方式: _replace_at_columns() 按精确列位置替换
  (解决同一行多个同名调用时 str.replace 无法区分的问题)

来源: TritonBench/KernelBench 错误日志中 LLM 频繁遗忘显式 broadcast
```

#### D2 `LayoutAssume` — 移除内存布局保证

```
搜索 (2 种模式):
  ├── .contiguous()        # 确保张量连续
  └── .is_contiguous()     # 连续性检查

  过滤:
  ├── _outside_comment_or_string() 排除注释/字符串/docstring 内的误匹配
  └── **layout-sensitive 前置条件** (仅 .contiguous() 路径):
      _LAYOUT_SENSITIVE 正则检查同一行 .contiguous() 前方是否存在
      layout 改变操作 (.transpose / .permute / .mT / .T / .t() 或切片 [...:...]),
      若无则跳过——防御性冗余 .contiguous() 删除后无实际效果 (等价变异)

变异: 删除调用 (通过 _replace_at_columns 精确列替换)
  ├── out.transpose(1, 2).contiguous()  → out.transpose(1, 2)
  ├── x[::2].contiguous()              → x[::2]
  └── x.is_contiguous()                → (删除检查)

  注: 纯防御性调用如 x.contiguous() (前方无 transpose/slice) 不生成变异

来源: LLM 生成的 kernel 常假设输入 tensor 已连续
      但经过 transpose/slice 后实际不连续
```

### 3.7 辅助工具函数

辅助函数分布在三个文件中，按层级被不同类别复用：

```
base.py — 核心共享 (A/B/C/D 四个类别共用):
  _strip_cuda_comment(line)
    → 去除 C++ 行注释 (// ...), 保留字符串字面量中的 //
  _cuda_find_pattern_sites(source, pattern, node_type)
    → 在所有源码行中按正则搜索 (CUDA 代码在 Python 三引号字符串内,
      Python 上下文过滤器会错误地将其排除, 此函数不受此限制)
  _replace_at_columns(source, site, replacement)
    → 按精确列位置替换 (Python/CUDA 通用, 解决同行多同名调用的精度问题)

ml_semantic.py — Python / Triton 路径 (C 类算子使用):
  _index_before_outside_string_comment(line, idx)
    → 判断位置是否在 Python 注释/字符串之外 (含三引号)
  _line_code_and_comment_start(line)
    → 分离代码和 # 注释部分
  _mutation_site_from_span(line_no, line, start, end, node_type)
    → 从位置信息构造 MutationSite, 自动过滤注释/字符串内的匹配
  _balanced_paren_span(s, open_idx)
    → 找到匹配的括号范围 (处理嵌套, Python/CUDA 通用)
  _first_call_arg(inner)
    → 提取函数调用的第一个参数 (处理嵌套逗号)

llm_pattern.py — D 类算子自用:
  _outside_comment_or_string(line, col)
    → 判断位置是否在 Python 注释/字符串之外 (含三引号 ''' 和 \"\"\")
    → 用于 D 类算子过滤 Python 层的 .expand()/.contiguous() 等匹配
```

**双路径设计的原因**：KernelBench 中 LLM 生成的 CUDA kernel 以三引号字符串形式
嵌入 Python 文件。Python 路径的 `_index_before_outside_string_comment` 会将三引号
内的 CUDA 代码视为"在字符串内"而跳过。CUDA 路径直接在每行上搜索，仅去除 `//` 注释。
每个 C 类算子的 `find_sites()` 同时执行两条路径，取并集。

### 3.8 数据流总结

```
源码字符串 (可能同时包含 Python 逻辑 + 三引号内的 CUDA C++ kernel)
  │
  ├── [A类] 双路径并行:
  │   ├── Python/Triton: ast.parse() → AST 遍历 → tokenize 定位运算符
  │   └── CUDA C++: 字符扫描 + _strip_cuda_comment + 二元判定 / 模板过滤
  │   合并去重 → MutationSite
  │
  ├── [B类] 正则搜索 → _is_in_comment_or_string 过滤 → MutationSite
  │         (已内置 Triton + CUDA 双路径: threadIdx/blockIdx 等)
  │
  ├── [C类] 双路径并行:
  │   ├── Python/Triton 路径:
  │   │     正则搜索 + _balanced_paren_span + _first_call_arg
  │   │     + _line_code_and_comment_start (# 注释过滤)
  │   │     → MutationSite 集合 A
  │   │
  │   └── CUDA C++ 路径:
  │         _cuda_find_pattern_sites(source, cuda_regex, node_type)
  │         + _strip_cuda_comment (// 注释过滤)
  │         → MutationSite 集合 B
  │
  │   合并: A ∪ B → 最终 MutationSite 列表
  │
  └── [D类] 正则搜索 + 括号平衡 + _outside_comment_or_string 过滤
            → MutationSite (仅作用于 Python 层: .expand(), .contiguous() 等)

MutationSite
  │
  └── apply(source, site)
        │
        ├── 根据 site.node_type 判断是 Python 匹配还是 CUDA 匹配
        │   (CUDA 匹配的 node_type 通常带 cuda_ 前缀或 CUDA 特定名称)
        │
        └── _replace_at_columns(source, site, replacement)
              → 按精确列位置替换，生成变异后的完整源码字符串
```
