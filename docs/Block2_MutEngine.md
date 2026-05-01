# Block 2: MutEngine — 变异测试执行引擎

> **对应文件**: `src/mutengine/mutant_runner.py`, `src/mutengine/equivalent_detector.py`, `src/mutengine/report.py`, `src/mutengine/parser/triton_parser.py`, `src/mutengine/parser/cuda_parser.py`, `src/bridge/eval_bridge.py`
>
> **论文位置**: Section 3.1（变异测试基本流程）+ Section 3.4（等价变异体处理）— 核心贡献 1 的执行基础设施

---

## 一、问题背景

### 1.1 GPU Kernel 变异测试没有现成基础设施

变异测试在通用代码领域已有成熟框架（PIT for Java、mutmut for Python），但 **GPU kernel 的变异测试面临独特的工程挑战**：

1. **编译链复杂**：CUDA kernel 需要 `nvcc` + `load_inline` 的 JIT 编译，Triton kernel 需要 `@triton.jit` 的即时编译。每个变异体都需要独立编译，且编译耗时远超 CPU 代码
2. **执行环境特殊**：必须在 GPU 上运行，需要管理 device memory、CUDA context 等资源
3. **输出比较的模糊 oracle**：GPU 浮点运算的非确定性使得 `==` 无法使用，必须用 `torch.allclose` + 容差判定
4. **等价变异体问题严重**：浮点精度的微小差异使得大量变异在统计上可能是等价的
5. **KernelBench 集成**：需要从 KernelBench 的特定目录结构中正确加载问题定义和已生成的 kernel

### 1.2 为什么不能用已有框架

| 已有框架 | 为什么不行 |
|---------|----------|
| PIT (Java) | 不支持 Python/CUDA/Triton |
| mutmut (Python) | 不支持 JIT 编译、GPU 执行、torch.allclose 比较 |
| MUTGPU 工具 | 不开源，且仅支持 CUDA SDK 程序，不支持 PyTorch 生态 |
| 通用 fuzzer | 面向崩溃检测，不面向数值正确性 |

**因此 MutEngine 必须从零构建。**

### 1.3 等价变异体的特殊挑战

在 ML kernel 的变异测试中，等价变异体的比例可能偏高：
- C6 `ReductionReorder` 改变 reduction 顺序，但浮点结合律差异极小
- C5 `CastRemove` 删除类型转换，但如果输入 dtype 与目标一致则无效
- 如果不处理等价变异体，mutation score 会被人为压低，影响论文结论的可信度

---

## 二、本块解决了什么问题

### 2.1 核心功能

MutEngine 提供了一套完整的 **GPU kernel 变异测试执行管线**：

```
KernelBench 数据 → 解析 → 变异 → 编译 → 执行 → 比较 → 判定 → 等价检测 → 报告
```

### 2.2 解决的具体问题

| 问题 | 解决方案 | 实现模块 |
|------|---------|---------|
| CUDA/Triton 解析 | 双语言解析器，自动检测语言类型 | `triton_parser.py`, `cuda_parser.py` |
| JIT 编译管理 | 临时文件 + `importlib.util` 动态导入 | `mutant_runner.py` |
| GPU 执行与对比 | `torch.allclose` + 多 seed 多次运行 | `mutant_runner.py` |
| 等价变异体识别 | 语法归一化 + 100 次统计比较 | `equivalent_detector.py` |
| KernelBench 集成 | 路径解析 + eval_results.json 读取 | `eval_bridge.py` |
| 结果报告 | JSON + Markdown 双格式汇总 | `report.py` |

### 2.3 为什么有意义

1. **这是第一个面向 LLM 生成 GPU kernel 的变异测试执行框架**，填补了工具链空白
2. **双语言支持**：同时处理 Triton（Python AST）和 CUDA（嵌入字符串正则）
3. **与 KernelBench 深度集成**：直接消费 KernelBench 的评测结果和生成的 kernel，无需人工转换
4. **统计等价检测**：100 次 bitwise 比较是对等价变异体问题的严谨处理，保证 mutation score 的可信度

---

## 三、实现流程

### 3.1 整体架构

```
                    ┌─────────────────┐
                    │  KernelBench    │
                    │  eval_bridge.py │
                    └────────┬────────┘
                             │ KernelInfo
                    ┌────────▼────────┐
                    │  Parser Layer   │
                    │  triton_parser  │
                    │  cuda_parser    │
                    └────────┬────────┘
                             │ source + AST
                    ┌────────▼────────┐
                    │  MutantRunner   │
                    │  (核心调度器)    │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        生成变异体      编译+执行       判定状态
     (调用 Operators)  (JIT compile)  (allclose)
                             │
                    ┌────────▼────────┐
                    │ EquivalentDetector│
                    │ (等价检测)       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  MutationReporter│
                    │  (结果报告)      │
                    └─────────────────┘
```

### 3.2 KernelBench 集成桥 (`eval_bridge.py`)

```
KernelBenchBridge 初始化:
  ├── root: KernelBench-0/ 根目录
  ├── problems_root: root/KernelBench/  (问题定义)
  └── run_name_template: "iter_full_l{level}_caesar_paper_v2" (生成结果)

数据加载流程:
  1. load_eval_results(level)
     → 读取 runs/{run_name}/eval_results.json
     → 返回 {problem_key: {correctness: bool, ...}}

  2. list_correct_kernels(level) / list_failed_kernels(level)
     → 过滤 correctness=True / False 的 kernel

  3. load_kernel_info(level, problem_key)
     │
     ├── find_generated_kernel(level, key)
     │   → 在 runs/{run_name}/ 下找 level_{L}_problem_{ID}_sample_0_kernel.py
     │
     ├── find_problem_file(level, key)
     │   → 在 KernelBench/level{L}/ 下找 {ID}_{name}.py
     │
     └── 构造 KernelInfo:
         ├── kernel_code: 生成的 kernel 源码
         ├── reference_module_path: 问题定义文件路径
         ├── language: "cuda" 或 "triton" (自动检测)
         └── problem_id, level, problem_name

  4. load_runtime_components(kernel)
     → 动态导入参考模块
     → 提取 get_inputs(), get_init_inputs() 函数
     → 返回 (ref_module, get_inputs_fn, get_init_inputs_fn)

语言检测: 源码中出现 __global__/__device__/load_inline/cuda_source ≥2 → CUDA
```

### 3.3 解析器层 (`parser/`)

#### TritonParser

```
parse(source):
  1. ast.parse(source) 解析完整 Python AST
  2. 遍历 AST 找 @triton.jit 装饰的函数
  3. 提取函数名、参数列表、行号范围
  4. 返回 TritonParseResult(kernels=[...], full_ast=...)

extract_mutatable_source(source):
  → 返回完整源码 + AST (因为 Triton kernel 就是 Python 代码)
```

#### CudaParser

```
parse(source):
  1. 正则搜索字符串字面量中的 CUDA 代码
     (匹配 三引号字符串 或 被赋值给含 "cuda"/"source" 变量名的字符串)
  2. 在每段 CUDA 字符串中正则搜索 __global__ void func_name(...)
  3. 提取 kernel 名称、参数、行号范围
  4. 返回 CudaParseResult(cuda_strings=[...], kernels=[...])

extract_mutatable_source(source):
  → 返回完整 Python 源码 + None (CUDA 代码用正则变异, 不用 AST)
```

### 3.4 变异体运行器 (`mutant_runner.py`) — 核心模块

#### 3.4.1 生成变异体

```
generate_mutants(kernel: KernelInfo) → List[Mutant]:
  │
  ├── [Step 1] ast.parse(source) 尝试解析 AST (失败则 tree=None)
  │
  ├── [Step 2] 根据 self.categories 获取目标算子列表
  │   └── get_operators_by_category("A"/"B"/"C"/"D")
  │
  └── [Step 3] 对每个算子调用 op.generate_mutants(source, kernel_id, tree)
      └── 内部: find_sites → apply → 构造 Mutant 对象
      └── 如果算子抛异常: 记录 warning, 跳过 (不中断整体流程)
```

#### 3.4.2 运行单个变异体

```
run_mutant(kernel, mutant, ref_module, get_inputs_fn, get_init_inputs_fn):
  │
  ├── [Step 1] 编译变异体
  │   └── _load_module_from_source(mutant.mutated_code, unique_name, tmp_dir)
  │       ├── **_patch_load_inline_names(source, unique_suffix)**:
  │       │     正则重写源码中所有 load_inline(name="xxx") → load_inline(name="xxx_<suffix>")
  │       │     确保每个变异体触发独立的 CUDA JIT 编译,
  │       │     避免复用原始 kernel 的编译缓存 (否则变异无效, kill rate 虚低)
  │       ├── 将变异后源码写入临时 .py 文件
  │       ├── importlib.util.spec_from_file_location() 创建模块 spec
  │       ├── spec.loader.exec_module() 执行模块 (触发 CUDA JIT 编译)
  │       └── 失败 → status=STILLBORN, 记录 error_message, 返回
  │
  ├── [Step 2] 实例化模型
  │   ├── 从 ref_module 取 Model 类, 从 mut_module 取 ModelNew 类
  │   │   (如果 ModelNew 不存在, 回退到 Model)
  │   ├── 调用 get_init_inputs_fn() 获取初始化参数
  │   └── 实例化: model = ModelNew(*init_inputs)
  │       └── 失败 → status=STILLBORN
  │
  ├── [Step 3] 多轮随机输入测试
  │   for trial in range(num_test_inputs):  # 默认 5 次
  │   │
  │   ├── torch.manual_seed(seed + trial) 设定随机种子
  │   ├── inputs = get_inputs_fn()
  │   ├── ref_output = _run_model(ref_model, inputs, device)
  │   │   └── model.to(device).eval(); model(*inputs)
  │   ├── torch.manual_seed(seed + trial) 重设种子 (保证输入一致)
  │   ├── mut_output = _run_model(mut_model, inputs, device)
  │   │
  │   └── _compare_outputs(ref_output, mut_output, atol, rtol):
  │       ├── Tensor: torch.allclose(ref.float().cpu(), mut.float().cpu(), atol, rtol)
  │       │   默认 atol=1e-2, rtol=1e-2 (对齐 KernelBench 原生验证容差)
  │       ├── Tuple/List: 递归逐元素比较
  │       └── 其他: == 比较
  │
  │       如果比较结果为 False → killed=True, 记录 kill_input_seed, break
  │       如果执行抛异常 → killed=True (运行时错误也算 killed)
  │
  └── [Step 4] 判定状态
      └── killed → KILLED, 否则 → SURVIVED
```

#### 3.4.3 批量运行

```
run_all_mutants(kernel, mutants, ref_module, get_inputs_fn, get_init_inputs_fn):
  │
  ├── 逐个调用 run_mutant() (串行, 因 GPU 资源竞争)
  ├── 逐个记录日志: operator_name, line, status, time
  │
  └── 构造 MutationTestResult:
      ├── kernel: KernelInfo
      ├── mutants: List[Mutant] (已更新 status)
      └── 自动计算: total, killed, survived, stillborn, mutation_score
```

### 3.5 等价变异体检测器 (`equivalent_detector.py`)

```
classify_survived_mutants(mutants, run_original_fn, run_mutant_fns, get_inputs_fn):
  │
  ├── 筛选: 仅处理 status=SURVIVED 的变异体
  │
  └── 对每个 survived mutant:
      │
      ├── [检查 1] 语法等价
      │   └── _normalize_source(): 去空行、去注释、去缩进
      │       if normalize(original) == normalize(mutated):
      │         → status=EQUIVALENT, reason="Syntactically equivalent"
      │
      └── [检查 2] 统计等价 (如果语法不等价)
          │
          └── check_statistical_equivalence():
              for i in range(100):  # 100 次不同种子
                seed = 10000 + i
                torch.manual_seed(seed)
                ref_output = run_original(get_inputs())
                torch.manual_seed(seed)
                mut_output = run_mutant(get_inputs())
                │
                if NOT _bitwise_identical(ref, mut):
                  → 不等价, return False (提前终止)
                │
              全部 100 次 bitwise identical:
                → status=EQUIVALENT, reason="Statistically equivalent (100 runs)"

_bitwise_identical():
  ├── Tensor: torch.equal(a, b)  # 严格 bitwise, 不是 allclose
  ├── Tuple/List: 递归逐元素
  └── 其他: ==
```

**关键设计决策**：
- 统计检测用 `torch.equal`（bitwise）而非 `torch.allclose`，因为等价变异体应该产生完全相同的输出
- 100 次是学术常用阈值，可配置
- 提前终止：一旦发现差异就不再继续

### 3.6 结果报告 (`report.py`)

```
MutationReporter:
  │
  ├── save_kernel_result(result: MutationTestResult)
  │   → 保存单个 kernel 的详细结果到 JSON
  │   └── 包含每个 mutant 的 operator, site, status, error_message, time
  │   └── SURVIVED 变异体自动附带 mutated_code (供 StressEnhance 直接读取)
  │
  ├── generate_summary(results: List[MutationTestResult])
  │   → 汇总所有 kernel 的结果:
  │     ├── 总体: total, killed, survived, stillborn, equivalent, score
  │     ├── 按类别: A/B/C/D 各自的 score
  │     └── 按算子: 每个具体算子的 score
  │
  ├── save_summary(results)
  │   → 输出 JSON 和 Markdown 双格式
  │   └── Markdown 包含表格，可直接嵌入论文
  │
  └── save_stress_report(stress_summary: dict)    ← 新增
      → 输出 StressEnhance 实验报告:
        ├── 可见率对比: native vs stress-enhanced + visibility lift
        ├── 归因分布表: Type 1-6 各计数 (含 Non-deterministic 和 Infra Defect)
        └── 每个 stress policy / mode 的 kill 统计
```

**关键数据模型更新**：

`Mutant.to_dict()` 新增 `include_code` 参数。当变异体状态为 SURVIVED 时，
`MutationTestResult.to_dict()` 自动设置 `include_code=True`，在 JSON 中保存
`mutated_code` 字段。这使得 StressEnhance 实验可以直接从 JSON 读取变异代码，
无需重新生成。`Mutant.from_dict()` 也对应更新，能读回 `mutated_code`。

### 3.7 关键数据流

```
KernelBench-0/
  ├── KernelBench/level1/1_Square_matrix_multiplication_.py  (问题定义)
  └── runs/.../level_1_problem_1_sample_0_kernel.py          (LLM 生成的 kernel)
        │
        ▼
[eval_bridge] load_kernel_info() + load_runtime_components()
        │
        ├── KernelInfo(code, ref_path, language, ...)
        ├── ref_module (含 Model, get_inputs, get_init_inputs)
        │
        ▼
[mutant_runner] generate_mutants()
        │
        ├── [A] ArithReplace, RelOpReplace, ConstPerturb
        ├── [B] IndexReplace, SyncRemove, MaskBoundary, LaunchConfigMutate
        ├── [C] StabRemove, AccDowngrade, EpsilonModify, ScaleModify, ...
        └── [D] BroadcastUnsafe, LayoutAssume
        │
        ▼  List[Mutant]
[mutant_runner] run_all_mutants()
        │
        ├── 编译 (JIT)         → STILLBORN if fail
        ├── 执行 (5x random)   → KILLED if allclose fails
        └── 比较 (allclose)     → SURVIVED if all pass
        │
        ▼  List[Mutant] (with status)
[equivalent_detector] classify_survived_mutants()
        │
        ├── 语法归一化 → EQUIVALENT
        └── 100x bitwise → EQUIVALENT
        │
        ▼  List[Mutant] (final status)
[report] generate_summary()
        │
        └── JSON + Markdown 报告
            ├── Overall: mutation_score = killed / (total - stillborn - equivalent)
            ├── Per-Category: A=?%, B=?%, C=?%, D=?%
            └── Per-Operator: arith_replace=?%, stab_remove=?%, ...
```
