# Block 3: RealismGuard — 变异体现实性验证

> **对应文件**: `src/mutengine/realism_validator.py`, `scripts/validate_realism.py`
>
> **论文位置**: Section 3.3（Mutant Realism 验证）— 构造效度的关键证据

---

## 一、问题背景

### 1.1 变异测试最核心的质疑：你的变异体是真的吗？

变异测试的根本假设是 **coupling effect**：如果测试能检出小的人为缺陷（变异体），就能检出真实的大缺陷。但这个假设的前提是——**变异体必须代表真实的缺陷模式**。

对于 MutaKernel 而言，这个质疑尤其尖锐：

> **"你的 C 类 ML 数值语义算子（StabRemove, AccDowngrade, EpsilonModify 等）是你人为设计的。你怎么知道 LLM 真的会犯这些错？如果 LLM 从不犯这类错，那你的 mutation score 就是在度量一个不存在的问题。"**

这是 CCF-A 审稿人必然会问的 **construct validity** 问题。如果回答不好，论文直接被拒。

### 1.2 为什么不能"想当然"

以下论证是不够的：
- "这些是常见的数值编程错误" — 常见不代表 LLM 会犯
- "我们觉得 LLM 可能会遗漏 max-subtraction" — 主观猜测
- "相关工作提到过这类问题" — 需要针对 LLM 生成 kernel 的具体证据

### 1.3 我们有什么数据

KernelBench 的现有数据提供了两类验证素材：

**数据源 1（主力）— 迭代历史配对**：

正确 kernel 是 LLM 经过最多 10 轮迭代修改得到的。迭代过程中错对交叉出现：

```
Problem 5 的迭代历史（示例）:
  Turn 0: 编译失败                          → 跳过
  Turn 1: 编译成功, 运行错误 (correctness=false) → ★ 可用样本
  Turn 2: 编译成功, 运行正确 (speedup=0.8)       → 候选正确版本
  Turn 3: 编译成功, 运行错误 (correctness=false) → ★ 可用样本
  Turn 4: 编译失败                          → 跳过
  Turn 5: 编译成功, 运行正确 (speedup=1.3)       → ★ 最佳正确版本
  ...
```

筛选规则：
- **"对"选最佳**：correctness=True 且 speedup 最高的 turn
- **"错"筛编译通过的**：排除编译失败（无 kernel 文件），只保留编译通过但运行错误的
- **每个"错"都和最佳"对"配对** → 一个问题可产出多个高质量 diff 样本

**数据源 2（补充）— 最终失败 kernel**：

这些 kernel 没有同问题的正确版本（同一 run 中每个问题只有一个最终结果），大部分只能做 standalone 分析（质量较低）。具体数量随实验版本变化,在实验章节报告。

---

## 二、本块解决了什么问题

### 2.1 核心目标

RealismGuard 回答一个关键问题：

> **"C/D 类变异算子覆盖了多少比例的真实 LLM 错误？"**

这个数字（Realism Coverage Rate）直接决定了论文的构造效度是否成立。

### 2.2 解决方案

采用两种数据源互相补充：

| 数据源 | 做法 | 优势 |
|--------|------|------|
| **迭代历史配对**（主力） | 同问题的失败 turn vs 最佳正确 turn → diff 分析 | diff 精确，来自同一 LLM 的真实迭代 |
| **最终失败 kernel**（补充） | standalone 代码分析 + 关键词匹配 | 覆盖没有正确版本的问题 |

### 2.3 为什么有意义

1. **CCF-A 审稿标准要求**：Construct validity 是实证研究论文的必答项。没有 RealismGuard，论文的贡献 1（MutEngine + 变异算子）的所有结论都是可质疑的
2. **数据驱动而非人为论证**：我们不是"声称"算子有用，而是用真实 LLM 迭代错误样本进行定量验证
3. **如果覆盖率低，我们会调整算子**：这不是走过场——如果某个算子完全没有真实错误对应 → 考虑降级或移除

### 2.4 做到了什么别人做不到的事情

- MUTGPU (Salman et al., ICST 2020)：未做针对变异算子的 realism validation
- 通用变异工具：依赖 coupling effect 的一般性论证，不做领域特定验证
- **MutaKernel**：针对 LLM 生成 GPU kernel 这一特定场景，用真实迭代错误样本量化验证

---

## 三、实现流程

### 3.1 核心数据结构

```
BugPattern (realism_validator.py):
  ├── bug_id: str           # 唯一标识 (如 "L1_P5_iter_t1")
  ├── problem_id: int       # KernelBench 问题编号
  ├── level: int            # KernelBench 难度级别
  ├── root_cause: str       # 错误根因分类
  ├── error_category: str   # 高层错误类别 (numerical_semantic / structural / logic / unknown)
  ├── diff_summary: str     # diff 摘要 (截取前 500 字符)
  ├── matched_operators: List[str]  # 匹配到的变异算子列表
  └── source: str           # 数据来源 ("iteration_diff" / "final_fail_diff" / "standalone")

IterationPair (定义在 src/bridge/eval_bridge.py, 由 KernelBenchBridge 提供):
  ├── problem_id: int
  ├── level: int
  ├── failed_turn: int          # 失败 turn 编号
  ├── best_correct_turn: int    # 最佳正确 turn 编号
  ├── failed_kernel_path: str   # 失败 kernel 文件路径
  ├── correct_kernel_path: str  # 正确 kernel 文件路径
  └── best_speedup: float       # 最佳正确版本的加速比
  注: 此数据结构不属于 RealismValidator, 而是由外部 Bridge 层提供

RealismReport (dataclass, realism_validator.py):
  ├── total_bugs_analyzed: int
  ├── total_root_causes: int        # 去重后的根因种类数
  ├── bugs_covered_by_cd: int       # C/D 类算子覆盖的 bug 数
  ├── bugs_covered_by_ab: int       # A/B 类覆盖的
  ├── bugs_not_covered: int         # 未覆盖的
  ├── coverage_rate_cd: float       # 核心指标: C/D 覆盖率
  ├── coverage_rate_all: float      # 全部算子覆盖率
  ├── per_operator_realism: Dict    # 每个算子命中多少真实 bug
  ├── per_category_coverage: Dict   # 按高层类别的覆盖统计
  ├── uncovered_patterns: List[str] # 未被覆盖的根因列表
  └── bugs: List[BugPattern]        # 完整 bug 列表 (用于详细分析)

  注: per_source_count 不是 dataclass 字段, 而是在 to_dict() 中
  从 self.bugs 动态计算。to_dict() 还输出 bugs_sample (前 80 条)。
```

### 3.2 根因分类体系

实现了一个两级映射，共 20 种具体根因 + 1 个 unknown。

**Level 1: diff 变更行关键词 → 错误根因**（23 条领域正则 + 3 条兜底泛模式，按 dict 顺序定义优先级）

以下为 `DIFF_CHANGED_LINE_PATTERNS` 字典的完整 23+1 条正则（按 dict 顺序 = 匹配优先级）:

```
#  正则                                                              → 根因
 1  [\w\]\)]\s*-\s*max\w* | [\w\]\)]\s*-\s*row_m | …-\s*m_i        → missing_numerical_stability
 2  \.max\s*\( | tl\.max\s*\(                                        → overflow_no_max_subtract
 3  (?:accum|dot).*(?:float16|fp16|half) | (反向)                     → precision_loss_fp16_accumulator
 4  float\s*\(\s*['"](-)?inf                                         → wrong_init_value
 5  -\s*INFINITY | FLT_MAX | -\s*FLT_MAX                             → wrong_init_value
 6  tl\.sum\s*\( | torch\.sum\s*\( | \.sum\s*\(                      → reduction_precision
 7  \.float\(\)                                                       → missing_fp32_cast
 8  (?:torch\.)?float32 | (?:tl\.)?float32 | \bfp32\b                → missing_fp32_cast
 9  \beps\b | epsilon | 1e-[5-8]                                      → epsilon_missing
10  1\.0\s*/\s*(?:math\.)?sqrt | \/\s*(?:math\.)?sqrt                → scale_factor_missing
11  \bscale\s*=                                                       → scale_factor_wrong
12  \brsqrt\w* | (?<!\w)sqrt\w*                                       → scale_factor_missing
13  \.half\(\) | \.bfloat16\(\)                                       → missing_type_cast
14  \.to\s*\(\s*(?:torch\.)?(?:float16|bfloat16|half)\s*\)           → missing_type_cast
15  \.to\s*\(\s*(?:torch\.)?(?:float32)\s*\)                         → missing_fp32_cast
16  \.to\s*\(                                                         → missing_type_cast
17  (?:int|long)\s*\(.*(?:float|double)\)                             → implicit_type_coercion
18  (?:float|double)\s*\(.*(?:int|long)\)                             → implicit_type_coercion
19  \bfloat\s*\(\s*\w+\s*\).*\bfloat\s*\(\s*\w+\s*\)               → implicit_type_coercion
20  \.expand\s*\( | \.expand_as\s*\( | \.broadcast                   → missing_broadcast
21  \.view\s*\( | \.reshape\s*\(                                      → shape_mismatch_no_expand
22  \.contiguous\s*\(                                                  → contiguous_assumption
23  program_id\s*\(\s*\d\s*\) | threadIdx\.\w | blockIdx\.\w | blockDim\.\w → wrong_index_dimension
24  (?:off(?:set)?|idx|index)\s*<= | <=\s*(?:N\b|n\b|size|length)   → off_by_one_boundary
```

**补充 — 兜底泛模式** (非正则匹配, 而是结构差异分析, 详见 §3.5):

```
Round 4a: 替换所有比较运算符为 "CMP", 行对结构相同 → wrong_comparison_op
Round 4b: 替换所有数字为 "NUM", 行对结构相同且值不同 → wrong_constant
Round 4c: 替换所有算术运算符为 "OP", 行对结构相同  → wrong_arithmetic_op
```

注: 兜底模式仅在 `_classify_root_cause` (有 diff) 中使用,
`_classify_root_cause_from_code` (standalone) 中**不使用**兜底。

**Level 2: 错误根因 → 变异算子**

```python
ROOT_CAUSE_TO_OPERATORS = {
    "missing_numerical_stability":      ["stab_remove"],
    "overflow_no_max_subtract":         ["stab_remove"],
    "precision_loss_fp16_accumulator":  ["acc_downgrade"],
    "missing_fp32_cast":                ["acc_downgrade", "cast_remove"],
    "epsilon_missing":                  ["epsilon_modify"],
    "epsilon_wrong_value":              ["epsilon_modify"],
    "scale_factor_missing":             ["scale_modify"],
    "scale_factor_wrong":               ["scale_modify"],
    "missing_type_cast":                ["cast_remove"],
    "implicit_type_coercion":           ["cast_remove"],
    "reduction_precision":              ["reduction_reorder", "acc_downgrade"],
    "wrong_init_value":                 ["init_modify"],
    "missing_broadcast":                ["broadcast_unsafe"],
    "shape_mismatch_no_expand":         ["broadcast_unsafe"],
    "contiguous_assumption":            ["layout_assume"],
    "wrong_index_dimension":            ["index_replace"],
    "off_by_one_boundary":              ["mask_boundary"],
    "wrong_arithmetic_op":              ["arith_replace"],
    "wrong_comparison_op":              ["relop_replace"],
    "wrong_constant":                   ["const_perturb"],
}
```

**Level 3: 高层分类（4 类）**

```
_categorize_error(root_cause) 的映射:

numerical_semantic (12 种):
  missing_numerical_stability, overflow_no_max_subtract,
  precision_loss_fp16_accumulator, missing_fp32_cast,
  epsilon_missing, epsilon_wrong_value,
  scale_factor_missing, scale_factor_wrong,
  missing_type_cast, implicit_type_coercion,
  reduction_precision, wrong_init_value

structural (3 种):
  missing_broadcast, shape_mismatch_no_expand, contiguous_assumption

logic (5 种):
  wrong_index_dimension, off_by_one_boundary,
  wrong_arithmetic_op, wrong_comparison_op, wrong_constant

unknown:
  上述均不匹配的根因
```

### 3.3 数据源 1: 迭代历史配对（主力）

```
list_iteration_pairs(level):     [由 KernelBenchBridge 实现]
  │
  ├── [Step 1] 扫描 iterations/ 目录下每个问题
  │   └── 读取 problem_summary.json 中的 turns 列表
  │
  ├── [Step 2] 选最佳正确 turn
  │   └── correctness=True 且 speedup 最高的 turn
  │       （文件必须存在，否则跳过）
  │
  ├── [Step 3] 筛选可用的失败 turn
  │   └── correctness=False 且 kernel 文件存在（= 编译通过）
  │       编译失败的 turn（无文件）直接跳过
  │
  └── [Step 4] 配对
      每个失败 turn 都和最佳正确 turn 形成一对 IterationPair

对每个 IterationPair (在 validate_realism.py 中执行):
  ├── 读取 failed_kernel 和 correct_kernel 的源码
  ├── 跳过源码完全相同的配对 (strip 后比较)
  ├── analyze_bug_from_diff() → BugPattern (source="iteration_diff")
  └── 记录到 validator.bugs 列表
```

### 3.4 数据源 2: 最终失败 kernel（补充）

```
对每个 level (在 validate_realism.py 中执行):
  │
  ├── 从 eval_results 获取 failed/correct 列表
  │   (通过 KernelBenchBridge.list_failed_kernels / list_correct_kernels)
  │
  └── 对每个 failed kernel:
      │
      ├── 如果同问题有正确 kernel (极少):
      │   → analyze_bug_from_diff() (source="final_fail_diff")
      │
      └── 如果没有正确 kernel (绝大部分):
          → analyze_buggy_kernel_standalone() (source="standalone")
          仅对 buggy_code + error_message 文本使用 23 条正则搜索,
          不使用兜底泛模式 (无 diff 行对), 质量较低
```

### 3.5 diff 分析：主匹配 + 兜底泛模式

`_classify_root_cause(diff_text, correct_code, buggy_code)` 的完整流程:

```
_classify_root_cause():
  │
  ├── [准备] 解析 unified_diff(correct, buggy) 输出:
  │   ├── removed_lines (−行): 正确版本有、错误版本没有的代码
  │   │   (= LLM 遗漏的关键代码, 如忘记加 x - max(x))
  │   │   提取: l[1:] for l startswith "-" and not "---"
  │   └── added_lines (+行): 错误版本有、正确版本没有的代码
  │       (= LLM 错误引入的代码)
  │       提取: l[1:] for l startswith "+" and not "+++"
  │   注: unified_diff(correct, buggy) 中 − 行来自 correct, + 行来自 buggy
  │
  ├── [主匹配] 按 pattern 优先级遍历 23 条领域正则
  │   all_changed_lines ← removed_lines + added_lines  (合并)
  │   for pattern in DIFF_CHANGED_LINE_PATTERNS (按 dict 顺序):
  │     for line in all_changed_lines:
  │       if re.search(pattern, line, IGNORECASE): return cause
  │   → 高优先级 pattern (如数值稳定化) 总是先于低优先级 pattern
  │     无论它出现在 diff 的哪一行
  │
  ├── [兜底 Round 4a] 比较运算符变更检测
  │   对 removed_lines × added_lines 的所有行对:
  │     将 <=, >=, ==, !=, <, > 统一替换为 "CMP"
  │     若替换后两行结构相同且含 "CMP" → return "wrong_comparison_op"
  │
  ├── [兜底 Round 4b] 常量值变更检测
  │   对 removed_lines × added_lines 的所有行对:
  │     将所有数字 (\d+\.?\d*) 统一替换为 "NUM"
  │     若替换后结构相同且含 "NUM", 且数字值不同 → return "wrong_constant"
  │
  ├── [兜底 Round 4c] 算术运算符变更检测
  │   对 removed_lines × added_lines 的所有行对:
  │     将 +, -, *, / 统一替换为 "OP"
  │     若替换后结构相同且含 "OP" → return "wrong_arithmetic_op"
  │   若所有变更行中包含任何算术运算符 → return "wrong_arithmetic_op"
  │
  └── 全部不匹配 → return "unknown"
```

**设计要点**: 兜底泛模式 (Round 4a-4c) 确保即使领域正则没有命中,
仍能通过**结构差异分析**捕捉到算术/比较/常量类错误。
这些 bug 被映射到 A 类通用算子 (arith_replace / relop_replace / const_perturb),
虽然覆盖率计入 "A/B 覆盖" 而非 "C/D 覆盖", 但减少了 "unknown" 类别的比例。

**对 standalone 分析的补充**: 当没有正确版本可供 diff 时,
`_classify_root_cause_from_code(text)` 直接对 buggy_code + error_message 文本
按同一组 23 条正则搜索 (即 `DIFF_KEYWORDS_TO_ROOT_CAUSE`, 别名指向同一 dict),
首次命中即返回。此函数**不使用兜底泛模式**
(因为无 diff 行对可做结构对比), 因此准确率较低, 在论文中需注明。

### 3.6 报告生成

```
generate_report() → RealismReport:
  │
  ├── [Step 0] 构建算子集合 (用于区分 C/D 类和 A/B 类)
  │   cd_operators: 从 ROOT_CAUSE_TO_OPERATORS 的所有算子中,
  │     通过前缀匹配自动收集 —— 前缀 ∈ {stab_, acc_, epsilon_,
  │     scale_, cast_, reduction_, init_, broadcast_, layout_}
  │     → 覆盖全部 C 类 (7 个) + D 类 (2 个) 算子
  │   ab_operators: 硬编码 7 个名称 —— {arith_replace, relop_replace,
  │     const_perturb, index_replace, sync_remove, mask_boundary,
  │     launch_config_mutate}
  │   注: 若新增算子, 需确保其前缀属于上述列表或更新硬编码集合
  │
  ├── [Step 1] 统计每个 bug 的覆盖情况
  │   for bug in self.bugs:
  │     ops_set = set(bug.matched_operators)
  │     ├── 无匹配算子 → not_covered += 1
  │     ├── ops_set ∩ cd_operators ≠ ∅ → covered_cd += 1
  │     ├── 仅 ops_set ∩ ab_operators ≠ ∅ → covered_ab += 1
  │     └── 有匹配但既不在 cd 也不在 ab → not_covered += 1
  │
  ├── [Step 2] 统计每个算子的命中频次
  │   per_operator_realism = {op_name: count}
  │
  ├── [Step 3] 计算覆盖率
  │   coverage_rate_cd = covered_cd / total
  │   coverage_rate_all = (covered_cd + covered_ab) / total
  │
  └── [Step 4] 收集未覆盖模式
      uncovered_patterns = [bug.root_cause for bug if no matched ops]

  注: per_source_count 在 to_dict() 中动态计算,
  不是 generate_report 的逻辑:
    source_counts = Counter(b.source for b in self.bugs)
```

### 3.7 实验脚本用法

```bash
# 完整运行（迭代历史 + 最终失败，推荐）
python scripts/validate_realism.py --levels 1 2

# 仅用最终失败数据（兼容旧流程）
python scripts/validate_realism.py --levels 1 2 --no-iterations

# 仅 L1
python scripts/validate_realism.py --levels 1
```

### 3.8 对论文的直接支撑

RealismGuard 的输出直接支撑论文中三处：

| 论文位置 | 用什么数据 | 说什么 |
|---------|----------|--------|
| Section 3.3 | coverage_rate_cd | "C/D 类算子覆盖了 X% 的真实 LLM 数值语义错误" |
| RQ1 讨论 | per_operator_realism | "cast_remove 最频繁，对应 Y 个真实错误" |
| Threats to Validity | uncovered_patterns | "Z% 的真实错误未被覆盖，主要是 xxx 类型" |

### 3.9 方法论局限与论文注意事项

RealismGuard 的验证结果为算子设计的反馈依据（研究者手动分析决策）：

```
如果某个 C/D 算子的 per_operator_realism = 0:
  → 该算子没有真实错误对应
  → 考虑: (a) 降级为辅助算子, (b) 从核心创新中移除, (c) 补充更多数据源

如果 uncovered_patterns 中出现高频新模式:
  → 可能需要设计新的变异算子
```

**论文撰写时需注意的 Threats to Validity**:

1. **diff 算法选择**: 代码使用 `difflib.unified_diff` 进行行级 diff。对于变量重命名或结构重构等语义等价但文本不同的修改，行级 diff 可能产生噪声。更精细的 token-level diff 可能提高准确率,但在当前数据规模下行级 diff 已足够。

2. **正则优先级依赖**: 23 条正则按 Python 3.7+ dict 插入顺序排列,首次命中即返回。这意味着高优先级模式 (如数值稳定化) 会"遮蔽"低优先级模式 (如通用类型转换)。若某个 bug 同时命中多条正则,仅返回最高优先级的根因。

3. **standalone 分析质量**: 无正确版本对照时,`_classify_root_cause_from_code` 直接对 buggy code 做关键词搜索,不使用兜底泛模式。**结果**: 此路径的分类准确率显著低于 diff 路径,建议在论文中分别报告两种数据源的覆盖率。

4. **"unknown" 比例的解读**: 高 "unknown" 率可能源于: (a) 正则覆盖范围不足, (b) 真实错误属于全新类型, (c) diff 过于复杂导致关键词被稀释。论文应对 uncovered_patterns 做定性分析。

---

## 四、与其他 Block 的关系

```
Block 1（算子清单）──→ Block 3 用这份清单做匹配
                        │
Block 3 的报告 ──→ 研究者判断后手动调整 Block 1
                        │
Block 3 与 Block 2 无运行时依赖（操作不同的数据）
                        │
Block 3 的 per_operator_realism ──→ 为 Block 5 (StressEnhance) 的结论提供构造效度支撑
```

**Block 3 不依赖 Block 2 的运行结果。** Block 2 操作正确 kernel（种 bug），Block 3 操作失败 kernel（分析真实错误）。两者数据源完全独立。

**与 StressEnhance 的关系**：StressEnhance 的"属性可见性"叙事成立的前提是 C/D 类变异体对应真实 LLM 错误。如果某个 C 类算子在 RealismGuard 中没有真实错误对应，那么该算子在 StressEnhance 中的 visibility lift 结论也不可信。因此 Block 3 是 Block 5 结论可信度的关键防线。
