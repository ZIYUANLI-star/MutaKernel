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

这是 ICSE 审稿人必然会问的 **construct validity** 问题。如果回答不好，论文直接被拒。

### 1.2 为什么不能"想当然"

以下论证是不够的：
- ❌ "这些是常见的数值编程错误" — 常见不代表 LLM 会犯
- ❌ "我们觉得 LLM 可能会遗漏 max-subtraction" — 主观猜测
- ❌ "相关工作提到过这类问题" — 需要针对 LLM 生成 kernel 的具体证据

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

| 数据 | 数量 | 含义 |
|------|------|------|
| L1 失败 kernel | 36 | LLM 最终未通过 `torch.allclose` 的 kernel |
| L2 失败 kernel | 66 | 同上 |

这些 kernel 没有同问题的正确版本（同一 run 中每个问题只有一个最终结果），大部分只能做 standalone 分析（质量较低）。

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

1. **ICSE 审稿标准要求**：Construct validity 是实证研究论文的必答项。没有 RealismGuard，论文的贡献 1（MutEngine + 变异算子）的所有结论都是可质疑的
2. **数据驱动而非人为论证**：我们不是"声称"算子有用，而是用真实 LLM 迭代错误样本进行定量验证
3. **如果覆盖率低，我们会调整算子**：这不是走过场——如果某个算子完全没有真实错误对应 → 考虑降级或移除

### 2.4 做到了什么别人做不到的事情

- MUTGPU：没做 realism validation，直接声称算子有效
- 通用变异工具：依赖 coupling effect 的一般性论证，不做领域特定验证
- **MutaKernel**：针对 LLM 生成 GPU kernel 这一特定场景，用真实迭代错误样本量化验证

---

## 三、实现流程

### 3.1 核心数据结构

```
BugPattern:
  ├── bug_id: str           # 唯一标识 (如 "L1_P5_iter_t1")
  ├── problem_id: int       # KernelBench 问题编号
  ├── level: int            # KernelBench 难度级别
  ├── root_cause: str       # 错误根因分类
  ├── error_category: str   # 高层错误类别 (numerical_semantic / structural / logic / unknown)
  ├── diff_summary: str     # diff 摘要
  ├── matched_operators: List[str]  # 匹配到的变异算子列表
  └── source: str           # 数据来源 ("iteration_diff" / "final_fail_diff" / "standalone")

IterationPair (eval_bridge 提供):
  ├── problem_id: int
  ├── level: int
  ├── failed_turn: int          # 失败 turn 编号
  ├── best_correct_turn: int    # 最佳正确 turn 编号
  ├── failed_kernel_path: str   # 失败 kernel 文件路径
  ├── correct_kernel_path: str  # 正确 kernel 文件路径
  └── best_speedup: float       # 最佳正确版本的加速比

RealismReport:
  ├── total_bugs_analyzed: int
  ├── bugs_covered_by_cd: int       # C/D 类算子覆盖的 bug 数
  ├── bugs_covered_by_ab: int       # A/B 类覆盖的
  ├── bugs_not_covered: int         # 未覆盖的
  ├── coverage_rate_cd: float       # 核心指标: C/D 覆盖率
  ├── coverage_rate_all: float      # 全部算子覆盖率
  ├── per_operator_realism: Dict    # 每个算子命中多少真实 bug
  ├── per_source_count: Dict        # 每种数据源贡献的样本数
  └── uncovered_patterns: List[str] # 未被覆盖的错误类型
```

### 3.2 根因分类体系

实现了一个两级映射，共 20 种具体根因 + 1 个 unknown。

**Level 1: diff 关键词 → 错误根因**（30 条正则规则，按优先级排序）

| 正则范围 | 覆盖的根因 |
|---------|-----------|
| `max.*sub`, `.max(` | missing_numerical_stability, overflow_no_max_subtract |
| `accum.*half`, `float16.*sum` | precision_loss_fp16_accumulator |
| `float32`, `fp32`, `.float()` | missing_fp32_cast |
| `1e-[5-8]`, `eps`, `epsilon` | epsilon_missing, epsilon_wrong_value |
| `sqrt`, `rsqrt`, `scale` | scale_factor_missing, scale_factor_wrong |
| `.to(`, `.half()`, `.bfloat16()` | missing_type_cast |
| `int(.*float)`, `float(.*int)` | implicit_type_coercion |
| `tl.sum(`, `torch.sum(`, `.sum(` | reduction_precision |
| `.contiguous()` | contiguous_assumption |
| `.expand(`, `.broadcast` | missing_broadcast |
| `.view(`, `.reshape(` | shape_mismatch_no_expand |
| `float('-inf')`, `-inf`, `INFINITY` | wrong_init_value |
| `program_id`, `threadIdx`, `blockIdx` | wrong_index_dimension |
| `off.*<=`, `idx<=N` | off_by_one_boundary |
| diff 变更行中的运算符/比较符/常量 | wrong_arithmetic_op, wrong_comparison_op, wrong_constant |

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

| 高层类别 | 包含的根因 |
|---------|-----------|
| numerical_semantic | 12 种（精度、稳定性、类型转换等） |
| structural | 3 种（broadcast、shape、contiguous） |
| logic | 5 种（索引、边界、算术、比较、常量） |
| unknown | 无法识别 |

### 3.3 数据源 1: 迭代历史配对（主力）

```
list_iteration_pairs(level):
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

对每个 IterationPair:
  ├── 读取 failed_kernel 和 correct_kernel 的源码
  ├── analyze_bug_from_diff() → BugPattern (source="iteration_diff")
  └── 记录到 validator.bugs 列表
```

### 3.4 数据源 2: 最终失败 kernel（补充）

```
对每个 level:
  │
  ├── 从 eval_results.json 获取 failed/correct 列表
  │
  └── 对每个 failed kernel:
      │
      ├── 如果同问题有正确 kernel (极少):
      │   → analyze_bug_from_diff() (source="final_fail_diff")
      │
      └── 如果没有正确 kernel (绝大部分):
          → analyze_buggy_kernel_standalone() (source="standalone")
          （仅在代码中搜关键词，质量较低）
```

### 3.5 diff 分析：三轮匹配

```
_classify_root_cause(diff_text, correct_code, buggy_code):
  │
  ├── 第一轮: 在完整 diff 文本中搜索关键词
  │   └── 按优先级遍历 30 条正则，首次命中即返回
  │
  ├── 第二轮: 仅在"正确版本有、错误版本没有"的行（+行）中搜索
  │   └── 这些是 LLM 遗漏的关键代码（如忘记加 -max）
  │
  ├── 第三轮: 仅在"错误版本有、正确版本没有"的行（-行）中搜索
  │   └── 这些是 LLM 错误引入的代码
  │
  └── 都不匹配 → "unknown"
```

### 3.6 报告生成

```
generate_report() → RealismReport:
  │
  ├── [Step 1] 统计每个 bug 的覆盖情况
  │   for bug in self.bugs:
  │     ops_set = set(bug.matched_operators)
  │     ├── 无匹配算子 → not_covered += 1
  │     ├── 匹配 C/D 类算子 → covered_cd += 1
  │     └── 仅匹配 A/B 类算子 → covered_ab += 1
  │
  ├── [Step 2] 统计每个算子的命中频次
  │   per_operator_realism = {op_name: count}
  │
  ├── [Step 3] 按数据源分组统计
  │   per_source_count = {"iteration_diff": N1, "standalone": N2, ...}
  │
  ├── [Step 4] 计算覆盖率
  │   coverage_rate_cd = covered_cd / total
  │   coverage_rate_all = (covered_cd + covered_ab) / total
  │
  └── [Step 5] 收集未覆盖模式
      uncovered_patterns = [bug.root_cause for bug if no matched ops]
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

### 3.9 自我修正机制

RealismGuard 不仅是验证工具，还是**算子设计的反馈信号**（研究者手动执行）：

```
如果某个 C/D 算子的 per_operator_realism = 0:
  → 该算子没有真实错误对应
  → 考虑: (a) 降级为辅助算子, (b) 从核心创新中移除, (c) 补充更多数据源

如果 uncovered_patterns 中出现高频新模式:
  → 可能需要设计新的变异算子
  → 这可作为 MutEvolve (experimental, src/mutevolve/) 的输入
```

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
