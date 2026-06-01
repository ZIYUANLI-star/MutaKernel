# 第一次实验（旧框架）归档说明

> 归档日期：2026-06-01　　原目录名：`第一次实验汇总`
>
> 背景：项目早期框架为《Mutation-Guided Diagnosis and Augmentation》（目标 ICSE 2027），
> 后重构为最终投稿框架《Validating the Validators》（EuroSys 投稿 #303）。
> 本目录是旧框架那一次完整实验的产物，经逐数据核对后归档。

## 一、已被第二次实验取代（`已被第二次取代/`）

经与 `第二次实验汇总/full_block12_results/summary.json` 逐项比对：

| 维度 | 第一次（本目录，旧） | 第二次（最终，论文采用） |
|---|---|---|
| kernel 数 | 90（与第二次为**同一批**） | 90 |
| 变异体总数 | 1663 | 1646 |
| killed | 944 | 939 |
| mutation score（保守） | 0.7457 | 0.6375 |
| 等价分类 | `equivalent=234`（未细分） | `strict=10 + candidate=264`（EMD 拆分） |

差异来源：第二次在方法学定稿后重跑，EMD 把等价拆成 strict/candidate、C 类变异采样微调（195→178）。
**论文所有数字以第二次为准**，故本目录下的 `full_block12_results/`、`stress_enhance_results/`、
`第一次实验docs/` 仅作历史留档。

## 二、第一次独有、必须保留（`独有保留_LLM归因与测试构造规则/`）

以下产物由旧框架的 DeepSeek-R1 LLM 归因阶段生成，**第二次实验与论文均未复现**，属独有资产：

- `llm_analysis_results/test_construction_rules.json` —— **31 条**测试用例构造规则，
  每条含 `rule_name / description / applicable_operators / 可执行 policy_code / mutant_id`。
  论文正文未写，但实际确做了"从被杀变异体提炼测试构造规则"这一步，仅此一份。
- `llm_analysis_results/robustness_suggestions.json` —— 216 条 kernel 鲁棒性修复建议（旧框架 RQ4）。
- `llm_analysis_results/taxonomy.json` —— 7 类存活原因数据驱动聚类
  （Ineffective Mutation / Kernel Design Resilience / Numerical Tolerance / Race Condition /
  Out-of-Bounds / Algorithmic Invariance / Test Framework Limitations）。
  概念上被论文的 5 类 reason_category（来自 Opus 审计）取代，但作为独立产物保留。
- `llm_analysis_results/details/`、`prompts/`、`llm_*_report.md` —— 上述结论的明细与可读报告。

## 三、注意

`scripts/` 下有约 9 个调试脚本（`recheck_equiv*.py`、`_show_mutant_diff*.py`、`_verify_5_mutants.py`、
`_test_one_equiv.py`、`_show_llm_detail.py`）硬编码引用了旧路径 `第一次实验汇总/...`，
移动后这些一次性脚本的路径已失效（如需复用请把路径改到本归档目录）。
