# 内部变异实验（RQ1 · RQ2 · RQ3 · RQ5）

> 原名：`第二次实验汇总`（含 `第二次实验汇总_补充`）。2026-06-01 按"实验内容 + 论文 RQ"重组。
> 对象：90 个 KernelBench L1+L2 的 LLM 生成 CUDA kernel（DeepSeek-chat + Caesar）。
> 全部为论文主干内部实验。重组只用 `git mv` / 文件移动，未改动任何实验数据。

## 目录 ↔ 论文对应

| 目录 | 实验内容 | 论文位置 | 关键数字 |
|---|---|---|---|
| `RQ1_基线变异与EMD(PhaseI)/` | Phase I：16 算子生成 1646 变异体 + 4 层 EMD 等价检测 | **RQ1**（§3-4；Table 2/3/5） | 939 killed；MS 63.75%（保守）/ 77.67%（乐观） |
| `RQ3_MutaKernel五维增强(PhaseII)/` | Phase II：5 维确定性 stress 增强 | **RQ3**（§5-6.2；Table 6/7） | **MutaKernel +166**（论文口径，见下）；audited→99.82% |
| `RQ2_审计_Opus5轮(TaskA)/` | Task A：Opus 4.5 五轮独立审计 368 个存活体 | **RQ2 审计**（§4.2；Table 4） | 349 prov-eq / 17 op-indist / 2 missed |
| `RQ2_消融_仅PhaseI(TaskC)/` | Task C：跳过 Phase II 直接喂 Opus 的消融 | **RQ2 消融**（§4.2 + Threats） | Opus 直杀 70，其中 69 与 Phase II 重合 → Phase II 不可替代 |
| `RQ5_修复实验(TaskB)/` | Task B：用失败证据让 Opus 修复 18 个 buggy kernel | **RQ5 / §6.4**（正文压缩→project website） | 声称修 16/18，严格审计发现部分作弊（`audit_taskB_strict.json`） |
| `报告/` | 3 份分析报告 | 写作素材 | 完整报告 / 未杀逐项分析 / TaskABC 总结 |
| `设计文档/` | 方法学设计文档 + 复现说明与凭证 | §3/§5 + 附录 | — |
| `_废弃归档/` | 作废重跑(`_archive`) + Phase I 控制台日志 | 无 | 不进论文 |

## ⚠️ 关于 RQ3 的 LLM 兜底（重要）

论文最终的 MutaKernel **只含 5 个确定性 stress 维度**，已**移除** Phase II 内嵌的 DeepSeek-R1 迭代分析兜底：

- 原始 `RQ3.../details/*.json` 仍保留全部字段（含 `llm_iterative_analysis`、`any_killed=169`），作历史留档。
- **论文口径以 `kill_summary.deterministic_killed` 为准 = 166**（= 169 − 3 个纯 LLM 杀）。
- 纯 LLM 的 prompt/响应原文已移入 `RQ3.../_已移除_PhaseII_LLM兜底IO/`。
- 一键复算与详细说明见 `RQ3.../_论文口径_无LLM/`（`recount_paper_no_llm.py` + `stress_summary_论文口径_无LLM.json` + README）。

## 命名说明：算子类别 A/B/C/D ≠ Task A/B/C

- 算子类别 **A/B/C/D** = Arithmetic / GPU-Parallel / ML-Numerical / LLM-Pattern（变异算子分类）。
- **Task A/B/C** = Phase III 三个实验臂（审计 / 修复 / 消融），与算子类别是两套独立命名。

## 数据完整性

重组前后物理文件数守恒（5127），核心 `details/` 原始数据零改动。详见本次整理的校验记录。
