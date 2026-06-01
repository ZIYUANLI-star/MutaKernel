# RQ3 论文口径（已移除 Phase II 的 LLM 兜底）

> 用途：论文最终的 MutaKernel 杀伤链**只含 5 个确定性 stress 维度**
> （value / dtype / training / repeated / config）+ tier1_replay，
> **已移除** Phase II 内嵌的 DeepSeek-R1 迭代分析兜底。
> 本目录给出"无 LLM"的重算，**不修改 `../details/*.json` 原始记录**。

## 一键复算

```bash
python recount_paper_no_llm.py
```

读取 `../details/*.json`，以 `kill_summary.deterministic_killed` 为准统计，
输出 `stress_summary_论文口径_无LLM.json`。

## 核对结论（已复算）

| 口径 | 杀数 | 说明 |
|---|---|---|
| **论文 MutaKernel（确定性，无 LLM）** | **166** | = `kill_summary.deterministic_killed=true` 的数量 |
| 旧 any_killed（含 LLM 兜底，历史口径） | 169 | = `any_killed=true`（`../details` 原字段，保留） |
| 被移除的纯 LLM 杀 | 3 | 见下 |
| 确定性 ∩ LLM 同时杀 | 0 | LLM 仅在"5 维全未杀"时触发，故两集合不相交 |

**被移除的 3 个纯 LLM 杀**（论文改由 Task A / RQ2 审计处理）：
- `L1_P49__init_modify__0` → MutaKernel-missed
- `L1_P23__init_modify__0` → MutaKernel-missed
- `L1_P49__arith_replace__11` → operationally_indistinguishable

> 即：旧 169 − 3 = 论文 **166**；下游加固分母不变(1124)，加固分由 98.66% 调整为 **98.40%**
> （详见 `../../报告/` 与 `Eurosys论文写作/eval_section_updates_2026_05_15.md`）。

## 原始数据中的对应字段（未改动）

`../details/*.json` 每个文件内：
- `main_track` = value/dtype/training/repeated_run（4 维，保留）
- `config_track` = config_stress（第 5 维，保留）
- `llm_iterative_analysis` = DeepSeek-R1 兜底（**论文已移除**，仅作历史留档）
- `kill_summary.deterministic_killed` / `.llm_killed` / `.final_killed` = 分层标记
- 纯 LLM 的 prompt/响应原文在 `../_已移除_PhaseII_LLM兜底IO/`
