# 外部 Benchmark 差分测试（RQ4）

> 原 `第三次实验汇总` + `第四次实验汇总/results` 合并而来。2026-06-01 按"实验内容 + 论文 RQ"重组。
> 对象：4 个公开 GPU kernel benchmark 的生成/人工内核，做 **baseline + 5 维 stress 差分测试**。
> 对应论文：**RQ4（§6.3，Table 8 / 9 / 10）**。重组只用 `git mv` / 文件移动，未改动任何实验数据。

## 目录 ↔ 论文对应

| 子目录 | 数据集 | 内核来源 | 论文 Table 8 |
|---|---|---|---|
| `CUDA-L1/` | CUDA-L1 (deepreinforce-ai) | RL 自动生成 CUDA | 231 完成，B=101，S\B=60，联合 69.70% |
| `AI-CUDA-Engineer/` | SakanaAI AI-CUDA-Engineer | LLM-Agent 生成 CUDA | 222 完成，B=25，S\B=50，联合 33.78% |
| `TritonBench-G/` | thunlp TritonBench-G | 人工编写 Triton（55 仓库） | 138 完成，B=39，S\B=11，联合 36.23% |
| `CUDA-Agent/` | ByteDance·THU CUDA-Agent | Agentic RL 生成 CUDA | 176 完成，B=5，S\B=101，联合 60.23% |
| **合计** | — | — | **767 完成，B=170，S\B=222，联合 51.11%** |

- `报告/`：`RQ4_四公开benchmark差分测试完整报告.md`（主报告）+ `CUDA-Agent结果分析.md`。
- `设计文档与脚本/`：方法/复现 docs（含 `闭环验证实验方案.md` = ADRS 闭环方案，仅方案无数据）、`run_scripts/`、`logs/`、`analyze_anomaly*.py`。
- `_废弃归档/apex_new/`：apex__mlp 单核重跑，**论文未采用**（废弃）。

> 每个数据集目录含 `checkpoint.json`（全部内核，含 SKIPPED）+ `summary.json` + `details/*.json`（仅 COMPLETED）。
> 缺陷判定（论文口径）：`D(k) = baseline.failed>0 OR total_discrepancies>0`（在 COMPLETED 内核上）。

## 三方一致性（论文 ↔ 报告 ↔ 真实数据，2026-06-01 重算核对）

直接从各 `checkpoint.json` 重算，与论文 Table 8 / 9 / 10 **逐格吻合**：

- **Table 8**（联合检出）：四数据集 B/S\B/联合率 = 101·60·69.70% / 25·50·33.78% / 39·11·36.23% / 5·101·60.23%；合计 170 / 222 / 51.11%。✓
- **Table 9**（每维度 applicable / flagged / rate）：4×5=20 格全部吻合（如 CUDA-L1 training 193/113/58.5%、CUDA-Agent value 174/90/51.7%）。✓
- **Table 10**（单维度独占）：CUDA-L1 17、AI-CUDA-Engineer 9、TritonBench-G 3、CUDA-Agent 12，合计 41；|S| = 126/74/30/104 = 334。✓

## 关联实验

- **CUDA-Agent 修复实验（§6.4 / Task D）** 已拆出为顶层目录 `修复实验_CUDA-Agent_TaskD/`：从本目录 `CUDA-Agent/` 里筛出 104 个"baseline 通过但 stress 失败"的 kernel，让 Opus 4.5 修复。**真实修复率 14.4%（15/104，v2.0 严格审查版）**，其余多为"删自定义 CUDA 退回 PyTorch"的作弊。
- 论文 RQ4 关键论据：**CUDA-Agent 声称 98.8% pass rate，本实验 baseline 复现 97.2%，但 5 维 stress 又在 baseline 全过的 171 个里发现 101 个缺陷（59.06%）**——SOTA 系统 release-time 验证远不足以保证鲁棒性。
