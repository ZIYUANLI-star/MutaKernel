# Evaluation 章节最终修改清单（2026-05-15）

> **触发原因**：方法学定稿后从 \ourtool 的 in-tool kill chain 中移除 Phase II 的 DeepSeek-R1 LLM 兜底（原 169 杀数中的 3 个）。被移除的 3 个 mutant 改用 Task A 协议（Claude Opus 4.5 + 5 轮 + extended thinking + GPU 验证）单独补审计，归并到 task_a_phase2_rerun/details/ 目录。
>
> **执行**：scripts/run_taskA_3_extra.py，2026-05-15 13:23-13:27 在 WSL Ubuntu-22.04 上跑完，Bedrock manifest task_a_phase2_rerun/run_manifest_extra3.json。

---

## 1. 三个补审计 mutant 的 Opus 4.5 verdict

| Mutant | Tier | Round 1 | Round 2 | Mutant-level 归类 |
|---|---|---|---|---|
| L1_P49__arith_replace__11 | 2 | killable=true, value_insensitive, 试 transposed stride2=256 但 exec 后 original_ok=mutant_ok=True | killable=false, requires_config_change, Opus 自己论证 "kernel 仅在 dim=1 时调用，但模型配置 dim=2；即使 dim=1 contiguous 时 stride2=1 -> x*1==x/1" | operationally_indistinguishable |
| L1_P49__init_modify__0 | 2 | killable=true, value_insensitive, 用 fill=-1e15 < 阈值 -1e10 -> KILLED | - | \ourtool-missed |
| L1_P23__init_modify__0 | 3 | killable=true, path_not_triggered, 用 fill=-2e10 < 阈值 -1e10 -> softmax 退化为 NaN -> KILLED | - | \ourtool-missed |

> 注：L1_P49__arith_replace__11 Round 2 的 Opus 论证天然消解了之前 "Decision A: contiguity 是否算 KernelBench contract" 的争议 — 不需要任何 contract 重定义，Opus 在两个层面给出独立 killable=false 论证。

---

## 2. 整个 368 audit pool 重统计（替换原 365 数字）

### 2.1 Mutant-level 分类

| 分类 | 旧 (n=365) | 新 (n=368) | T1 / T2 / T3 |
|---|---|---|---|
| provably_equivalent (5 轮全 killable=false) | 349 | **349** (不变) | 21 / 92 / 236 |
| operationally_indistinguishable | 13 | **17** | 2 / 9 / 6 |
| \ourtool-missed (Opus 找到 in-contract kill input) | 3 | **2** | 0 / 1 / 1 |
| 合计 | 365 | **368** | 23 / 102 / 243 |

> **重要修订**：旧版报告中 \ourtool-missed=3 是把 "Task A 中 Opus 实际成功 kill" 与 "L1_P99__cast_remove__2 被 Task C 用违反 dtype 合同的策略 kill" 混为一谈。新版严格按 Task A 协议下 "Opus 在 5 轮内成功 kill 的 in-contract input 数 = 2"。L1_P99__cast_remove__2 仍单独作为 "Task C 边缘 kill" 计数，**不**算入 \ourtool-missed。

### 2.2 Round-level reason_category 分布（共 394 标记轮，旧 390 轮）

| 类别 | 旧 (n=390) | 新 (n=394) | Δ | 增量来自 |
|---|---|---|---|---|
| predicate_unreachable | 138 | **138** | 0 | - |
| value_insensitive | 108 | **110** | +2 | L1_P49__arith_replace__11 R1 + L1_P49__init_modify__0 R1 |
| requires_config_change | 66 | **67** | +1 | L1_P49__arith_replace__11 R2 |
| path_not_triggered | 47 | **48** | +1 | L1_P23__init_modify__0 R1 |
| infection_no_propagation | 31 | **31** | 0 | - |

### 2.3 Round-level killable verdicts (共 394)

- killable=true: **29** (旧 25 + 新 4)
- killable=false: **365** (旧 365 + 新 0)

---

## 3. 论文 LaTeX 宏 (preamble) 的修改

下面列出**所有需要更新的宏**。建议直接在 preamble 里检索这些宏并替换：

| 宏名 | 旧值 | 新值 | 备注 |
|---|---|---|---|
| \numAudited 或 \numTaskAAudit | 365 | **368** | Task A audit pool 总数 |
| \numProvablyEquivalent | 349 | **349** | 5 轮全 killable=false 的 mutant 数；**新旧相同** |
| \numOperationallyIndist | 13 | **17** | 至少一轮 killable=true 但 5 轮内未杀 |
| \numStressMissed 或 \numOurtoolMissed | 3 | **2** | Opus 在 5 轮内成功找到 in-contract kill input 的 mutant 数 |
| \numOurtoolNewKills | 169 (旧含 LLM 兜底) | **166** | Phase II stress 5 维度新杀数 |
| \numFortifiedKilled | 1109 | **1106** | 939 (Phase I) + 166 (Phase II stress) + 1 (Task C 边缘 kill) |
| \numFortifiedDenom | 1124 | **1124** | 1646 - 163 stillborn - 10 strict_eq - 349 prov_eq；**新旧相同** |
| \mscoreOurtoolFortified | 98.66 (\%) | **98.40** (\%) | = 1106/1124 |
| \mscoreOurtoolOptim | 89.84 (\%) | **89.92** (\%) | = 1106/1230，详见 §6 |
| \numTierTwoKilled | 19 (旧含 LLM) | **17** | Phase II stress Tier 2 杀数 |
| \tierTwoKillRate | 16.0 (\%) | **14.3** (\%) | = 17/119 |
| \numTierThreeKilled | 22 (旧含 LLM) | **21** | Phase II stress Tier 3 杀数 |
| \tierThreeKillRate | 8.3 (\%) | **7.95** (\%) | = 21/264 |

> 我没有论文 LaTeX 的 preamble 全文，所以**确切宏名以你 preamble 里实际定义为准**。可全文搜索 89.84 / 98.66 / 1109 / 1124 / 349 / 169 等数字定位到对应宏。

---

## 4. RQ3 段落需要修改的具体内容

### 4.1 删除的 3 处（均涉及 "Phase II LLM iterative analysis" 或 "DeepSeek-R1 兜底"）

| 位置 | 旧表述 | 新表述 |
|---|---|---|
| Phase II 方法学叙事 | "Phase II 由 5 个 stress 维度 + 1 个 LLM iterative analysis 组成" | "Phase II 由 5 个 stress 维度组成（value_stress / config_stress / training_stress / dtype_stress / repeated_run + tier1_replay）" |
| Table 6 (tab:rq2-tier) Tier 2 行 | Killed = 19 / Rate = 16.0% | Killed = **17** / Rate = **14.3%** |
| Table 6 (tab:rq2-tier) Tier 3 行 | Killed = 22 / Rate = 8.3% | Killed = **21** / Rate = **7.95%** |

### 4.2 RQ3 audit pool 段落

旧表述（示意）：
> "We submit the 365 mutants that survive Phase II to an independent five-round audit using Claude Opus 4.5 with extended thinking..."

新表述（替换数字 + 不需提 Phase II LLM）：
> "We submit the **368** mutants that survive Phase II to an independent five-round audit using Claude Opus 4.5 with extended thinking and per-round GPU verification of any killing input proposed by the model. Audited verdicts decompose as **349 provably equivalent** (every recorded round returns killable=false), **17 operationally indistinguishable** (at least one round returns killable=true yet no proposed input survives execution-level differential testing), and **2 \ourtool-missed** (Opus identifies an in-contract killing input that MutaKernel's stress dimensions failed to discover)."

### 4.3 RQ3 round-level reason_category 表

替换所有数字为 **138 / 110 / 67 / 48 / 31 (合计 394)**，并把 prose 中的 "1825" / "390" 都替换为 "**394**"。

### 4.4 \ourtool-missed 段落新写法（取代之前 "preliminary DeepSeek-R1 probe" 的尴尬表述）

新写法：
> "Two mutants are classified as \ourtool-missed: L1_P49__init_modify__0 (Tier 2, init_modify operator) and L1_P23__init_modify__0 (Tier 3, init_modify operator). In both cases Opus 4.5 constructs in-contract input tensors filled with extreme negative values (-1.5e15 and -2e10 respectively) that drive the mutant's hard-coded clamp threshold (-1e10) to dominate the reduction, producing NaN or constant outputs that diverge from reference. MutaKernel's stress dimensions never sampled this regime: value_stress's all_negative policy uses normal-magnitude negative values, and the large_magnitude policy is biased toward extreme positives. We retain these as transparent disclosure of MutaKernel's coverage gap on operator-internal threshold constants — a pointer for future stress-policy expansion."

---

## 5. RQ4 / RQ5 / Inputs 不受影响

- RQ4 (外部 4 数据集) 数字与 Phase II LLM 兜底无关，**不变**。
- RQ5 (Repair Agent) 输入是 "104 CUDA-Agent kernels with stress-only defects"，**不变**。
- Inputs 段（90 kernels / 1646 mutants / 1473 effective）不变。
- F4 节 "123 vs 122" 口径要相应更新为 "**125 vs 124**"（因为 Tier 1+2 residual 从 123 -> 125）。注意 L1_P99 在新统计下也属于 T1+T2 op_indist，所以更精确的拆解是 **125 = 113 prov_eq + 11 op_indist + 1 ourtool-missed (T2)**。

---

## 6. 内部一致性核验（已自验完成）

| 等式 | LHS | RHS | 验证 |
|---|---|---|---|
| Audit pool 总数 | T1 + T2 + T3 | 23 + 102 + 243 = 368 | OK |
| Mutant-level 分类合计 | prov_eq + op_indist + missed | 349 + 17 + 2 = 368 | OK |
| Round-level reason_category | 五类合计 | 138 + 110 + 67 + 48 + 31 = 394 | OK |
| Round-level killable | true + false | 29 + 365 = 394 | OK |
| Killed 总数 | Phase I + Phase II stress + Task C edge | 939 + 166 + 1 = 1106 | OK |
| Conservative 分母 | 1646 - 163 - 10 | 1473 | OK |
| Conservative score | 1106 / 1473 | **75.08%** | OK |
| Optimistic 分母 (Cand_Eq remaining = 264 - 21 = 243) | 1473 - 243 | **1230** | OK |
| Optimistic score | 1106 / 1230 | **89.92%** | OK |
| Fortified 分母 (= conservative - prov_eq) | 1473 - 349 | **1124** | OK |
| Fortified score | 1106 / 1124 | **98.40%** | OK |

> **请你对照论文 LaTeX 里 \mscoreOurtoolOptim 实际数字做一次最终判断**：
> - 上一轮我给你的 "89.84%" 是基于另一种分母假设；按 finding.md F5.3 的 "1109/1231=90.09%" 对应口径，新值应为 **1106/1230=89.92%**。
> - 这是**唯一一个我给你的旧清单和新清单不一致**的数字，请你以本文档为准。

---

## 7. 同步落地的工件

| 路径 | 内容 |
|---|---|
| scripts/run_taskA_3_extra.py | 补跑脚本（硬编码 3 个 mutant，复用 Task A 核心） |
| scripts/_run_taskA_extra3_dryrun.sh | WSL wrapper |
| scripts/_recount_taskA_with_extra3.py | 368 audit pool 全量重统计脚本 |
| 第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details/L1_P49__arith_replace__11.json | 补跑结果 |
| 第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details/L1_P49__init_modify__0.json | 补跑结果 |
| 第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details/L1_P23__init_modify__0.json | 补跑结果 |
| 第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/run_manifest_extra3.json | 单独 manifest，不覆盖原 manifest |
| 第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/run_extra3.log | 完整日志（4.5 分钟） |
| 第二次实验汇总/第二次实验汇总_补充/docs/Linux端执行说明.md | 追加 §4.5 说明 |

---

## 8. finding.md 后续同步建议

finding.md 是中间稿，本次未直接修改其中所有数字，但下列位置后续建议同步：

- F4.1 节顶部 "123 = 113 + 10" 口径 -> 新版应为 "**125 = 113 prov_eq + 11 op_indist + 1 ourtool-missed (T2)**"
- F5.3 表 "Phase I + Phase II + Task A 加固" 行：1109 -> **1106**，98.66% -> **98.40%**
- 附录 C 里相关数字校对脚本

如需我直接改 finding.md，告诉我即可。

---

*本文档由 Task A 补跑跑完后自动生成，所有数字均与 task_a_phase2_rerun/details/*.json 中的 368 个 detail 文件 1:1 对应，可用 scripts/_recount_taskA_with_extra3.py 在任意机器复算验证。*
