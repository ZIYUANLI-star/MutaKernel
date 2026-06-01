# MutaKernel 方法学修订指导（V6 — 中文版）

> **角色**：导师 → 师姐（主要修订者）
> **文档用途**：师姐据此修订 `MutKernel_EuroSys.pdf` 的 § 3（Phase I：变异生成 / 初始测试 / EMD）与 § 4（Phase II：增强测试）。文档中所有以英文斜体 / 引号给出的段落都是**可以直接抄进论文的素材**，外层叙述与解释为中文。
> **审阅基础**：`MutKernel_EuroSys.pdf` 当前版本 + `D:\doctor_learning\Academic_Project\paper_1\MutaKernel` 代码仓 + 《第二次实验完整报告》。
> **日期**：2026-05-12
> **总体判断**：架构与方法学骨架站得住（EuroSys 投稿可达 Major Revision 区间）；当前最大风险来自**叙事–代码–实验数据三者尚未做系统性对齐**与**少数核心数字 / 设计选择尚缺形式化与 baseline 支撑**。
>
> **V6 修订原则**（相对 V5）：
>
> - **角色定位调整**：从"模拟 EuroSys 审稿人"调为"导师给师姐的写作指导"，去除 reviewer 视角的猜测性语言，改为更直接的修订动作。
> - **正文中文化**：所有解释性叙述改用中文，便于师姐 review。英文素材（论文原文 quote、可粘贴的论文段落、状态机标识符、公式）保留原文。
> - **保留 V5 的关键方法学判断**：撤回"50.3% cross-kill 削弱 Q1 正交性"的批评；保留 S1.2 状态机、finalize 公式块、§ 4.4 双向不对称语义、M10 代码层修复清单。

---

## 一、整体阅读印象

论文的故事线本身站得住：把 mutation testing 这一成熟范式系统地迁移到"LLM 生成 GPU kernel 的 validator 评估"问题，提出 GPU/ML/LLM-aware 算子分类、四层 EMD、算子 × 维度增强测试矩阵——这套架构对 systems community 有吸引力。**方法学的内核（P1–P6 + Q1–Q6 设计原则、graded provisional labels、双轨分离、one-way safety）在论文里已经写得相当清楚**，许多审稿人可能误读的点其实正文都有铺垫，但**这些铺垫被分散在不同段落，章节标题与导言没有把这些设计选择显性凸显出来**。

为方便师姐改稿，我把问题按四类分组，每类对应一种修订策略：

1. **叙事–数据失衡（M1、M3 局部、M5、M9）**：abstract / § 1 中某些关键数字（如 81.1%、把 D 类作为四大贡献之一）的口径或叙事位置与实证数据匹配度需要提升。
2. **代码 ↔ 方法学不一致（M10）**：论文方法描述合理，但代码实现过窄或缺 guard。**修改责任在代码而非论文**——若代码层修复，论文可保持当前写法甚至受益于更好的实验数据。
3. **关键 deliverable 缺失（M4、M6、M7、M8）**：mutation score 公式、Operator × Dimension Kill Matrix、§ 4.4 LLM Iterative Analysis 正文、baseline / ablation 等审稿必看 artifact 缺失。
4. **结构与可读性增强（S1）**：论文当前结构在实质内容上已经传达了"graded provisional → final"的两段式判定，但章节标题与导言对读者不够明显。需要 cosmetic 调整 + § 4.4 填空 + § 4.5 公式补充。

---

## 二、结构与可读性建议（在实质对齐基础上的 cosmetic 增强）

### S1. 让"等价判定的两段式累积"对读者显性化（论文实质已对，只需小幅调整 + 补缺）

**结论**：论文 § 3.3 与 § 4.5 在**实质上**已经把"Phase I = provisional graded labels；Phase II = final equivalence confirmation"这条逻辑表达清楚了，证据可见于以下原文（这几句师姐改稿时**不要动**）：

- § 3.3 Problem Setup：*we assign each mutant one of two **graded** equivalence labels reflecting the **strength of the underlying evidence***
- § 3.3 Pipeline Overview：*Layers 0 and 1 may assign a **provisional** strict label, Layer 2 may assign a **provisional** candidate label*
- § 3.3 Phase I Output：*Survived and candidate_eqivalent mutants remain **unresolved** and are the input to Phase II*
- § 4.5 Per-Mutant Final Label：*Confirmed_eqivalent mutants are Tier 3 entries that survived every deterministic dimension and the LLM stage. The original candidate verdict is now backed by both the EMD evidence chain and the Phase II battery.*

所以**论文的方法学骨架是对的**——这条主线已经在那里，**不需要做章节级重构**。但师姐当前正文里这条主线被分散在 § 3.3 三个子节和 § 4.5 末尾，审稿人第一遍读 Abstract → § 3.3 → § 4 时容易把 "EMD = final 判定、Phase II 只是补充测试" 当作默认理解，**与论文真实意图刚好相反**。下面给出"把 two-stage cumulative judgment 显性化"的详细写作指导，分两部分。

---

#### S1.1 Phase I 中 EMD 阶段的**显性定位**（保留四层 EMD 的命名权，但要把性质说清楚）

**新定位的一句话陈述**（建议师姐作为 § 3.3 第一段第一句直接写出，**可粘贴进论文**）：

> *Phase I's EMD stage is a **provisional, graded, precision-first** equivalence filter. It produces per-mutant equivalence verdicts whose **finality is conditional on the underlying layer's evidence strength** — only Layer 0/1 verdicts are terminal; Layer 2/3 verdicts are explicitly forwarded to Phase II for confirmation.*

这条定位由三个属性构成，每一条都要让读者看到论文已经做的设计选择，并配上对应的理由：

**(A) Provisional ≠ Final**：EMD 的产出在 mutant 标签空间里只占两个**终局格 (terminal cells)** —— `killed` (initial test 杀掉) 与 `strict_equivalent` (Layer 0/1 命中)。其余两个格 `candidate_equivalent` 和 `survived` 都是 **provisional**，明确等待 Phase II 增强测试与 LLM 推理进一步判定。师姐改稿时这一点必须在 § 3.3 Problem Setup 与 § 3.3 末尾**各重复一次**——论文当前只在 Pipeline Overview 中段隐式提到，太弱。

**为什么这样定位**：等价问题在通用情形下是不可判定的 [Budd & Angluin 1982]，任何**有界、低成本、可证明性优先**的等价判定器都必然留下一个 "我不能证明它等价但也不能证伪" 的灰区。论文 EMD 的设计选择就是**显式承认这个灰区并把它命名为 candidate_equivalent**，而不是把 Phase I 强行包装成 final 判定。这正是论文 P3 (cumulative evidence) 与 P4 (one-way LLM safety) 在标签空间里的具体体现。

**(B) Graded by evidence strength**：四层产出按**证据强度递降**排列，**对应不同的 finality**。建议师姐直接把下面这张表放进 § 3.3 Pipeline Overview 子节开头：


| Layer   | 证据类型                                       | 产出标签                                | 是否 terminal | 在最终判定中的角色                                  |
| ------- | ------------------------------------------ | ----------------------------------- | ----------- | ------------------------------------------ |
| Layer 0 | Source normalization（AST / token-level）    | `strict_equivalent`                 | **是**       | 直接进入 final equivalence 集                   |
| Layer 1 | Static rules（句法不可达、guard 等）                | `strict_equivalent`                 | **是**       | 直接进入 final equivalence 集                   |
| Layer 2 | Dynamic bitwise testing（operator-directed） | `candidate_equivalent`              | **否**       | 必须由 Phase II 增强维度确认                        |
| Layer 3 | LLM-assisted review（one-way revocation）    | `candidate_equivalent` 或 revocation | **否**       | 仅能撤销 Layer 2 给出的 candidate 标签，不能独立授予新的等价标签 |


这张表是论文目前缺的核心结构图。读者一旦看到它，就不会再质疑"为什么 Layer 0/1 命中少却仍值得保留"——因为 Layer 0/1 是**唯一**能在 Phase I 内部就终结判定的层，它们的低 yield 是被 precision-first 设计目标主动选择的结果。

**为什么这样定位**：mutation testing 文献里 graded equivalence reasoning 是经典做法（Offutt & Pan 1997 的 constraint-based equivalence detection；Just et al. 2014 的 mutation tagging），但当前论文没有显式挂上这套术语，读者会按"四层全等"的直觉去读，反而觉得 Layer 0/1 是 over-engineering。把 grading 表写明，正好把这个误读堵住。

**(C) Precision-over-recall, low-yield-by-design**：在 mutation testing 里，false equivalent（漏报：把 non-equivalent 误判为 equivalent）的代价是**永久**的——这个 mutant 会从 mutation score 分母里被永久剔除，对应的 validator gap 永远不会被发现；而 false survivor（多测：把 equivalent 误判为 survivor）只是多一次 Phase II 测试预算。因此 EMD 内部的设计准则必然是**宁可放过、不可错杀**。

师姐这一点必须在 § 3.3 显式写出来，并且锚定到论文已经存在的 P1/P3/P4 原则。当前正文虽然有 P1–P6 设计原则块，但没有把"为什么 Layer 0/1 命中低不是 bug"这个审稿人必问的问题预先回答。建议在 P3 (cumulative evidence) 后增补一句（**可粘贴进论文**）：

> *"This precision-first stance also explains the per-layer yield distribution observed in § 5: Layer 0 + Layer 1 together produce only 10 strict_equivalent verdicts (0.6% of corpus), but every such verdict is backed by source-level or static proof and never revisited. Layer 2 produces 264 candidate_equivalent verdicts (16.0% of corpus), all of which are explicitly subject to Phase II revocation. Layers 0/1 trade recall for finality; Layer 2 trades finality for coverage; Layer 3 is a one-way safety net that can only revoke."*

**对论文章节级的具体动作（最小改动版，师姐改稿可按表对照执行）**：

1. **§ 3.3 标题加副标题**：`Equivalent Mutant Detection (Provisional Graded Labeling)`。保留 four-layer EMD 的命名权（这是论文核心贡献之一），用括号副标题告诉读者性质。
2. **§ 3.3 Problem Setup 子节** 首段第一句替换为上文 S1.1 的"一句话陈述"。
3. **§ 3.3 Pipeline Overview 子节** 开头插入上文的 4 行 grading 表（或紧凑 inline 描述：`Layer 0/1 → strict_equivalent (terminal)`；`Layer 2/3 → candidate_equivalent (provisional)`）。
4. **§ 3.3 末尾 Phase I Output 子节** 把现有 "Survived and candidate_eqivalent mutants remain unresolved" 一句扩成显式承接 Phase II 的桥接段（**可粘贴进论文**）：
  > *"The output of Phase I is a per-mutant tuple `(provisional_label, evidence_layer, evidence_artifact)`. Mutants with `killed` or `strict_equivalent` exit the pipeline immediately and contribute to the denominator and numerator of the Phase-I mutation score (§ 4.5). Mutants with `candidate_equivalent` or `survived` are forwarded to Phase II (§ 4) for final adjudication; their Phase-I labels remain mutable until Phase II terminates."*

---

#### S1.2 等价变异体判定的**整体流程**（two-stage cumulative judgment）——师姐改稿时如何写

这条主线在论文当前版本中是**正确的、但隐藏的**。建议师姐在 § 3 开头（紧接 § 3.1 之前）或 § 3.3 开头插入一个 **半页的 lifecycle 描述**，明确给出"从 mutant 生成到 final label 决定"的完整状态机。下面给出具体内容与写法。

##### 1) 标签状态机（建议作为 Figure / 紧凑表 + 文字伴随）

```
Mutant lifecycle (canonical):

Phase I:
  1.  Generated (§ 3.1)
            │
            ▼
  2.  Initial Differential Testing (§ 3.2)
            │
            ├─→ stillborn       [TERMINAL]  (compile / runtime crash)
            ├─→ killed          [TERMINAL]  (initial test divergence)
            └─→ unresolved      → step 3
            │
            ▼
  3.  EMD (§ 3.3, four layers, precision-first)
            │
            ├─ Layer 0/1 hit  →  strict_equivalent       [TERMINAL — final]
            ├─ Layer 2 hit    →  candidate_equivalent    [PROVISIONAL]
            │                       └─ may be revoked by Layer 3 (one-way)
            └─ no hit         →  survived                [PROVISIONAL]

Phase II (input = candidate_equivalent ∪ survived):
  4.  Tier classification (§ 4.1) gates which dimensions apply per mutant.

  5.  Deterministic enhanced testing (§ 4.3, six dimensions, Q3 no short-circuit)
            │
            ├─ any dimension confirms a kill → killed   [TERMINAL — final]
            └─ all applicable dimensions clean → step 6

  6.  LLM iterative analysis (§ 4.4)
        Input: Phase I evidence chain + Phase II deterministic evidence
        Per round, the LLM emits exactly ONE of two verdicts:

            ┌─ verdict = "kill" → LLM proposes a concrete counter-example input
            │     │
            │     ├─ worker re-runs with the proposed input and confirms kill
            │     │      → killed                              [TERMINAL — final]
            │     │      (LLM-assisted kill; counts in mutation score numerator)
            │     │
            │     └─ worker re-runs and does NOT confirm kill
            │             → round DISCARDED; contributes NO equivalence evidence
            │               (P4 one-way safety: a failed kill attempt is not an
            │                equivalence assertion); continue to next round or
            │                budget exhaustion
            │
            └─ verdict = "equiv" → LLM emits a categorized equivalence reason
                  (predicate_unreachable / value_insensitive / path_not_triggered /
                   infection_no_propagation / requires_config_change)
                  → records one LLM equivalence assertion; LLM stage TERMINATES
                  → step 7 (with LLM_verdict = equiv)

        Budget exhausted with NO "equiv" verdict and NO confirmed kill
          → step 7 (with LLM_verdict = ⊥, treated as no LLM equivalence support)

  7.  Final label assignment (§ 4.5):
        // confirmed_equivalent requires ALL THREE conditions:
        if entered Phase II as candidate_equivalent
           AND all applicable deterministic dimensions clean
           AND LLM stage produced an explicit "equiv" verdict
              → confirmed_equivalent           [TERMINAL — final]

        // All other unkilled cases collapse to resistant_survivor:
        //   - entered as survived (any LLM outcome), OR
        //   - entered as candidate_equivalent but LLM only emitted failed kill
        //     attempts (LLM_verdict = ⊥), i.e., LLM never asserted equivalence
        otherwise (not killed in steps 2/3/5/6)
              → resistant_survivor             [TERMINAL — final]
                 (validator gap; NOT equivalence)

8. 最后人工判定等价变异体判定的准确率
```

**师姐在写流程图正文时必须反复点出的关键设计选择**（**P4 one-way safety 在 Phase II 的具体落点**）：

- **LLM 在 Phase II 中是 *双向但非对称* 的**：它既能支持 kill（路径 A），也能支持 equivalence（路径 B），但两条路径**都受外部验证 / 显式声明约束**——
  - 路径 A：LLM 提议 kill → 必须由 worker 实跑确认；worker 不确认则该轮丢弃，**绝不**反向当作等价证据。
  - 路径 B：LLM 主张 equivalence → 必须显式给出 categorized reason（5 类之一）；只有这样才能进入 confirmed_equivalent 的升级条件。
  - **"LLM 尝试杀但没杀掉" ≠ "LLM 声明等价"** —— 这是论文 P4 原则的核心实现细节，师姐必须在 § 4.4 用一段专门解释。
- `**strict_equivalent` 与 `confirmed_equivalent` 是两条不同的路径**，都属于 "final equivalent" 集，但证据基础不同：前者基于 Phase I 的 source-level / static proof，后者基于 Phase I candidate verdict + Phase II 全维度 negative 证据 + Phase II LLM 主动 equivalence verdict。论文 § 4.5 应明确：**同一个 equivalent 集合，但 evidence provenance 分开记录**。
- **三种结局必须严格区分**：`survived`（Phase I 未判定、Phase II 未启动或未完成）→ 不应出现在 final labels；`resistant_survivor`（Phase II 跑完但未被任何路径杀死且没有 LLM equiv verdict）→ 暴露 validator gap，**不属于 equivalent 集**；`confirmed_equivalent`（candidate + 全维度 clean + LLM 主动声明 equiv）→ 进入 equivalent 集。即使一个 `survived` mutant 通过了 Phase II 全部维度，其 final 标签也只能是 `resistant_survivor`——因为它从未进过 EMD 等价证据链，缺少 Layer 2 dynamic bitwise + Layer 3 LLM equivalence reasoning 的初始支撑。**审稿人会追问这条 hygiene，师姐务必预先答辩**。
- **Phase II LLM 的双重职能 (kill synthesizer + equivalence reasoner) 是不对称的**：作为 kill synthesizer，其建议**必须**经 worker 验证才有效；作为 equivalence reasoner，其判断需 categorized reason 并**显式输出 equiv verdict**。这是 § 4.4 必须填空的核心内容。

##### 2) Final-label assignment 公式块（建议放在 § 4.5）

```
Given a mutant m, let:
  initial(m) ∈ {stillborn, killed_initial, unresolved}       # Phase I initial test outcome
  L1(m)      ∈ {strict_eq, ⊥}                                # Phase I Layer 0/1 verdict
  L2(m)      ∈ {candidate_eq, ⊥}                             # Phase I Layer 2 verdict
  L3(m)      ∈ {confirm, revoke, ⊥}                          # Phase I Layer 3 verdict (revocation only)
  D(m)       ⊆ {value, dtype, repeat, train, cfg, tier1_replay}   # applicable Phase II deterministic dims
  K_D(m)     = 1 iff any d ∈ D(m) confirmed a kill in Phase II deterministic stage

  LLM(m)     ∈ {kill_confirmed, equiv, ⊥}                    # Phase II § 4.4 LLM stage verdict
                                                             #   kill_confirmed: LLM proposed an input AND
                                                             #                   the worker re-ran with that
                                                             #                   input AND confirmed a kill.
                                                             #   equiv:          LLM emitted an explicit
                                                             #                   equivalence verdict with one
                                                             #                   of five categorized reasons.
                                                             #   ⊥:              budget exhausted; LLM neither
                                                             #                   produced a worker-confirmed
                                                             #                   kill nor an equiv verdict.
                                                             #                   (Failed kill attempts collapse
                                                             #                    here; P4 one-way safety.)

Final label finalize(m):
  if initial(m) = stillborn:                                 return stillborn
  if initial(m) = killed_initial:                            return killed
  if L1(m) = strict_eq:                                      return strict_equivalent       # path P1
  if K_D(m) = 1  or  LLM(m) = kill_confirmed:                return killed                  # path P2
  if L2(m) = candidate_eq  and  L3(m) ≠ revoke
     and  K_D(m) = 0  and  LLM(m) = equiv:                   return confirmed_equivalent    # path P3
  otherwise:                                                 return resistant_survivor      # path P4

Equivalent set = {m : finalize(m) ∈ {strict_equivalent, confirmed_equivalent}}
Killed set     = {m : finalize(m) = killed}
Mutation score = |Killed| / (|Generated| − |Stillborn| − |Equivalent|)
```

**这套公式的关键点师姐必须在 § 4.5 用中文/英文伴随写出来**：

1. **path P2 中 `LLM(m) = kill_confirmed` 而非 `LLM proposed a kill`**：LLM 提出 kill 不等于 kill 成立，必须 worker 实跑确认。这条与代码 `scripts/run_stress_enhance.py` 的 LLM-worker 验证回路完全对齐。
2. **path P3 中 `LLM(m) = equiv` 是硬条件**：仅当 LLM **显式**产出 equivalence verdict（5 类 reason 之一）时才允许升级；budget exhausted 或只发生过 failed kill attempts（`LLM(m) = ⊥`）都**不**满足升级条件。这一点是 P4 one-way safety 在公式层的精确实现。
3. **path P4 兜底**：所有未被 P1/P2/P3 命中的 unkilled mutant 一律落入 `resistant_survivor`——包括 (a) Phase II 入口标签为 `survived`、(b) 入口为 `candidate_equivalent` 但 LLM 阶段失败收尾（`LLM(m) = ⊥`）、(c) 入口为 `candidate_equivalent` 但被 Layer 3 撤销（`L3(m) = revoke`）等所有边缘情形。这些都暴露 validator gap，**不进入 equivalent 集**。
4. **Mutation score 分母排除 Stillborn 与 Equivalent**：这是 mutation testing 的标准定义（Just et al. 2014），但当前论文 abstract 给出 81.1% 等数字时没有显式给出该分母——必须形式化。

这个公式块解决三件事：(i) 让审稿人在 ≤ 1 页内验证每一条路径都**封闭、互斥、覆盖全集**；(ii) 让 abstract / introduction 里的 81.1% 等数字都能 trace 回这个 finalize 函数的具体路径；(iii) 把 P1/P3/P4 原则与代码 `scripts/full_block12.py` 的标签状态机对齐——师姐的代码里就是这么实现的，只是论文还没把它显性写出来。

##### 3) 章节级 stitch（师姐改稿对照表：改哪里、改多长）


| 位置                           | 当前状态                          | 建议改动                                 | 字数     |
| ---------------------------- | ----------------------------- | ------------------------------------ | ------ |
| § 3 章节开头                     | 直接进 § 3.1                     | 加 1 段 lifecycle 概览 + 引一张状态机 figure   | ~200 词 |
| § 3.3 标题                     | "Equivalent Mutant Detection" | 加副标题 "(Provisional Graded Labeling)" | 4 词    |
| § 3.3 Problem Setup 首段       | 现有 graded labels 描述           | 替换首句为 S1.1 的"一句话陈述"                  | ~30 词  |
| § 3.3 Pipeline Overview 首段   | 直接列四层                         | 先插入 4 行 grading 表                    | ~80 词  |
| § 3.3 末尾 Phase I Output      | 1 句 "forwarded to Phase II"   | 扩成完整桥接段（见 S1.1 第 4 点）                | ~80 词  |
| § 4 章节导言（首段）                 | 直接进 § 4.1                     | 加 dual-function 声明（保留 V4 原句）         | ~70 词  |
| § 4.4 LLM Iterative Analysis | **空标题**                       | 必填，写双重职能 + cost-benefit（详见 M4）       | ~250 词 |
| § 4.5 Per-Mutant Final Label | 现有 confirmed_equivalent 描述    | 增补 finalize 公式块 + 4 条路径列表            | ~150 词 |


总增量约 **800 词 + 1 张状态机图 + 1 个公式块**，**不构成章节级重写**。

##### 4) 与 Abstract / § 1 的回写

完成上述结构性补充后，师姐要相应回写 Abstract 与 § 1：

- Abstract 当前提到 "two-phase pipeline" 但没有点出"等价判定本身是两段累积"。建议加一句（**可粘贴进论文**）：
  > *"Equivalence verdicts are produced by a **two-stage cumulative pipeline**: Phase I assigns a graded provisional label (strict, candidate, or unresolved); a mutant is promoted to `confirmed_equivalent` only if it (i) entered Phase II as a candidate, (ii) survived every applicable deterministic stress dimension, and (iii) received an **explicit categorized equivalence verdict** from the Phase II LLM stage — a failed LLM kill attempt does not, by itself, support equivalence (one-way safety)."*
- § 1 Contributions 列表中 "four-layer EMD" 应改为 "four-layer EMD as a **precision-first provisional** equivalence filter"，并加一条新贡献（**可粘贴进论文**）：
  > *"A two-stage cumulative judgment protocol that separates evidence accumulation (Phase I EMD) from final adjudication (Phase II battery + LLM), with **asymmetric LLM authority** — LLM-proposed kills require worker re-execution to count, and LLM-asserted equivalence requires an explicit categorized verdict, ruling out single-stage over-claiming on either side."*

---

##### 写作哲学（写给师姐的提醒）

**为什么这套 cosmetic 增强值得做**：论文当前已经把 two-stage cumulative judgment 写进正文，但这条主线被分散在 4 个不同子节里，读者必须自己把碎片拼起来才能看到它。EuroSys 一审审稿人通常只会按 "abstract → contributions → method overview → 一节一节扫" 的顺序读，**任何需要读者主动拼图的设计选择都会被默认丢失**。S1.1 给出的"一句话定位 + grading 表 + precision-first 理由"，与 S1.2 给出的"状态机 + finalize 公式 + 章节 stitch 表"，本质上是把师姐已经完成的设计工作**显性挂到读者第一眼能看到的位置**。

**为什么坚持不做章节级重写**：mutation testing 这一范式的审稿人群体（Just / Offutt / Papadakis 圈）对 "Equivalent Mutant Detection" 这个术语非常敏感，**重命名或拆分 EMD 章节会引发"作者是不是不熟悉这一文献"的疑虑**。保留 § 3.3 EMD 的命名与四层结构、只加副标题与导言句，是既维护学术 lineage、又消除读者误读的最稳态修订。

**与 V6 整体修订哲学的一致性**：S1 全部建议都不修改论文 *方法本身*，只调整 *叙事顺序与显性度*。这与 V6 总原则"论文方法骨架对、只在叙事位置精确化"完全一致；同时把 M3 (81.1% 口径)、M4 (§ 4.4 填空 + Kill Matrix)、M7 (mutation score 公式) 这三个 P0 项统一收编到 S1.2 的 finalize 公式 + lifecycle 图之下——一次性修复，避免局部打补丁。

---

## 三、重大问题（Major — 审稿人会重点追问）

### M1. 算子分类法的实证规模与叙事并重感不匹配

**问题**：abstract 与 § 1 把 "GPU-, ML-, and LLM-aware mutation taxonomy" 列为四大贡献之一；§ 3.1 把 16 个算子按 A/B/C/D 四类近乎均匀展开，行文上四类享有同等地位。但实际产出严重不均衡。

**证据**（来源：《第二次实验完整报告》§ 2.3–2.4）：


| 类别       | 算子数 | 变异体总数 | 占比        | 备注      |
| -------- | --- | ----- | --------- | ------- |
| A 经典算术   | 3   | 757   | 46.0%     | 主力      |
| B GPU 并行 | 4   | 702   | 42.6%     | 主力      |
| C ML 数值  | 7   | 178   | 10.8%     | 算子多但产出少 |
| D LLM 特定 | 2   | **9** | **0.55%** | 几乎可忽略   |


算子级数据中几处突出：

- `reduction_reorder` = **0 mutant**（C 类核心算子之一）→ 详见 M10 反 2，**代码问题**
- `stab_remove` = 7 mutants（但 § 3.1 重点描述了它）
- `layout_assume` = 5 mutants（D 类两算子之一）
- `broadcast_unsafe` = 4 mutants，**0 kill**（D 类两算子之一）
- `acc_downgrade` = 46 mutants，**40 stillborn (87%)** → 详见 M10 反 3，**代码问题**
- `cast_remove` = 38 mutants，20 stillborn (53%) → 详见 M10 反 5，**代码问题**

**师姐改稿建议**：

1. **D 类降级措辞**（论文修改，**可粘贴**）：
  > *"We additionally include a preliminary set of LLM-specific operators (Category D). Its current empirical coverage is limited (2 operators, 9 mutants in our corpus); we present it as a starting point for future expansion rather than as a fully developed sibling of Categories A–C."*
2. **§ 3.1 末尾或附录补一张算子产出表**：列出每个算子的设计触发条件、实际触发次数、stillborn 率、kill 率。所有数据 `summary.json` 都有。
3. **关于 `reduction_reorder` 0 mutant + `acc_downgrade` 87% stillborn + `cast_remove` 53% stillborn**：这三处异常**主要是代码问题而非论文写作问题**（详见 M10 反 2/3/5）。建议**优先 fix 代码后重跑变异体生成**——预计能把 60+ 个 stillborn 转化为有效 mutant，对实验数据有显著提升。**若不重跑，则论文 § 3.1 必须诚实承认这些算子的局限性**。
4. **§ 3.1 必须显式披露 `SAMPLE_PER_OP = 3`**（论文遗漏）：代码 `scripts/full_block12.py:57` 中每个 (kernel, operator) 对最多采样 3 个变异点。建议写（**可粘贴**）：
  > *"For each (kernel, operator) pair we generate all candidate mutants then uniformly sample at most three of them (SAMPLE_PER_OP=3) to bound per-kernel JIT cost; § 5.x reports the sensitivity of our headline numbers to this sampling rate."*

---

### M2. EMD 四层架构：论文实质定位已对，但缺数据 deliverable

**结论**：论文 § 3.3 Problem Setup + P1/P3/P4 设计原则**已经**把 four layers 锚定为 "graded evidence with precision-over-recall"——即 Layer 0/1 是 high-precision low-yield filters、Layer 2 是 statistical screen、Layer 3 是 one-way revocation。**所以 V3 中"重新框定四层定位"的大段重写建议是冗余的**——论文方法描述实质上已经这样写了。

**真正缺失的是数据 deliverable 与代码一致性**：

**问题 1（数据缺失）**：四层各自的产出分布没有展示。读者只能从其他报告反推（Layer 0 → strict_eq = 1；Layer 1 → strict_eq = 9 全来自 dead_host_constant；Layer 2 → candidate_eq = 264；Layer 3 撤销 ≥ 119 次）。

**问题 2（论文未披露的细节）**：

- Layer 1 四条规则中 3 条 0 命中（`_boundary_unreachable`, `_dead_write`, `_mask_noreach`）。这部分有两种来源：
  - `_boundary_unreachable` 论文与代码都明确限定为 `threadIdx.x <op> blockDim.x` 形态——**这是设计选择**，论文已诚实描述；只是该形态在 LLM 实际生成代码中罕见。
  - `_mask_noreach` 论文描述较广（"*exclude only threads outside the output range*"），但代码实现窄（5 变量名正则）——**这是代码窄化于论文，详见 M10 反 4**。
- Layer 3 撤销率 ≥ 119 次但未在论文报告。

**问题 3（代码与论文不一致）**：库函数 `src/mutengine/equivalent_detector.py:459-466` 中 `classify_survived_mutants` 在 `cuda_only` 分支直接 short-circuit 到 CANDIDATE_EQUIVALENT，与论文 P6 (No short-circuit) 矛盾。生产管线 `scripts/full_block12.py:514-525` 正确遵循 P6。**论文对，代码库函数错，详见 M10 反 1**。

**师姐改稿建议**：

1. **§ 3.3 末尾必须补 Layer 1–3 贡献分解表**：列出每层处理多少变异体、其中多少被分入 strict / candidate / passed-down / revoked。这一改动**不需要重写正文**，只需要把代码已生成的数据汇总到一张表。
2. **§ 3.3 Layer 3 段补一句撤销率**（**可粘贴**）：
  > *"In our experiment, Layer 3 reviewed N mutants previously labeled candidate-equivalent by Layer 2 and revoked the verdict in M cases under P4 one-way safety; these revoked mutants subsequently appear as Tier 2 in Phase II."*
3. **§ 3.3 Layer 1 段加一句承认 3 条规则 0 命中**（**可粘贴**）：
  > *"In our corpus, the `boundary_unreachable`, `dead_write`, and `mask_noreach` rules yielded zero or one hit; this conservative recall reflects the syntactic shape of our pattern matchers, and we view the rules as a sound but narrow first-pass. We discuss broadening pattern coverage as future work."*
4. **代码库函数 short-circuit 路径**：删除死代码，或对齐生产管线（详见 M10 反 1）。

---

### M3. abstract 中 81.1% 口径精确化 + Kill Matrix 必须可见

**结论（撤回 V4 的 Q1 批评）**：经过对 Q1 与 Q3 在论文中的实际作用域复核，我**撤回** V3 / V4 中"50.3% cross-kill 削弱 Q1 正交性"的判断。Q1 与 cross-kill 数据本质上不在同一层级：

- **Q1 关心的是测试维度本身的正交性**——每个维度扰动一个独立的输入轴（values / dtype / repetition / mode / batch size），以及由此带来的 *per-kill attribution* 唯一性。
- **First-kill mode 分布 (137 + 12 + 9 + 3 + 3 + 3 + 2 = 169) 是一个精确的 partition**——由 first-kill 调度器**构造性保证**每个 kill 被归因到恰好一个维度，**这正是 Q1 第二个 claim "attributes cleanly to a single mechanism" 的实证**。
- **50.3% cross-kill 是 Q3 (no cross-dimension short-circuit) 强制全维度执行带来的另一个度量**——它讲的是 *bug 在输入空间中的冗余覆盖*（同一个 bug 可在多个独立输入轴上被触发），属于 bug 空间的性质，而非测试维度的性质。

所以**论文 Q1 文字本身已经准确**——用 cross-kill 数据去推测 Q1 不成立是方法学上的越位。**Q1 不需要任何澄清性修订**；保持论文原文即可。

**师姐改稿建议**（M3 缩减到两项）：

1. **abstract 与 § 1 中 81.1% 口径必须精确化**（这是真正的写作问题，与 Q1 无关）：当前 abstract "*targeted value regimes accounting for 81.1% of newly killed mutants*"——读者自然误解为 "81.1% are due to value-regime stress only"。实际是 first-kill mode 占比。建议改为（**可粘贴**）：
  > *"In our Phase II battery, value-regime stress is the **first-kill mode** for 137 of 169 newly killed mutants (81.1%); 151 of 169 (89.3%) are killable by value-regime stress at least once, and 50 of 169 (29.6%) are independently killed by configuration-stress variation. Together, these statistics suggest value-regime weakness is the primary but not the sole gap in current validators."*
  >  这一修订**反而强化** "validator weakness is structured and operator-dependent" 的结论——因为它给出了 first-kill / independent-kill / cross-kill 三种口径的细分证据。
2. **§ 4.5 承诺的 Operator × Dimension Kill Matrix 必须在正文展示**（详见 M4 deliverable 清单）。注意：cross-kill 数据本身是这个 matrix 的一种聚合视图（按 mutant 计数），它对于 Q4 (dual-track separation) 与 Q5 (LLM as last resort) 的有效性论证仍然是有信息量的——但**不是用来质疑 Q1，而是用来支撑"value 维度是主要但非唯一的 validator gap"这一论文核心结论**。

---

### M4. 关键 deliverable 缺失（含 § 4.4 空标题）

论文当前有若干个被审稿人视为 "deliverable" 的产物在正文里缺失或不可见。这些都是 P0 级修订。

**缺失 1：§ 4.4 LLM Iterative Analysis 空标题**。这是提交版的 desk-reject 信号。

**实验事实**（来源：《第二次实验完整报告》§ 3.8）：

- LLM 触发：368 个变异体
- LLM 总轮次：406 轮
- LLM 实际 kill：**3 个**（0.8% conversion，1.8% of Phase II kills）
- LLM 等价判断 reason 分布：predicate_unreachable 186 (45.8%) + value_insensitive 128 (31.5%) + path_not_triggered 58 (14.3%) + infection_no_propagation 30 (7.4%) + requires_config_change 4 (1.0%) = 406 轮

LLM 阶段的真实工作量分布是 **99% 等价推理 + 1% kill 合成**。这与论文当前对 LLM 阶段定位为"kill 路径"的暗示不一致。建议 § 4.4 内容包含（**可整段粘贴进论文**）：

> *"The Phase II LLM stage serves a **dual, asymmetric** function under our P4 one-way safety principle and Q5 LLM-as-last-resort constraint. Per round, the LLM emits exactly one of two verdicts.*
>
> *(a) **Kill synthesizer.** When the LLM proposes a counter-example input, the input is dispatched to the worker for actual re-execution against the reference implementation. Only a worker-confirmed divergence promotes the mutant to `killed`; a failed kill attempt is discarded and contributes no equivalence evidence (P4: the LLM cannot ratify equivalence by negation). In our experiment this path yielded 3 worker-confirmed kills out of 368 invocations (0.8% conversion).*
>
> *(b) **Equivalence reasoner.** Alternatively, the LLM may emit an explicit categorized equivalence verdict, restricted to one of five reasons that map onto known equivalence mechanisms: predicate_unreachable (45.8%), value_insensitive (31.5%), path_not_triggered (14.3%), infection_no_propagation (7.4%), and requires_config_change (1.0%) — 406 verdicts in total. Such a verdict, combined with the mutant having survived every applicable deterministic dimension, promotes a `candidate_equivalent` mutant to `confirmed_equivalent` (§ 4.5).*
>
> *The asymmetry is essential: budget exhaustion or rounds containing only failed kill attempts are treated as ⊥ (no LLM equivalence support), and the mutant defaults to `resistant_survivor` rather than `confirmed_equivalent`. The deterministic battery must first exhaust the relevant input axes, and the LLM must affirmatively assert equivalence — neither condition alone is sufficient."*

**缺失 2：Operator × Dimension Kill Matrix 不可见**。§ 4.5 一句话提及 "feeds the per-operator analysis in §??"，§?? 是死链接。这是 Phase II 整套架构服务的最终输出物，缺失等于核心 deliverable 不可见。建议在 § 4.5 或 § 5 用 16×7 小热力图给出。

**缺失 3：所有 `[TODO]` 占位**。提交版必须清零（与 `[?]` 不同，TODO 是作者自己留的）。

**缺失 4：模型版本与确定性的诚实化**。§ 3.3 Layer 3 写 "DeepSeek-chat with temperature 0 for deterministic output"，但《第二次实验完整报告》中 Phase II LLM 实际用 DeepSeek-R1。两阶段调用不同模型如果是有意设计应说明；"deterministic output" 是 over-claim（主流商业 LLM 在 temp=0 下仍有非确定性），建议改为（**可粘贴**）：

> *"We set temperature to 0 to minimize sampling variance and log every (prompt, response, verdict, confidence) tuple for repeatability auditing."*

---

### M5. "1,646 mutants from 90 LLM-generated kernels" 的统计基线缺失（在变异规则守卫中有讲到）

**问题**：abstract 与 § 1 多次出现 "thousands of mutants from real LLM-generated kernels"；§ 3.1 末尾给出精确数字但未交代 90 这个数字的筛选漏斗。EuroSys 对 reproducibility 与 sampling bias 极敏感。

**师姐改稿建议**：

1. **§ 3.1 必须给出 reproduction filter 漏斗**：从 X 个 KernelBench L1+L2 task 出发，LLM 每个 task 生成 Y 次，编译通过 Z 个，运行通过 = 90 个。reproduce filter 阈值（`torch.allclose` 容差、是否要求 speedup > 0）也必须给出。
2. **90 个 kernel 在 L1/L2 间的分布**：从 mutant id（`L1_P`*, `L2_P`*）反推显然跨 level，但具体分布需在 § 3.1 给出。
3. **"thousands of mutants" 应改为精确数字**：直接写 "1,646 mutants from 90 kernels across KernelBench Level 1 and Level 2"。
4. **种子 kernel 在 stress regime 下的正确性是隐含假设**（详见 m3）。

---

### M6. 标题中的 "Repair Framework" 在 § 3 / § 4 中没有得到对应

**问题**：论文标题 *MutaKernel: A Mutation Analysis Driven Testing & Repair Framework for LLM-Generated GPU Kernels*。但当前 § 3 / § 4 主要描述 testing / validator adequacy / stress testing / coverage suggestion——**没有 repair 闭环**。§ 4.5 末尾 "*coverage suggestions form the bridge from Phase II's diagnostic output to the test-suite enhancement we evaluate in §??*" 是死链接。

**师姐改稿建议**（二选一）：

- **路径 A（推荐，工作量小）**：标题改写为更精确的方法定位，例如 *"MutaKernel: Mutation-Driven Validator Adequacy Measurement and Coverage Enhancement for LLM-Generated GPU Kernels"*，避免审稿人追问"repair 在哪"。
- **路径 B（工作量大）**：保留 "Repair Framework" 但必须在 § 4 或 § 5 补全闭环：(a) 从 kill matrix 自动生成多少条测试套件增强建议；(b) 吸收后 validator 的 mutation score 提升多少；(c) 对原始 LLM generation pipeline 的 false-accept rate 是否减少。（正在做这部分实验）

---

### M7. Mutation Score 公式与标签流转必须形式化

**问题**：§ 3.2 / § 3.3 描述了 5 种 Phase I 状态；§ 4.5 又引入 confirmed_equivalent、first-kill mode；但**论文中没有出现 final mutation score 的形式化公式**。abstract 给出 81.1% 但没有定义任何 score。

**师姐改稿建议**：在 § 4.5 增补一个公式块。**完整 finalize 状态函数应采用 S1.2 第 2 节给出的精细版本**（包含 `LLM(m) ∈ {kill_confirmed, equiv, ⊥}` 三态、显式 P4 one-way safety 兜底等关键细节）。本节给出与之配套的 **mutation score 多口径定义**：

```
Final per-mutant label after Phase II (see S1.2 for full finalize() definition):
  killed                if any Phase I initial test OR any Phase II deterministic
                          dimension produced a kill, OR LLM proposed an input
                          whose worker re-execution confirmed a kill
  stillborn             if Phase I initial testing crashed / failed to compile  
  strict_equivalent     if Phase I Layer 0 or Layer 1 proved equivalence
  confirmed_equivalent  if m entered Phase II as candidate_equivalent AND
                          survived every applicable deterministic dimension AND
                          § 4.4 LLM issued an explicit categorized equivalence
                          verdict (NOT merely budget exhaustion or failed kill
                          attempts; see P4 one-way safety in S1.2)
  resistant_survivor    otherwise (validator gap; NOT equivalence)

Mutation scores (denominator excludes stillborn and certified-equivalent classes):
  Phase I conservative MS   = K_I / (N − N_still − N_strictEQ)
  Phase I optimistic MS     = K_I / (N − N_still − N_strictEQ − N_candEQ)
  Final conservative MS     = (K_I + K_II,det + K_II,LLM) / (N − N_still − N_strictEQ)
  Final optimistic MS       = (K_I + K_II,det + K_II,LLM) / (N − N_still − N_strictEQ − N_confEQ)
  Aux. config-stress MS     = K_II,config / N_PhaseII  (reported separately per Q4)
```

并配口径说明：(a) `stillborn` 不携带行为信号，排除；(b) `strict_eq` 可证明等价，排除；(c) `candidate_eq` 在保守口径保留（仍可能被 Phase II 杀掉），在乐观口径排除；(d) `confirmed_eq` 只在 Phase II 后才能确定，乐观口径下排除；(e) `K_II,LLM` 仅计入 *worker-confirmed* 的 LLM kill，与 P4 一致；(f) `config_stress` 按 Q4 双轨原则独立报告（不与 main score 合并）。

**与 S1.2 的关系**：M7 这里给出的是 *score-aggregation* 层面的多口径定义；S1.2 给出的是 *per-mutant 状态转移* 层面的精细 finalize 函数。两者应在 § 4.5 中并列出现——前者负责对外报告的数字，后者负责单个 mutant 的标签可追溯性。

---

### M8. Tier 3 challenge 子集白名单的 post-hoc 修补需诚实化

**问题**：§ 4.1 "Tier 3 Challenge Subset" 把 10 算子白名单描述为预设设计，但实际是后扩——首批白名单只有 6 个算子，导致 35 个 Tier 3 变异体被跳过；修复后扩为 10 并补跑这 35 个。Artifact 公开后 git history / 代码注释会暴露。

**师姐改稿建议**：

1. 改写为（**可粘贴**）：
  > *"We initially selected six operators based on a pilot analysis (`sync_remove`, `launch_config_mutate`, `mask_boundary`, `index_replace`, `relop_replace`, `const_perturb`); after observing 35 Tier 3 mutants from four additional operators (`arith_replace`, `cast_remove`, `init_modify`, `scale_modify`) that satisfied other challenge criteria, we expanded the whitelist to the present ten. All 264 candidate-equivalent mutants therefore enter Phase II."*
2. 同时说明两阈值差异：`confidence < 0.98`（Phase II 挑战阈值）与 `confidence > 0.7`（Phase I Layer 3 revoke 阈值）对应 P1 风险偏好在两阶段的应用——Phase I 阶段对 LLM 更信任（避免 false equivalent 累积），Phase II 阶段更激进（因为已经有 deterministic battery 兜底）。

---

## 四、中等问题（Mid — 影响论证强度，但不构成 reject）

### m1. Tier 2 的代码定义比论文宽

**问题**：论文 § 4.1 *"Tier 2 contains survived mutants that passed all 112 rounds of Layer 2 but whose Layer 3 LLM review returned `possibly_killable`"*。但代码 `scripts/run_stress_enhance.py:150-165` 最后一行 `return 2` 是默认 fallback——所有 Layer 2 通过但 Layer 3 未明确返回 `possibly_killable` 的 survived 变异体都被分到 Tier 2。

**师姐改稿建议**：改为（**可粘贴**）：

> *"Tier 2 contains survived mutants that passed all 112 Layer 2 rounds and were either explicitly demoted by Layer 3 (verdict = `possibly_killable`, P4) or not certified as candidate-equivalent by Layer 3 (default); under P1 we treat the latter as Tier 2 by default."*

### m2. Layer 2 早停语义未在论文披露

**问题**：论文 § 3.3 *"compares their outputs bitwise across 112 rounds of inputs"*；代码 `equivalent_detector.py:387-433` 首次出现 bitwise 差异即返回 `(False, ...)`，**早停**。

**师姐改稿建议**：改为（**可粘贴**） *"Layer 2 runs **up to** 112 rounds; the first bitwise divergence in the random or stress phase short-circuits the comparison."* 同时披露 `tested_random_seeds` / `tested_policies` 在早停后可能不完整——这是 § 4.3 value_stress 去重逻辑的关键前提。

### m3. NaN/Inf fallback 形式化与种子正确性假设

**结论**：论文 § 4.2 fallback 描述实质合理（*"we fall back to comparing o against m directly: the implicit assumption is that the original kernel's output on that input is correct enough to act as a reference. If o is also NaN/Inf the round is skipped."*），但形式化 `killed ⇐⇒ ok(r,o) ∧ ¬ok(r,m)` 没有覆盖 fallback 路径。

**师姐改稿建议**：

1. **形式化补全**：
  ```
   令 compare_target(r, o) = o  if r contains NaN/Inf else r
   killed ⇐⇒ ok(compare_target, o) ∧ ¬ ok(compare_target, m)
  ```
2. **bitwise 端点明确**：value_stress 辅助击杀使用的 bitwise 比较是 `o vs m`（代码 `_stress_worker.py:244`：`bitwise_orig_mut_eq = _bitwise_eq(orig_out, mut_out)`），**不是 r vs m 也不是 compare_target vs m**。建议改写为（**可粘贴**）：
  > *"The value_stress dimension adds one auxiliary kill criterion. Let `compare_target = r` in the normal case and `compare_target = o` in the NaN/Inf fallback case. If both `o` and `m` pass `allclose` against `compare_target` but `o` and `m` differ **bit-for-bit**, the mutant is killed by bitwise divergence."*
3. **种子正确性是隐含假设（threat to validity）**：fallback 把 o 当作 ground truth，但 o 只通过了 KernelBench 默认 oracle 的 5 个 random seed，在 stress regime 下可能本身就有微小偏差，系统性地隐藏 m 与 o 同向漂移的 mutation。建议在 § 4.2 末尾加一段 threat-to-validity 讨论。

### m4. Replay 步骤的归类不一致

**问题**：§ 4.3 *"The deterministic block contains five testing dimensions ..., preceded by a Tier 1-only replay step. ... yielding six deterministic kill modes plus the LLM stage. The first four dimensions and the replay step form the main track"*——replay 既是 "preceded by" 又是 main-track 的一行，语义混乱。

**师姐改稿建议**：统一为（**可粘贴**）：

> *Main track = {tier1_replay (T1 only), value_stress, dtype_stress, repeated_run, training_stress}*
> *Config-stress track = {config_stress}*
> *Together they form six deterministic kill modes (M1–M6); the LLM iterative analysis is reported as a seventh (M7).*

### m5. atol/rtol 多套阈值切换缺少依据

**问题**：Phase I oracle = 1e-2；Phase II main = 1e-2；repeated_run self-consistency = 1e-6；Phase II Layer 2 = bitwise。阈值切换缺少统一论证。

**师姐改稿建议**：§ 4.2 或表 4 footnote 给出各阈值依据。1e-6 可表述为（**可粘贴**）：

> *"lower bound on FP32 round-off when identical CUDA invocations run on the same JIT path; divergence above this threshold across launches signals data race rather than benign numerical noise."*

### m6. 应报告 per-kernel 聚类后的统计

**问题**：当前论文以 1,646 mutants 为单位统计，但 mutant 非独立样本——多个 mutant 可能来自同一 kernel / 同一算子。81.1% / 75.22% / 90.01% 这些数字可能被少数高产 kernel 拉偏。

**师姐改稿建议**：§ 5 中加一张 per-kernel 聚合表（per-mutant / per-kernel median / per-kernel IQR），并做 bootstrap over kernels 的 95% CI 报告。

### m7. Mutant Realism 必须显式呈现

**问题**：§ 3.1 仅在 D 类一处提及 *"two operators were distilled from patterns we observed in this class"*，但代码仓 `src/mutengine/realism_validator.py` 已实现完整 Realism Coverage Rate 评估（19 类真实根因 → 16 算子的映射，diff-driven mining），论文当前几乎完全没有提及。这是 construct validity 的核心证据。

**师姐改稿建议**：§ 3.1 末尾或附录补一个 "Mutant Realism" 半页小节：mutation rules 如何从 compile-correct/run-wrong kernels 的 iteration diff 中提炼；每类算子给 1 个真实 LLM 错误片段；量化指标 N 个真实错误中 M 个被 16 算子覆盖 = X% Realism Coverage Rate；人工抽样验证。

### m8. 运行成本与可扩展性必须报告

**问题**：论文 § 3.3 表 1 已把 JIT 编译 30–120 秒列为挑战，但方法部分没给出整体成本数字。EuroSys 是 systems venue，审稿人对成本敏感。

**师姐改稿建议**：§ 5 或附录补一张成本表：总 GPU/CPU/墙钟小时；JIT 编译次数；timeout / OOM 数；平均每 mutant Phase I / Phase II 时间；LLM token / 美元成本；cache 命中率；并行度。所有数据已在 run_log 中。

### m9. 硬件与软件环境必须精确

**师姐改稿建议**：§ 5 头部或附录给出：GPU 型号 / CUDA 版本 / driver / PyTorch / Triton / nvcc flags / compute capability / TF32 / fast math / FTZ / denormal 行为 / random seed 控制方式 / deterministic flags。

### m10. LLM 使用应作为 Threat to Validity 集中讨论

**师姐改稿建议**：§ 6 (Related Work) 之前增加一个 § 5.x "Threats to Validity"：模型版本漂移；temperature=0 不保证确定；prompt / response 是否公开；数据污染风险；LLM 是否偏向作者预设；缓解措施（Phase I Layer 3 only revokes、Phase II LLM-proposed kill 必须 worker 验证、所有 trace 作为 artifact 发布）。

---

## 五、次要问题（Minor — 影响表述质量）

### s1. 拼写错误

§ 3.3 末尾 "Phase I Output" 段落连续 4 次 `strict_eqivalent` / `candidate_eqivalent`（少一个 `u`）。

### s2. "16 operators" 分组数应在 § 3.1 开头一次性给出

建议加 *"We organize 16 operators into four categories (A: 3, B: 4, C: 7, D: 2)."*

### s3. "fixed-shape isolates value from shape weakness" 缺前置铺垫

§ 1 出现得突然。建议移至 § 4 dual-track separation (Q4) 段作为 main-track / config-stress-track 分离的动机。

### s4. 21 stress strategies 完整列表应放附录

§ 4.3 列 4 个 + "..." 后省略其余 17 个；建议附录给完整 21 条策略小表，并明确每条所属的 5 个 rationale 之一。

### s5. 流程图建议加色彩分级

图 3 / 图 4 用色彩区分 Phase I 强证据路径（L0/L1 strict）、弱证据路径（L2/L3 candidate）、Phase II main-track vs config-stress-track。

### s6. 表 1 与四层 EMD 的 traceability

表 1 列 6 个 GPU/CUDA challenges，但 § 3.3 之后的四层设计没有逐一映射回表 1。建议每层描述末尾用括号标注 "(addresses challenges X, Y in Table 1)"。

### s7. § 3.1 各 Category 段落开头缺 design hypothesis

每个 Category 段落开头应有 *"the design hypothesis is ..."*，让审稿人在读到实验结果时能直接核对 hypothesis 是否被证伪 / 验证。

---

## 六、师姐改稿优先级（按工作量与收益排序）

### 总体判断（写给师姐）

师姐，这不是一篇 idea 不成立的论文——架构对应一个真实且重要的问题，设计原则 P1–P6 与 Q1–Q6 是有理论依据的，方法学正文也在大体上把这些原则表达清楚了。问题主要在三个维度：

1. **代码层与论文方法描述的差距**：M10 列出的 6 处反向不对齐，**论文方法都是合理的**，是代码窄化 / 缺 guard / 违反统一框架的问题。这些应当**优先修代码**，不应改论文。修代码后，论文当前写法可以保持不变并受益于更好的实验数据。
2. **deliverable 缺失**：Kill Matrix、§ 4.4 正文、Mutation Score 公式、baseline、ablation——这些都是审稿人必看 artifact，必须补。
3. **少数关键叙事位置的精确化**：D 类降级、81.1% 口径精确（first-kill / independent-kill / cross-kill 三口径并列）、reproduction filter 漏斗、temperature 表述弱化、Tier 3 post-hoc 诚实化——这些是真正的论文写作修订。**注意：Q1 / 维度正交性论文原文已经准确**，cross-kill 数据不构成对 Q1 的反例，**师姐不要动这一段**。

### P0（提交前必须修复，否则 desk-reject 风险）

1. § 4.4 LLM Iterative Analysis 必须有正文（M4 缺失 1）——可粘贴段落已在 M4 给出。
2. 所有 `[TODO]` 占位填空（M4 缺失 3）。
3. § 4.5 承诺的 **Operator × Dimension Kill Matrix** 必须在正文展示（M4 缺失 2）。
4. 标题中的 "Repair Framework" 与方法学内容对齐（M6 二选一，推荐路径 A）。

### P1（核心方法学诚信修订 — 代码与论文并行）

**代码层（建议优先，因为可减少论文改写量）**：

1. **M10 反 3 + 反 5：加 dtype / include guard 后重跑** `acc_downgrade` 与 `cast_remove` 的变异体生成——预计带来 60+ 个新有效 mutant，强化论文已有结论，**论文无需改动**。
2. **M10 反 2：扩展 `reduction_reorder` 实现**到手写循环 + warp reduction，使该算子在实验中产生有效 mutant；**论文无需改动**。
3. **M10 反 4：扩展 `_mask_noreach` 变量名表**与句法形态；**论文无需改动**。
4. **M10 反 1：删除库函数 `equivalent_detector.py:459-466` 死代码路径**或对齐生产管线；**论文无需改动**。
5. **M10 反 6：让 config_stress 遵守 § 4.2 通用 NaN fallback 框架**；**论文无需改动**。

**论文层（在代码修复后展开）**：

1. M1: D 类降级 + 算子产出表 + 披露 `SAMPLE_PER_OP=3`。
2. M2: § 3.3 末尾补 Layer 1–3 贡献分解表 + Layer 3 撤销率 + Layer 1 三规则 0 命中诚实承认（**注意**：四层定位本身已正确，不需要大段重写）。
3. M3: 81.1% 口径精确化（first-kill / independent-kill / cross-kill 三种口径都给出）。**Q1 原文无需修订**——cross-kill 数据不在 Q1 作用域内，Q1 关心的是测试维度正交性与 per-kill attribution 唯一性，两者论文均已成立。
4. M4: § 4.4 LLM 阶段双重职能 + cost-benefit 数字 + 模型版本一致性 + temperature 表述弱化。
5. M5: reproduction filter 漏斗 + L1/L2 分布 + 精确数字替代 "thousands"。
6. M7: Mutation score 公式形式化（在 § 4.5 落地）。
7. M8: 至少 3–4 项 baseline / ablation。
8. M9: Tier 3 白名单 post-hoc 诚实化。
9. m3: NaN/Inf fallback 形式化 + bitwise 端点显式 + 种子正确性 threat-to-validity。
10. m7: Mutant Realism 显式呈现（半页小节）。

### P2（可读性与可复现性）

1. **S1 cosmetic 调整**: § 3.3 加副标题 "(Provisional Graded Labeling)"；§ 4 导言加一句 "dual function" 声明；图 3/4 标注 "provisional → final"。
2. m1, m2, m4, m5：术语统一、阈值依据、双阈值解释。
3. m6: per-kernel 聚类统计与 bootstrap CI。
4. m8, m9, m10：运行成本、硬件环境、LLM threats to validity。
5. s1–s7：拼写、章节铺垫、附录补充、图色彩、design hypothesis。

完成以上 P0 + P1 后，方法学部分基本可以撑住 EuroSys 严格审稿。

---

> **文档备注**：本指导所引用的所有代码行号与统计数据均可在 `D:\doctor_learning\Academic_Project\paper_1\MutaKernel\` 仓库当前 head 复现；主要交叉来源为 `第二次实验汇总\第二次实验完整报告.md` 与 `scripts/full_block12.py`、`scripts/run_stress_enhance.py`、`scripts/_stress_worker.py`、`src/mutengine/equivalent_detector.py`、`src/mutengine/static_equiv_rules.py`、`src/mutengine/operators/ml_semantic.py`、`src/mutrepair/enhanced_inputs.py`、`src/stress/policy_bank.py` 等文件。师姐改稿过程中如有任何方法学判断不确定的地方，可随时回来对照本文档对应章节，或直接和我讨论。

