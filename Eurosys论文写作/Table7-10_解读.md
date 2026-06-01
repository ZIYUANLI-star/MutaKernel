# MutaKernel EuroSys 论文 — Table 7 起表格逐项解读

> **来源**：`MutKernel_EuroSys (3).pdf`
> **解读范围**：Table 7 至 Table 10（全文最后 4 张表，对应 §6 Evaluation）
> **结构**：每张表给出 1) 标题原文 / 2) 表格内容 / 3) 这张表在论文里**说明了什么**（含数据来源、与上下文章节的联系、对论文核心论点的贡献）

---

## Table 7 — MutaKernel 相对 baseline validator 在三种分母约定下的 mutation score 对比

### 7.1 原标题

> *Mutation score comparison between the baseline validator and MutaKernel under the three denominator conventions from RQ1 (conservative, optimistic) and RQ2 (audited). The audited denominator excludes the 366 mutants the RQ2 audit confirmed as equivalent or operationally indistinguishable.*

### 7.2 表格内容

| Stage | Killed | Conservative | Optimistic | Audited |
|---|---|---|---|---|
| KernelBench (baseline) | 939 | 63.75% | 77.67% | 84.82% |
| **MutaKernel** | **1,105** | **75.02%** | **89.84%** | **99.82%** |
| Δ over baseline | **+166** | **+11.27 pp** | **+12.17 pp** | **+15.00 pp** |

### 7.3 这张表说明什么

这是论文 **RQ3（MutaKernel 相对 baseline 的提升量）** 的**核心结论表**，定位在 §6.2 Overall improvement 段。它把三个不同 mutation-score 口径**并排呈现**，目的是把"MutaKernel 多杀了多少"这个简单数字拆解到三种解读层次：

1. **Conservative（保守口径，分母 = 1,473）**：把 EMD 流水线没有源码级证明的所有 mutant 都算作"潜在 validator 逃逸"——即把 264 个 candidate_equivalent 也算进分母。这是 mutation score 的**下界**，MutaKernel 提升 **+11.27 pp**。
2. **Optimistic（乐观口径，分母 1,209 → 1,230）**：沿用经典 mutation-testing 惯例排除 candidate_equivalent。注意 MutaKernel 在 264 个 candidate_equivalent 中**反过来杀掉了 21 个**（推翻 EMD 的等价判定），所以乐观分母从 1,209 升到 1,230。提升 **+12.17 pp**。
3. **Audited（审计口径，分母 = 1,107）**：用 RQ2 中 Opus 4.5 + 人工审查得到的 366 个等价 mutant（349 provably + 17 operationally indistinguishable）做更精确的等价过滤。**MutaKernel 拿到 99.82%**，相对 baseline 的 84.82% 提升 **+15.00 pp**。

#### 7.3.1 三个口径相互校验的方法学意义

- **Conservative** 是**对 MutaKernel 最不利**的口径（分母里含一堆其实是等价的 mutant），却仍然提升 11.27 pp → 抗"分母作弊"指控
- **Audited** 是**对 MutaKernel 最有利**的口径（已经把真等价的拿掉），到达 99.82%——只剩 2 个未杀，已接近上限
- **Optimistic** 与 Conservative 的差距（77.67% vs 63.75% = 13.92 pp）大小反映了 candidate_equivalent 这一类不确定标签的体量

#### 7.3.2 与论文核心论点的关系

- 摘要中写 "raises the conservative mutation score from 63.75% to 75.02%"——这就是 Table 7 第一列的数字
- 摘要中写 "raises the mutation score to 99.82%"——这就是 Table 7 audited 列的数字
- "Only 2 of 1,107 effective mutants remain as residual coverage gaps"：1,107 − 1,105 = 2，**直接由 Table 7 audited 列推出**
- **+166** = 1,105 − 939：MutaKernel 在 baseline 失败的 534 个 mutant 中又抓回 166 个，构成论文反复引用的"166 additional kills"

---

## Table 8 — 四个公开 benchmark 上 baseline vs MutaKernel 的缺陷检测对比

### 8.1 原标题

> *Defect detection on the four public benchmarks. 𝐵 is the set of kernels flagged by the baseline validator; 𝑆 is the set flagged by any of MutaKernel's five stress dimensions. The joint detection rate is |𝐵 ∪ 𝑆|/Completed.*

### 8.2 表格内容

| Benchmark | Completed | 𝐵 (baseline) | 𝑆 \ 𝐵 (stress-only) | \|𝐵 ∪ 𝑆\| | Joint rate |
|---|---|---|---|---|---|
| CUDA-L1 | 231 | 101 | 60 | 161 | 69.70% |
| AI-CUDA-Engineer | 222 | 25 | 50 | 75 | 33.78% |
| TritonBench-G | 138 | 39 | 11 | 50 | 36.23% |
| **CUDA-Agent** | 176 | 5 | **101** | **106** | **60.23%** |
| **Total** | **767** | **170** | **222** | **392** | **51.11%** |

### 8.3 这张表说明什么

这是 **RQ4（MutaKernel 在真实 LLM/RL 生成的公开 kernel 上能发现多少 baseline 漏掉的缺陷）** 的**主结论表**，定位在 §6.3 Overall detection rate 段。它把"控制好的 mutation 设置"扩展到**真实生产场景**，验证 §6.2（study dataset 内）的结论是否推广。

#### 8.3.1 关键数据点

- **总体提升**：767 个 kernel 中 baseline 只标 170 个有问题，MutaKernel 多发现 **222 个 (56.63% of 392)**，把联合检出率从 22.2% 提升到 **51.11%**
- 这个 22.2% 是怎么算的：170 / 767 = 22.16% → 是 baseline 单独的检出率；论文摘要里 "raising the joint defect rate from 22.2% to 51.11%" 的两个数字都来自这张表
- **MutaKernel 的"独占发现"比例 = 222/392 = 56.63%**：超过一半的真缺陷只有 MutaKernel 能发现

#### 8.3.2 CUDA-Agent 是**核心案例**（也是摘要里的"招牌数字"）

- 作者声称 KernelBench pass rate 98.8%（其 paper 自报）
- 论文用 baseline validator 复现：只有 5/176 = **2.84% 缺陷率**——印证作者的高 pass rate 声明
- MutaKernel 五维 stress test 上场：又抓出 **101 个 stress-only 缺陷**，把联合缺陷率拉到 **60.23%**
- 这 101 个不是"输入越合同"造成的——每个 kernel 的失败都是在 declared shape & dtype 内
- 所以摘要中说："the gap between the 98.8% author-reported pass rate and the 60.23% stress-tested defect rate quantifies, for one SOTA system, how much pass rates reported under default-input validation overstate the correctness of generated kernels"

#### 8.3.3 跨 benchmark 差异巨大

`𝑆 \ 𝐵`（stress-only 比例）跨 benchmark 从 22% 到 95%：

| Benchmark | stress-only / 总缺陷 | 解释 |
|---|---|---|
| TritonBench-G | 11 / 50 = 22% | **人工写的 Triton**，本来就有更多明显 bug，baseline 自己就能抓到大部分 |
| CUDA-L1 | 60 / 161 = 37% | RL 生成的 kernel |
| AI-CUDA-Engineer | 50 / 75 = 67% | LLM-Agent 生成 |
| **CUDA-Agent** | **101 / 106 = 95%** | Agentic-RL 训练时**过拟合到 baseline validator**——典型 reward hacking |

→ **stress-only 比例越高，说明该系统训练时越依赖 baseline validator 当做 reward signal，过拟合越严重**。这是论文 §1 reward hacking 论述的实证。

#### 8.3.4 与上下文章节的联系

- 衔接到 §6.3 后面的 Table 9（每维度适用性 / 检出率）和 Table 10（单维度独占贡献）
- 摘要中数字 "MutaKernel flags 222 defective kernels that the baseline misses across the four benchmarks"——直接由这张表的 Total `𝑆 \ 𝐵` 列得出

---

## Table 9 — 每个 stress dimension 在四个 benchmark 上的可适用性与检出率

### 9.1 原标题

> *Per-dimension applicability and detection rate across the four benchmarks. Appl. is the number of completed kernels on which the dimension could execute. Flag. is the number on which the dimension detected a discrepancy. Rate = Flag. / Appl. Rows are ordered by cross-benchmark total Flag count.*

### 9.2 表格内容（缩排为可读形式）

| Dimension | CUDA-L1 Appl. | CUDA-L1 Flag. | CUDA-L1 Rate | AI-CUDA-Engineer Appl. | AI-CUDA-Engineer Flag. | AI-CUDA-Engineer Rate | TritonBench-G Appl. | TritonBench-G Flag. | TritonBench-G Rate | CUDA-Agent Appl. | CUDA-Agent Flag. | CUDA-Agent Rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| training_stress | 193 | 113 | 58.5% | 221 | 65 | 29.4% | 120 | 23 | 19.2% | 174 | 94 | 54.0% |
| value_stress | 193 | 110 | 57.0% | 221 | 67 | 30.3% | 120 | 22 | 18.3% | 174 | 90 | 51.7% |
| config_stress | 203 | 49 | 24.1% | 219 | 22 | 10.0% | 121 | 9 | 7.4% | 132 | 4 | 3.0% |
| dtype_stress | 186 | 44 | 23.7% | 42 | 4 | 9.5% | 123 | 5 | 4.1% | 109 | 11 | 10.1% |
| repeated_run | 215 | 63 | 29.3% | 221 | 29 | 13.1% | 138 | 24 | 17.4% | 156 | 15 | 9.6% |

### 9.3 这张表说明什么

这是 §6.3 Per-benchmark dimension contribution 段的**第一张分解表**，回答"五个 stress dimension 的相对实力"问题。设计上特别考究两件事：

#### 9.3.1 "Applicable" 列的引入是关键的方法学诚信

不像传统 benchmark 表只报 "flag/total"，这里区分 **Applicable**（这维度能不能在这 kernel 上跑）和 **Flagged**（跑了之后有没有发现 bug）。原因：

- `dtype_stress` 需要参考实现接受 perturbed dtype（FP16/BF16）——很多 KernelBench 模型只支持 FP32
- `config_stress` 需要参考和 mutant 都能在 perturbed batch size 上不挂
- `repeated_run` 需要 5 次 rerun 能在 worker timeout 内完成

→ **Rate = Flagged / Applicable**（不是 / Completed），保证不会把"无法测试"和"测试通过"混为一谈

#### 9.3.2 行排序反映"绝对杀力"

按四个 benchmark 的 **Flagged 总和** 降序排：`training_stress > value_stress > config_stress > dtype_stress > repeated_run`。

但这个"绝对杀力"排序**可能误导**——因为：

1. `training_stress` 和 `value_stress` 在每个 benchmark 上的 Rate 都很高（最高 58.5% 在 CUDA-L1）
2. `dtype_stress` Rate 看起来很低（9.5% 在 AI-CUDA-Engineer），但 Applicable 只有 42——它就跑了 42 个 kernel，已经抓了 4 个
3. `config_stress` 在 CUDA-Agent 上 Rate 仅 3.0% (4/132)，但仍然抓到了 2 个其他维度都漏掉的（见 Table 10）

→ **绝对杀力 ≠ 不可替代**。论文的 narrative 是：不要因为 `dtype_stress` / `config_stress` Rate 低就裁掉它们——具体不可替代的证据要看 Table 10

#### 9.3.3 三个具体发现（§6.3 三个 observations）

1. **检测率与"独占贡献"解耦**：`training_stress` 和 `value_stress` Rate 最高，但独占贡献其实并不绝对最高
2. **跨 benchmark 关系会翻转**：CUDA-Agent 上 `repeated_run` Rate 仅 9.6%，但**独占抓 5 个**——比 `training_stress` 在同 benchmark 的独占数（4 个）还多
3. **低 Rate 多是 coverage artifact**：`dtype_stress` 在 AI-CUDA-Engineer 只 9.5%，但因为 Applicable 只有 42（reference 拒收非 FP32），实际利用率正常

#### 9.3.4 与表 8 / 表 10 的关系

- Table 8 给"baseline vs MutaKernel"对比（**横向**：每个 benchmark 上两个 validator 的对比）
- Table 9 给"五个 dimension 的横向比较"（**纵向**：每个 benchmark 内拆解 MutaKernel 内部 5 个 dimension）
- Table 10 接着给出"哪几个 dimension 是不能裁的"实证（独占抓获数）

---

## Table 10 — 每个 dimension 单独抓住（其他维度都漏）的 kernel 数

### 10.1 原标题

> *Number of stress-flagged kernels caught by exactly one dimension on each benchmark. The bottom row reports the total stress-flagged set size |𝑆| on each benchmark.*

### 10.2 表格内容

| Dimension | CUDA-L1 | AI-CUDA-Engineer | TritonBench-G | CUDA-Agent | Total |
|---|---|---|---|---|---|
| training_stress | 7 | 2 | 1 | 4 | **14** |
| value_stress | 2 | 4 | 0 | 0 | **6** |
| repeated_run | 1 | 0 | 1 | 5 | **7** |
| config_stress | 5 | 0 | 0 | 2 | **7** |
| dtype_stress | 2 | 3 | 1 | 1 | **7** |
| **Single-dim total** | **17** | **9** | **3** | **12** | **41** |
| Stress-flagged \|𝑆\| | 126 | 74 | 30 | 104 | 334 |

### 10.3 这张表说明什么

这是 §6.3 的"消融实验等价物"。**核心问题：如果删掉某个 dimension，会不会漏掉某些缺陷？** 答案是：**会**——因为这表中每行的非零值代表"删掉这个 dimension 会丢失多少缺陷"。

#### 10.3.1 数据解读

- **41 = 总单维度独占**（每个被某一个 dimension 独家抓到的 kernel）。占 334 stress-flagged 的 12.3%——其余 87.7% 是多个 dimension 同时抓到（cross-confirmed）
- 单维度独占总数没有一个为 0：**五个 dimension 全部都不能删**
- `training_stress` 单独贡献 14 — 占独占总数 34%，最多
- `value_stress` 单独贡献 6 — 比预期低（因为它的 Rate 最高，但很多 case 同时被其他 dimension 也抓到了）

#### 10.3.2 跨 benchmark 模式

- **CUDA-L1（17 单维独占）**：`training_stress` 7 + `config_stress` 5 主导；GPU 配置敏感的 RL 训练特化
- **TritonBench-G（3 单维独占）**：最少；说明手写 Triton kernel 的 bug 多数能被多维度交叉确认
- **CUDA-Agent（12 单维独占）**：`repeated_run` 5 + `training_stress` 4 主导——指向 race condition 类的"训练态独有"bug
- **AI-CUDA-Engineer（9 单维独占）**：`value_stress` 4 + `dtype_stress` 3——说明该系统在 value distribution 上没怎么测过

#### 10.3.3 与 Table 9 形成对照（这是论文方法学的精彩之处）

| Dimension | Table 9 跨 benchmark Flag 总和 | Table 10 单维度独占总和 | 解读 |
|---|---|---|---|
| training_stress | 295 | 14 | 杀力最强，独占也最强 |
| value_stress | 289 | 6 | **杀力第 2，但独占只第 5**——大部分 case 别人也抓得到 |
| repeated_run | 131 | 7 | 杀力中游，独占很强（关键 race 检测）|
| config_stress | 84 | 7 | 杀力次低，但独占高（grid/block 敏感 bug 独家）|
| dtype_stress | 64 | 7 | 杀力最低，独占同样不低（FP16/BF16 敏感独家）|

→ **结论**：`value_stress` 看起来强，但其实最容易"被替代"；反而 `repeated_run` / `config_stress` / `dtype_stress` 这三个看起来 Rate 低的，是**真的不可替代**。

#### 10.3.4 论文用途

§6.3 末尾的 Finding 段直接引用：
> *"Removing any of these three lower-rate dimensions would therefore cost more uncaught defects than removing value_stress, even though those dimensions look weaker on aggregate detection rate."*

这个 finding 是论文方法学合理性的关键防御——避免审稿人说"你应该删掉那几个低 Rate 的 dimension"——Table 10 直接给出反证。

---

## 综合解读：Table 7-10 的整体逻辑

Table 7-10 构成 §6 Evaluation 的**四步论证链**，每张表回答一个不同问题：

```
Table 7 ──→ "MutaKernel 相对 baseline 提升多少？"（mutation score 三口径对比）
   ↓
Table 8 ──→ "推广到真实 LLM-generated kernel 怎么样？"（四 benchmark 上的真实缺陷）
   ↓
Table 9 ──→ "五个 dimension 的相对实力如何？"（每个 dimension 的覆盖率和检出率）
   ↓
Table 10 ──→ "能不能裁掉某个 dimension？"（独占贡献实证：都不能裁）
```

### 论文核心数字回溯

| 摘要 / Intro 中的数字 | 来自哪张表 |
|----------------------|----------|
| 63.75% baseline conservative mutation score | Table 7 (Conservative 列) |
| 75.02% MutaKernel conservative mutation score | Table 7 |
| 99.82% MutaKernel audited score | Table 7 (Audited 列) |
| 2 of 1,107 effective mutants remain as residual gaps | Table 7 (1,107 − 1,105 = 2) |
| 222 defective kernels missed by baseline | Table 8 (Total `𝑆 \ 𝐵`) |
| 51.11% joint defect rate | Table 8 (Total Joint rate) |
| 60.23% CUDA-Agent stress-tested defect rate | Table 8 (CUDA-Agent 行) |
| 98.8% CUDA-Agent pass rate (author-reported, contrast) | 不在表格中，引用自 [10] |
| 82.5%（intro 中 "newly killed mutants accounted for by value_stress"）| 由 RQ3 stress-dimension contribution 段叙述，**未在 Table 9/10 中直接显示**，是 study dataset 内的数字 |

### 论文方法学的两个关键防御点（都靠这 4 张表撑住）

1. **三种 mutation score 口径并报**（Table 7）——抵御 "你的分母是不是有水分" 的质疑
2. **Applicable + Flagged + 独占贡献三维呈现**（Table 9 + Table 10）——抵御 "你为什么不删掉低 Rate dimension" 的消融质疑

---

## 附：表格 1-6 内容速览（前文出现但本次解读不展开）

| 表 | 标题简述 | 位置 |
|---|---|---|
| Table 1 | 16 个 mutation operator 跨 4 个 category 的分类 | §3.1 |
| Table 2 | Study dataset 的标签分布 + baseline mutation score 两种约定 | §4.1 |
| Table 3 | KernelBench mutation score 按 operator category 分类（保守口径）| §4.1 |
| Table 4 | 534 个 baseline-survived mutant 的审计结果 | §4.2 |
| Table 5 | 三种 denominator convention 下 KernelBench mutation score 对比 | §4.2 |
| Table 6 | MutaKernel 五个 deterministic dimension 的设计描述 | §5.2 |

---

*文档生成时间：2026-05-15*
