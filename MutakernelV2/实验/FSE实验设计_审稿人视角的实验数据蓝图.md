<!--
使用说明（投稿前删除本注释块）：
- 本文件是 MutaKernel V2 论文 Evaluation 章节的 camera-ready 底稿，英文论文文体。
- 全部待实验数据以 "?" 占位；跑完对应实验后直接填入即可。
- 已知常量（探针数 1,646、预算 32、语料构成等）已填死；依赖 stable-ID 对账或
  重跑的计数一律 "?"。
- 表格为 Markdown 形式，转 LaTeX 时逐表迁移（booktabs 三线表）。
- 内部设计依据（基线取舍理由、成本预估、审稿关切映射）见同目录
  《附_实验设计内部说明_非论文稿.md》，不随论文发布。
- 数据落盘：重跑数据 → 重跑实验数据/；新增实验数据 → 补充实验数据/。

修订记录 v1.1（2026-07-21，回应外部审稿意见）：
- P1 真值偏差：新增 Cov（发现池覆盖率）与 Reĉ（Horvitz–Thompson 加权总体召回）
  双指标，Reĉ 为 headline；Table 4/6/7/11 表头联动修改；
- P2 数据泄漏：§5.1.1 新增 Data separation 段（开发/校准/冻结测试三分 +
  留一语料 + map 先于 C2–C5 执行冻结）；RQ4 分类学覆盖率改为
  开发集（描述性）/ 留出集（评估性）双报告；
- P3 契约主观性：§5.1.1 新增 Contract governance 段（自动提取、双人审核、
  观测前冻结、版本化变更、保守 INCONCLUSIVE）；
- P4 移植公平性：§5.1.2 新增 Port fidelity 段（原生数据集复现、逐条对齐、
  作者联系记录、native/port 永不同行）；
- P5 证明范围匹配：RQ3b 改为 proof-scope-matched 口径，范围外告警单列；
- P6 规模裁剪（按 18 页单栏预算）：正文保留 RQ1–RQ5 + 成本；
  形式化范式对比表、L3/backward、B10 LLM 自审、定向比例敏感性、
  native 无限预算、ADRS 修复案例全部移附录（§5.8 明示清单）。
- 人工标注时序确认：标注在全部检测器结果封存后进行（与 §5.1.4 协议一致）；
  唯一先行项是 20–30 条 pilot 码本校准（不看正式队列，不违反封存）。

修订记录 v1.2（2026-07-21，回应外部审稿第二轮）：
- 容差混杂（最大剩余问题）：§5.3 新增 Oracle unification 段——主对比全部
  测试型验证器共享同一判定管线；新增 2×2 因子分解行（B1 legacy / B1u 统一
  oracle / M-legacy / M-full），headline 效应定义为 B1u→M-full（oracle 固定），
  oracle 效应单独报告、永不并入主张；消融阶梯基准从 B1 改为 B1u；
- 全通过样本主动缺陷搜索：§5.1.4 区分两类审计任务——告警项做反例确认，
  全 PASS 抽样项做限时独立缺陷搜索（源码+reference+契约+可运行环境+
  可自构输入，? 分钟/内核预算，缺陷标签须附可复现契约内反例）；
- 抽样功效：§5.1.5 全 PASS 层样本量由预注册功效分析确定（检测联合漏检率
  ≥?% 的 ?% 功效），全阴性样本只能给联合漏检率上界；HT 与 Hájek 并报，
  披露抽样概率/权重/FPC/设计效应/有效样本量；
- Reĉ 显著性：设计加权配对 bootstrap 差值为主，McNemar（对 Cov）降为辅助；
- 聚类单位：明确为跨语料的底层 benchmark 任务标识（同任务多候选并簇，
  逐字节重复候选去重标记）；
- RQ5 交叉拟合：C1 上评价 site-directed 改为任务级 k 折交叉拟合
  （同任务探针不跨折），部署 map（全 C1 构建、C2–C5 前冻结）在自然语料
  上二次评价，两者一致才声明映射预测力；非交叉拟合曲线作为上界进附录；
- RQ3 定位：C6 明示为"完全独立标签下的补充可信度证据"，报 Clopper–Pearson
  精确区间 + 逐 bug 明细 + 漏检根因；
- 文字一致性：语料数（八）、章节路标（5.2–5.6 答 RQ）、主表/附录行划分、
  C1 真值列（注入类别非缺陷标签）、machine-proven 的证明方式与降级规则。

修订记录 v1.3（2026-07-22，对比/消融拆表）：
- 应作者要求，原 Table 6（外部基线与自构建消融阶梯混排）拆为两张独立表：
  Table 6（Comparison with external validators：B1/B1u 锚点 + B7–B9 + B11 +
  M-full/M-dir）与 Table 7（Ablation study：2×2 oracle 因子分解 B1/B1u/
  M-legacy/M-full + 消融阶梯 B3–B6 + M-dir）；
- 原 Table 7–12 顺延为 Table 8–13，全文引用同步更新（§5.1.2、§5.3、§5.4、
  §5.5、§5.6）；等 GPU 时间表（新 Table 8）的占位行按外部/消融两组拆行；
- §5.1.3 与 Table 3（flip-rate）未做任何改动。

修订记录 v1.4（2026-07-22，Table 3 口径重构，填入 E0 先导真实数据）：
- E0 先导（单张 A800，串行，180 探针配对重跑 + 逐内核未变异原始对照）完成，
  §5.1.3 翻转率叙述段与 Table 3 按真实结果重写：
- 主指标改为同环境配对翻转率（legacy vs corrected 背靠背，隔离 substrate 效应）；
  对历史标签的比较（混杂 GPU 更换）降为次要口径单独一句报告（4/27，全部
  killed→survived）；
- 可分析层 L1-stateless：27/60 对照双臂通过，配对翻转 0/27（Wilson 95% CI
  [0, 12.5]%），如实填入；
- stateful 两层（L1/L2 各 60 探针）对照全部未通过，根因诊断为 corrected 基座
  两个完备性缺口（严格按名状态同步拒绝键名不一致的候选；隔离输入验证的显存
  放大在大张量科目上 OOM），非科目故障、非架构不兼容——两层保持 "?"，
  注明 state-sync gap，待缺口修复后重测；
- 原表 "L2, stateless" 行删除（历史探针总体中不存在 L2 stateless 科目）；
  "killed→survived / survived→killed" 两列在配对口径下合并为 "Paired flips"；
- 新增对照发现：legacy 在 A800 上对 26/67 个未变异原始内核判杀（历史上同协议
  通过），corrected 仅确认 2 个、其余拒绝背书返回 INCONCLUSIVE——作为 legacy
  判定环境敏感性的证据写入正文；
- 硬件口径：明确 E0 先导在单张 A800 上进行，主研究硬件（RTX 3090/H20）叙述不变。

修订记录 v1.5（2026-07-22，回应外部审稿第三轮：跨语料矩阵与表格角色净化）：
- 新增 Table 6a（验证器 × 真实语料覆盖率矩阵）：B1u/B7/B8/B9/B11/M-full/M-dir
  逐列展示在 C2–C5（CUDA-L1、AI-CUDA-Engineer、TritonBench-G、CUDA-Agent）
  各语料缺陷池上的 Cov，末列 Overall 与聚合表对齐；单元格填
  "检出数/该语料池规模（百分比）"；每语料 precision/TTFC/成本与每语料 Reĉ
  移附录；显著性检验只在聚合层计算（避免小格多重比较）；
- 原 Table 6 更名 Table 6b（Aggregate comparison with external validators），
  结构不变；
- 原 Table 7 "Ablation study" 更名 "Controlled baselines and design-increment
  decomposition"（受控基线与设计增量分解），正文及 §5.1.2 中 "ablation
  study/ablation ladder" 提法同步改为 "design decomposition / strengthening
  ladder"，"ablation" 名义保留给 Table 12 的 leave-one-dimension-out；
- 原 Table 8 等 GPU 时间表拆为 Table 8a（外部方法组，逐验证器成行）与
  Table 8b（构造配置组，逐配置成行）；
- 表号策略：采用 6a/6b、8a/8b 子表编号，Table 9–13 编号不变；§5.1.2、§5.3、
  RQ2 答案框、§5.8 附录清单引用同步更新。

修订记录 v1.6（2026-07-22，硬件口径改为 A800）：
- 应作者决定，主研究硬件由 RTX 3090 + H20 改为 NVIDIA A800（80 GB）单一
  架构（与 E0 先导同环境）；
- §5.1.3 硬件句同步更新；headline 反例"第二架构复放"改为"独立重放其
  replay bundle 验证"；E0 先导括注简化（先导与主研究同硬件）；
- §6 External 威胁改为单架构限制声明（tensor-core 代码路径、SM 数敏感的
  launch 配置等架构相关行为留待复制研究），"跨架构判定一致率 ?%" 项删除。

修订记录 v1.7（2026-07-22，外部资源核验后的语料替换与删除）：
- 背景：外部资源获取核验（见 补充实验数据/外部资源获取清单.md）确认两个语料
  实际不可得——C5 CUDA-Agent 生成内核未公开（仓库仅有 agent 环境模板与训练
  任务集，模型权重亦未发布）；C7 Kuiper 100 个验证 L1 内核未推送到任何公开
  渠道（已穷尽官方仓库全部分支、作者仓库、Zenodo 工件）。替代检索结论：
  Kevin-32B/KernelLLM/AutoTriton 只发布模型权重不发布生成内核；
  ProofWright、Volta（Dubey et al.）、Gimlet 等价检查器均未开源工件——
  形式化验证内核无可用替代。
- C5 替换：CUDA-Agent → KernelBench-samples（ScalingIntelligence/
  kernelbench-samples，Apache-2.0，KernelBench 论文官方发布的 frontier-model
  生成内核档案，含官方 harness 判定；收集框取 baseline_eval 贪心通道，
  与 C2 的 RL 生成、C3 的 agentic 进化生成构成不同生成机制）；Table 1 行、
  Table 4 行、Table 6a 列、RQ1 答案框、§5.3 叙述同步改名，引用改为 [31]；
  自然语料仍为四个，C2–C5 编号与提法不变；
- C7 删除：Table 1 删行（C8 编号不动，语料总数句 eight → seven）；§5 RQ3
  表述改为单一第三方语料（seeded-bug）；§5.4 标题改单数、删除 "Formally
  verified kernels (C7)" 小节与 Table 10；RQ3 答案框删除 proof-carrying
  假阳性部分；§5.1.2 与 §5.7 删除 "Kuiper 内核作为 RQ3 真值" 分句（Kuiper/
  ProofWright/Dubey 作为范式对比文献引用保留）；§6 "two external ground-truth
  corpora" 改单数；
- 表号顺延：原 Table 11–13 → Table 10–12（Table 10 已删），正文引用同步
  （§5.1.2、§5.3、§5.5、§5.6）；文件头历史修订记录按惯例不追改；
- §5.1.3 与 Table 3（E0）未做任何改动。

修订记录 v1.8（2026-07-22，E0 run5 完成，Table 3 全部真实数据落盘）：
- 基座两个完备性缺口修复后，run5 重跑 153 探针（含 stateful 两层全量）正常
  完成，与 run4 复用的 27 个 L1-stateless 配对结果合并；
- Table 3 全部 "?" 填充：L1-stateless 49 对照通过/1 翻转（2.1%），L1-stateful
  11/0（0.0%），L2-stateful 3/1（33.3%，n 小 CI 宽如实报告），合计 63 对照
  通过、62 可配对、2 翻转（3.2% [0.9, 11.0]）；
- 两个翻转方向互补：legacy 漏杀（stateless）与 legacy 误杀（L2 stateful，
  corrected 下 5/5 trial 一致）；
- 叙述段改写：核心证据为对照失败——legacy 判杀 26/50 未变异原始内核
  （L2 达 17/21），corrected 实测反驳其中 15 个、确认 7 个为 A800 架构不
  兼容、4 个 INCONCLUSIVE；另 8 个对照因参数非双射（7）与异步非法 launch
  配置（1）正当 INCONCLUSIVE；
- §6 Internal 威胁的 "?" 同步填充（3.2% 配对翻转 + 26/50 对照失效）；
- 次要口径（vs 历史标签，混杂 GPU 更换）：9/62 翻转（6 杀→存，3 存→杀）。

修订记录 v1.9（2026-07-23，E2 对账定稿 + 环境版本落盘）：
- Table 1 语料计数填充（included/collected，冻结裁剪规则 v1.0：源码非空、
  任务可定位、规范化去重、语言符合声明）：C2 85/241、C3 227/229（任务级，
  每任务一个内容寻址 accepted 代表）、C4 184/184、C5 1,720/1,724；
  表题注改为 included/collected 口径说明；
- Table 1 后新增 C2 构成说明段：141/156 排除项为纯 PyTorch 级优化
  （无自定义 CUDA kernel，超出内核验证范围），15 为规范化重复；
  四语料间规范化源码哈希零交集；
- §5.1.3 版本号填充：PyTorch 2.1.2、CUDA 12.1、Triton 2.1.0；
  Table 2 B11 注明 Compute Sanitizer 2023.1.1；
- 依据：E2 对账定稿报告（补充实验数据/E2_对账定稿报告.md），
  freeze digest 与逐条排除原因随 artifact 发布。

修订记录 v1.10（2026-07-27，Table 10 Killed 列与 stillborn 脚注填充）：
- 数据源：E1 基线阶段（已对账关账）baseline_observations.jsonl（1,646 条、
  probe_id 唯一，远端与本地副本 md5 一致：2ce2e0d542e2e7ae096c7d998595a836）；
  按 operator_category 分类计数与 Probes 列 757/702/178/9 逐类吻合；
- Killed 列填充：A 186、B 337、C 13、D 3、Overall 539；脚注 stillborn
  （compile-failed）填 105（A 56、B 3、C 45、D 1）；
- 其余列保持 "?"：Machine-proven eq. 待静态证明定版（基线账面现值 9，全部
  在 A 类，暂不预填）、Witnessed non-eq. 与 INCONCLUSIVE 待 CSE 收官、
  Detection rate 待人工盲审；分母处理（stillborn 之外尚有
  excluded_control_failed 441 等排除类）留待 Detection rate 定版时一并定稿。

修订记录 v1.11（2026-07-28，全量数据核实 + equiv 收官分母填充）：
- 独立复算核实（全部一致，未改动任何已填数字）：
  - Table 3 及 §5.1.3/§6 E0 叙述：远端 e0_run5_s0/observations.jsonl（153 条）
    与 e0_run4_s0 复用的 27 个 L1-stateless 配对合并为 180 条，逐格复算
    49/1、11/0、3/1、63/62/2 及全部 Wilson CI 吻合；对照 26/50（L2 17/21）、
    15 反驳/7 架构不兼容/4 INCONCLUSIVE、8 个 INCONCLUSIVE 对照细分
    （7 StateSyncError 非双射 + 1 异步 invalid configuration）、次要口径
    9/62（6 杀→存、3 存→杀）均与原始数据一致；
  - Table 10 Killed 列与 stillborn 脚注：远端 e1/baseline_observations.jsonl
    （1,646 条，md5 与本地副本一致 2ce2e0d5…）按 operator_category 复算，
    Probes 757/702/178/9、Killed 186/337/13/3=539、stillborn 105
    （A56/B3/C45/D1）、excluded_control_failed 441、机器证明账面 9（全 A，
    dead_host_constant 规则）逐项吻合；
  - Table 1：本地 collection_frames 逐行复算 C2 85/241（141 language_mismatch
    +15 duplicate）、C3 227/229（候选级 26,970/28,227）、C4 184/184、
    C5 1,720/1,724，跨语料 stable_id 交集全 0，sha256 与 freeze 一致；
  - Table 2 与 §5.1.3 版本号：远端实测 PyTorch 2.1.2+cu121 / CUDA 12.1 /
    Triton 2.1.0 / Compute Sanitizer 2023.1.1 / A800 80GB；E0/E1 协议常量
    （5 draws、atol=rtol=1e-2、seed 42）与 run manifest 一致；
  - C1 行 90 内核（L1 63 + L2 27）、≤3/（kernel,operator）对、1,646 探针
    经 probe_manifest.json 与观测复核。备注：16 算子分类学中
    reduction_reorder（C 类）在 90 内核上命中 0 位点，观测实例化 15 个算子；
    正文 "16 operators" 为分类学口径，保留不改。
- 填充（唯一一处，依赖已收官的 E1 equiv 阶段）：§5.5 反例搜索句的
  LIKELY_EQUIVALENT 分母 453——来源 e1/equiv_summary.json（merged：
  global+lane0–3+lane4_requeue，completed 527 = 基线幸存 527，分布
  LIKELY_EQUIVALENT 453 / WITNESSED_NON_EQUIVALENT 73 / INCONCLUSIVE 1），
  与 cse_lane_plan.json 输入池一致；被反例证伪计数（现刻 3/453，判级
  303/453 未收官）保持 "?"。
- 交叉拟合闭合率 73% 为 interim 下界（CSE 未收官），维持 "?" 不填。

# 5 Experimental Evaluation

We evaluate MutaKernel through five research questions:

- **RQ1 (Natural false acceptance).** Among real LLM- and RL-generated GPU
  kernels accepted by the de-facto standard validator, how many contain
  independently confirmed, in-contract defects?
- **RQ2 (Budget-matched effectiveness).** Under equal candidate-execution
  and equal GPU-time budgets and a *unified oracle*, how does MutaKernel
  compare against the baseline validators of Table 2 in confirmed-defect
  recall, alarm precision, and cost—and how much of any difference is
  attributable to inputs versus oracle strictness?
- **RQ3 (External validity).** How do all validators perform on an
  *independently labeled* corpus whose ground truth was established by a
  third party: a public seeded-bug corpus?
- **RQ4 (Controlled fault study).** Which fault classes does the baseline
  validator systematically miss on controlled fault probes, and does
  mutation-derived evidence predict the detection of natural defects?
- **RQ5 (Component contribution).** Which of MutaKernel's components—the
  five stress dimensions and site-directed policy selection—contribute to
  detection, and at what cost?

Section 5.1 describes subjects, baselines, infrastructure, the
ground-truth protocol, and metrics; Sections 5.2–5.6 answer RQ1–RQ5;
Section 5.7 reports cost; Section 5.8 delimits appendix material;
Section 6 discusses threats.

## 5.1 Experimental Setup

### 5.1.1 Subject Corpora

Table 1 summarizes the seven corpora. All corpora, task contracts, and
inclusion rules were frozen—with content-addressed stable IDs—*before* any
V2 validator outcome was observed.

**Table 1. Subject corpora.** (Subject counts read "included /
collected" under the frozen v1.0 inclusion rules—non-empty source,
resolvable benchmark task, non-duplicate after normalization, language
as declared; the frozen, content-hashed manifests and per-item exclusion
reasons are released with the artifact.)

| ID | Corpus | Language | Subjects | Role | Ground truth |
|----|--------|----------|---------:|------|--------------|
| C1 | Mutation probes over 90 validator-accepted KernelBench L1/L2 kernels (16 operators, first-order, ≤3 per kernel–operator pair) | CUDA | 1,646 | RQ4, RQ5 | Replayed non-equivalence witnesses + blinded equivalence audit (injected fault class/site known by construction, but not itself a defect label) |
| C2 | CUDA-L1 [26] public kernels | CUDA | 85 / 241 collected | RQ1, RQ2 | Blinded human audit |
| C3 | AI-CUDA-Engineer [22] archive | CUDA | 227 / 229 collected (task-level; one content-addressed accepted representative per task) | RQ1, RQ2 | Blinded human audit |
| C4 | TritonBench-G [25] kernels | Triton | 184 / 184 collected | RQ1, RQ2 | Blinded human audit |
| C5 | KernelBench-samples [31] frontier-model generations (official greedy-baseline release) | CUDA | 1,720 / 1,724 collected | RQ1, RQ2 | Blinded human audit |
| C6 | gpuemu seeded-bug corpus [Sarkar 2026] | Triton/CPU | 26 ops: 16 correct controls + 10 seeded bugs | RQ3 | Public third-party labels |
| C8 | KernelBench L3 / backward-enabled bounded sample | CUDA | ? (8–10 tasks) | Generalization (appendix) | Same as C1 |

The CUDA-L1 inclusion rate deserves note: 141 of the 156 excluded
entries contain no custom CUDA kernel at all—they are PyTorch-level
optimizations (backend flags, autocast, CUDA-graph capture) and thus
fall outside the scope of kernel validation; the remaining 15 are
normalized duplicates. Across the four natural corpora, no two included
subjects share a normalized source hash.

Every task carries a versioned *correctness contract* (schema v1) that
fixes valid shapes and variable dimensions, supported dtypes, value domain,
layout and aliasing assumptions, execution modes, determinism requirements,
and a dtype-aware oracle tolerance (float32: atol=rtol=1e-4; float16 and
bfloat16: 1e-2, following the vendored KernelBench implementation). A test
outside the contract can never produce an in-contract defect report; such
findings are reported separately as extended-contract observations.

*Contract governance.* Contracts are not an author degree of freedom.
Each contract is extracted automatically from the benchmark's own
artifacts wherever possible—the reference implementation, its
`get_inputs`/`get_init_inputs` specification, and published task
documentation—with only the boundary clauses (value domain, variable
dimensions, determinism) requiring human judgment. Two authors reviewed
every contract independently (clause-level agreement: ? %; disagreements
resolved before freezing), and all contracts were content-hashed and
frozen *before any V2 validator outcome was observed*. Later amendments
are version-controlled with recorded reasons and never applied
retroactively to already-sealed runs. When a case's contract status
cannot be established, its outcome is INCONCLUSIVE and is never counted
as a defect.

*Data separation.* To prevent leakage between design and measurement,
the corpora play fixed, disjoint roles: the operator taxonomy and the
stress-policy library were designed on the *historical pilot data and the
1,020-generation development corpus* (V1); all calibration constants
(directed-budget fraction, evidence-grade thresholds) were fixed on a
pilot subset of C1; the fault-to-stress map used by M-dir is derived
solely from the C1 probe study and frozen—content-hashed—*before* any
C2–C5 subject is executed. C2–C5 serve exclusively as test corpora and
contributed to no design or calibration decision. Taxonomy-coverage
claims (Section 5.6) are therefore reported twice: on the development
corpus (descriptive only) and on the *held-out natural defects confirmed
in C2–C5* (the evaluative number), complemented by leave-one-corpus-out
analysis. Within C1, any evaluation of an artifact derived from C1
itself (the fault-to-stress map in Section 5.6) uses task-level
cross-fitting so that no probe is judged by a map its own task helped
build.

### 5.1.2 Compared Validators

Table 2 lists the thirteen validator configurations. The primary
equal-budget evaluation is reported in three tables with disjoint roles:
the *per-corpus coverage matrix* (Table 6a) crosses every external
validator with the four natural corpora individually, the *aggregate
external comparison* (Table 6b) contains the third-party tools B7–B9,
the separately metered B11, the descriptive anchors B1/B1u, and our two
configurations, and the *design-increment decomposition* (Table 7)
contains the constructed ladder B3–B6 together with the oracle factorial
rows (Section 5.3); B2 is the Triton-side native anchor reported on its
compatible corpus, and B10 appears in the appendix. Rows B3–B6 form a
strengthening ladder of *simple but strengthened* baselines that
isolates, in order, the contribution of test volume, value-distribution
diversity, classical boundary-value analysis, and execution-context
diversity; they reuse individual policies from our library but apply
them uniformly, without fault-class priors or dimension composition.
(The term *ablation* is reserved for the leave-one-dimension-out
experiment of Table 11, which removes components from M-full; Table 7
instead builds the baseline up one design ingredient at a time.) External tools are
run in two modes: *native* (the tool's own protocol on its compatible task
interface, pinned to the commit in our artifact) and *port* (the tool's
testing idea re-implemented on our frozen subjects under matched budgets);
the two modes are never merged in one table row.

**Table 2. Compared validators.** Budget-matched rows execute exactly 32
candidate invocations per subject.

| ID | Validator | Protocol summary | Mode |
|----|-----------|------------------|------|
| B1 | KernelBench default [31] | 5 random draws, fixed shape/dtype, allclose, eval mode | Native (vendored) |
| B2 | TritonBench default [25] | Same family as B1, Triton side | Native |
| B3 | IID compute-matched | 32 i.i.d. random draws, default context | Constructed |
| B4 | Diversified values | 8 value distributions × 4 seeds, undirected | Constructed |
| B5 | Boundary-value suite | 8 boundary policies × 4 seeds (BVA [19, 30]) | Constructed |
| B6 | Dtype/mode/config grid | fp16/bf16, train(+backward), batch ∈ {1,4,16,64} | Constructed |
| B7 | robust-kbench [23] | Multi-init × multi-input × fwd+bwd, statistical output filters | Native + port |
| B8 | KernelBenchX [40] | Standard/outlier/boundary inputs, dtype-aware oracles, 176 Triton tasks | Native + port |
| B9 | Seeded differential fuzzing [Sarkar 2026] | Op-schema-aware sampling, fp64 CPU reference, per-(op,dtype) calibrated tolerances | Port |
| B10 | LLM self-review | Frontier-LLM code review, no execution | Separately costed (appendix) |
| B11 | Compute Sanitizer (NVIDIA, 2023.1.1) | memcheck/racecheck/synccheck/initcheck instrumentation | Native, alarm types reported separately |
| M-full | MutaKernel (full suite) | 21 stress policies + dtype/train/repeat/config dimensions, three-way differential oracle | Ours |
| M-dir | MutaKernel (site-directed) | Static site fingerprint → fault-to-stress map lookup; 70% directed + 30% general budget | Ours |

*Port fidelity.* For every ported baseline we (i) release the complete
port source, (ii) document a clause-by-clause alignment with the original
protocol, marking unsupported features explicitly rather than silently
dropping them, and (iii) validate the port by re-running it on the
original tool's *native* dataset and comparing against the tool's
published results (reproduction deltas: B7 ?, B8 ?, B9 ?; details in the
artifact). We contacted the original authors of B7–B9 for protocol
confirmation (correspondence status recorded in the artifact). Native
rows are reported under the tools' own unrestricted protocols in the
appendix and are never merged with budget-matched port rows.

Note that MutaKernel's *online* validation path executes no mutant:
M-dir consumes only a static source fingerprint (a millisecond-scale
regex/AST scan) and the offline-derived, frozen fault-to-stress map.
Mutation analysis appears exclusively in the offline study of RQ4.

Formal-verification approaches—Kuiper [Martínez et al. 2026], ProofWright
[6], and the equivalence checker of Dubey et al. [15]—are compared on
guarantees, scope, and cost in Table 12 rather than in executable rankings,
since they prove different properties than tolerance-based differential
testing.

### 5.1.3 Implementation and Hardware

All dynamic experiments run on the same execution substrate: paired
construction with RNG-state replay, strict name-exact state
synchronization, per-execution deep-cloned inputs, a structure-, dtype-,
and NaN-position-aware oracle, and three-valued outcomes
(PASS/FAIL/INCONCLUSIVE with ten failure categories). Infrastructure
failures can therefore never inflate defect counts. Every non-PASS
observation produces a replay bundle that re-executes in a fresh container;
every headline counterexample is additionally verified through an
independent replay of its bundle before being reported. All experiments
run on NVIDIA A800 (80 GB) GPUs with PyTorch 2.1.2, CUDA 12.1, and
Triton 2.1.0; run
manifests record
commits, environment fingerprints, contract/policy/operator versions,
seeds, and budgets.

Before the main study we quantified the effect of the corrected substrate
with a paired pilot (E0), run serially on a single A800 of the same
setup: 180 historical probes, stratified by task level × reference
statefulness × historical verdict group, were re-executed back-to-back
under the legacy and the corrected substrate with the identical V1
baseline protocol (5 random draws, atol = rtol = 1e-2, seed 42), together
with a per-kernel un-mutated original-kernel control that both substrates
must pass for a probe to enter the paired statistics. Three observations
shape the full rerun. First, where both substrates handled the subject,
verdicts are stable: 2 of 62 analyzable probes flipped (3.2 %; Wilson 95%
CI [0.9, 11.0] %; Table 3). Both flips are individually diagnostic rather
than noise: in one, the legacy harness failed to kill a probe whose
divergence the corrected substrate witnesses in its first trial; in the
other—on an L2 stateful subject—the legacy harness killed a probe whose
outputs agree in all five paired trials under the corrected substrate,
i.e., a legacy false kill. Second, the dominant substrate effect appears
in the controls, not in the flips: the legacy harness killed 26 of 50
un-mutated original kernels (L2: 17 of 21)—kernels that had passed this
very protocol in the historical study—whereas the corrected substrate
demonstrated that 15 of the 26 in fact pass, confirmed 7 as genuine
architecture incompatibilities of this GPU, and returned INCONCLUSIVE on
4; legacy verdicts on stateful subjects are therefore untrustworthy
wholesale, which motivates the full rerun reported in this paper. Third,
the exclusions concentrate in the stateful strata (11 and 3
control-passing probes): beyond the legacy control failures and the
architecture-incompatible subjects, 8 controls were INCONCLUSIVE under
the corrected substrate—7 because candidate and reference admit no state
bijection (the candidate folds or re-parameterizes weights), which the
substrate refuses to guess by design, and 1 because an
architecture-invalid launch configuration surfaces asynchronously in the
oracle (the probe population contains no L2 stateless subjects).

**Table 3. Paired verdict flips under the corrected execution substrate
(E0 pilot, single A800).**

| Stratum | Sampled probes | Controls passed | Paired flips | Flip rate (95% CI) |
|---------|---------------:|----------------:|-------------:|-------------------:|
| L1, stateless reference | 60 | 49 | 1 | 2.1 % [0.4, 10.9] |
| L1, stateful reference | 60 | 11 | 0 | 0.0 % [0.0, 25.9] |
| L2, stateful reference | 60 | 3 | 1 | 33.3 % [6.1, 79.2] |
| Overall (analyzable) | 180 | 63 | 2 | 3.2 % [0.9, 11.0] |

One control-passing stateless probe returned INCONCLUSIVE in the
corrected arm and is excluded from the flip denominators (62 analyzable
probes overall). Secondary comparison against the historical labels
(confounded with the GPU change and reported for completeness only):
9/62 verdicts flipped (6 killed→survived, 3 survived→killed).

### 5.1.4 Ground-Truth Protocol

*Alarm-level and subject-level labels are produced by an independent,
blinded human audit; no LLM decision and no detector verdict is ever used
as ground truth.* After all validator executions were sealed (digest
chain), we constructed an audit queue containing (i) the union of all
alarms from all thirteen validators, (ii) every execution-unresolved
subject, and (iii) a preregistered stratified random sample of all-PASS
subjects (strata: corpus × language × task level; n = ? per stratum). Each
item was presented under a neutral ID with a policy-neutral evidence
bundle (materialized input tensors, self-contained replay program, contract
text) that conceals which validator, policy, or seed produced it. Two
GPU-qualified annotators labeled independently; disagreements were settled
by a third adjudicator only after both primary labels were locked.
Pre-adjudication agreement was ? % (Cohen's κ = ?; per-label confusion in
the artifact).

The two item kinds receive different tasks. For *alarmed* items the
annotators confirm or reject the presented counterexample. For sampled
*all-PASS* items—where no counterexample exists to review—the annotators
perform an **independent, time-boxed defect search**: each receives the
candidate source, the reference, the full contract, and a live GPU
environment; may construct arbitrary in-contract inputs, vary in-contract
dtypes/configurations, repeat executions, and apply static reasoning or
standard tooling; and has a fixed budget of ? minutes per kernel. They
never see any validator's outcome for the item. A defect label requires
submitting a replayable in-contract counterexample; otherwise the item is
labeled `NO_DEFECT_FOUND`, which throughout this paper means "no defect
found within the search budget," never a proof of correctness—the
population-recall estimator (Section 5.1.5) is therefore itself a lower-
bound construction, and we report the search budget alongside it.

### 5.1.5 Metrics and Statistical Analysis

A defect population defined as "the union of what the compared detectors
found" would bias recall toward detectors that contribute more alarms to
the audit pool. We therefore separate two measures with different
epistemic status:

- **Discovered-defect coverage (Cov).** Let **A** be the set of
  audit-confirmed in-contract defective subjects discovered by *any*
  compared validator. Cov(v) = |confirmed alarms(v) ∩ A| / |A|. This is
  exactly computable but conditional on the discovered pool; we never
  present it as population recall.
- **Estimated population recall (Reĉ).** The audit design is a
  stratified probability sample: *all* alarmed subjects are audited with
  inclusion probability 1, and all-PASS subjects are audited via
  preregistered stratified random sampling with known per-stratum
  inclusion probabilities (strata: corpus × language × task level;
  probabilities, weights, and finite-population corrections disclosed in
  the artifact). The total number of defective subjects is estimated
  with both the Horvitz–Thompson estimator and the Hájek ratio estimator
  (reported side by side; design effects and effective sample sizes in
  the artifact), and
  Reĉ(v) = confirmed detections of *v* / estimated population defect
  count, with stratified-bootstrap confidence intervals. Reĉ is the
  headline recall metric in RQ1/RQ2; defects surfaced only through the
  all-PASS search (missed by every validator) enter its denominator.
  All-PASS strata are *sized by a preregistered power analysis*: n = ?
  per stratum gives ? % probability of observing at least one
  jointly-missed defect if the true joint miss rate is at least ? %; an
  all-clean sample is accordingly reported as bounding the joint miss
  rate, never as demonstrating its absence.
- **Precision** = audit-confirmed alarms(v) / alarms(v);
- **TTFC**: median and p95 time to first counterexample (wall-clock,
  cold-compile time reported separately);
- **Cost**: GPU-seconds per confirmed defect.

The clustering unit for all inference is the *underlying benchmark task*,
not the candidate kernel: C2–C5 contain multiple candidates generated for
the same KernelBench task (sharing the reference, input specification,
and often the same failure-triggering conditions), so candidates of one
task—across corpora—fall into one cluster. Byte-identical candidate
duplicates across corpora are deduplicated and flagged. Descriptive
counts are still reported per candidate; confidence intervals and tests
are task-clustered. We report task-clustered bootstrap 95% confidence
intervals (10,000 replicates) and, for Cov—whose per-subject pairing is
exact—paired exact McNemar tests with Holm correction against M-full and
odds ratios as effect sizes; Reĉ comparisons use design-weighted paired
bootstrap differences with their own intervals, and McNemar serves only
as auxiliary evidence for the population-level claims. INCONCLUSIVE
outcomes remain visible in all denominators; headline results are
accompanied by sensitivity bounds that count all INCONCLUSIVE cases
first as defects and then as non-defects. All metrics, exclusion rules,
sampling probabilities, and the two hypotheses tested in RQ2/RQ5
(H1: M-full ≥ B3 at equal budget; H2: M-dir ≥ M-full at equal budget)
were frozen before the primary runs.

## 5.2 RQ1: Natural False Acceptance

Table 4 reports, per corpus, how many baseline-accepted kernels carry an
audit-confirmed in-contract defect. We report *lower bounds*: absence of a
discovered counterexample is not evidence of correctness, and population
prevalence is only estimated for the audited strata with sampling weights.

**Table 4. Confirmed false acceptance among baseline-accepted kernels.**
The estimated rate applies Horvitz–Thompson weighting over the audit
sampling design (Section 5.1.5) and therefore accounts for defects found
in the all-PASS sample that every validator missed (n = ?).

| Corpus | Baseline-accepted | Stress-flagged | Audit-confirmed (alarmed) | Confirmed in all-PASS sample | Estimated defect rate [95% CI] |
|--------|------------------:|---------------:|--------------------------:|-----------------------------:|-------------------------------:|
| CUDA-L1 | ? | ? | ? | ? | ? % [?, ?] |
| AI-CUDA-Engineer | ? | ? | ? | ? | ? % [?, ?] |
| TritonBench-G | ? | ? | ? | ? | ? % [?, ?] |
| KernelBench-samples | ? | ? | ? | ? | ? % [?, ?] |
| Total | ? | ? | ? | ? | ? % [?, ?] |

Of the ? stress-flagged subjects, the audit confirmed ? (precision ? %),
rejected ? as reference/oracle artifacts, and left ? INCONCLUSIVE.
? flagged subjects depended solely on a dtype or batch configuration
outside the task's frozen contract; these are reported as
extended-contract findings and excluded from Table 4. Confirmed defects
distribute over fault classes as shown in Table 5, with
? and ? the two most frequent classes; this distribution is compared
against the natural error distribution of the RealismGuard corpus in
Section 5.6.

**Table 5. Fault-class distribution of confirmed defects.**

| Fault class | Confirmed defects | Share | Detected only under (dimension) |
|-------------|------------------:|------:|--------------------------------|
| F-EPS (epsilon handling) | ? | ? % | ? |
| F-STAB (numerical stabilization) | ? | ? % | ? |
| F-PREC-ACC (accumulator precision) | ? | ? % | ? |
| F-SYNC (synchronization) | ? | ? % | ? |
| F-BOUND (boundary/masking) | ? | ? % | ? |
| ... (remaining 11 classes) | ? | ? % | ? |
| Not represented in taxonomy | ? | ? % | — |

> **Answer to RQ1.** An estimated ? % (95% CI [?, ?]) of baseline-
> accepted public kernels contain confirmed in-contract defects
> (Horvitz–Thompson estimate over the audit sampling design; ? defects
> were confirmed among alarmed subjects and ? in the all-PASS sample),
> ranging from ? % (TritonBench-G) to ? % (KernelBench-samples). The de-facto
> validation standard materially overstates the correctness of generated
> kernels.

## 5.3 RQ2: Budget-Matched Validator Comparison

Table 6a reports, for each of the four natural corpora individually, how
much of that corpus's confirmed-defect pool every validator covers;
Table 6b aggregates the same comparison across corpora and adds
precision, latency, and cost, all under the equal-invocation budget
(32 candidate calls per subject); Table 7 decomposes MutaKernel's gain
under the same budget through controlled baselines; Tables 8a and 8b
repeat the external and constructed groups, respectively, under equal
GPU time. Native, unrestricted protocols are reported in the artifact
appendix and never mixed with budget-matched rows.

*Oracle unification.* A stricter tolerance alone can inflate apparent
detection power. Every test-based validator in the primary comparison
therefore shares the identical judging pipeline: the same reference
implementation, contract input validation, dtype-aware tolerances
(Section 5.1.1), NaN/Inf handling, and output-structure comparison; only
the *inputs and execution contexts* differ across rows. Native rows in
the appendix retain their own oracles. To decompose input-generation
effects from oracle effects, Table 7 includes a 2×2 factorial: the
baseline input suite and the MutaKernel suite are each judged under both
the legacy oracle (atol=rtol=1e-2, as shipped with KernelBench-style
harnesses) and the unified dtype-aware oracle (rows B1, B1u, M-legacy,
M-full). The detection gain reported for MutaKernel is the
input/dimension effect (M-full vs. B1u); the oracle effect (B1u vs. B1,
M-full vs. M-legacy) is reported separately and never folded into the
headline claim.

*Per-corpus view.* An aggregate number can mask large heterogeneity
across generation systems and kernel languages—a method that dominates
on KernelBench-samples but degrades on Triton subjects would look deceptively
uniform in a pooled table. Table 6a therefore crosses every external
validator with the four natural corpora individually: each cell reports
discovered-defect coverage on that corpus's confirmed-defect pool, so
the table answers directly whether MutaKernel's advantage—and each
baseline's strength—holds per generation system and per language, or is
driven by a single corpus. Because per-corpus pools can be small,
per-cell task-clustered bootstrap intervals are given in the artifact
and significance tests are computed only at the aggregate level
(Table 6b); the full per-corpus breakdown of precision, TTFC, cost, and
per-corpus Reĉ appears in the appendix.

**Table 6a. Per-corpus discovered-defect coverage (equal budget: 32
candidate invocations/subject).** Each cell reports "confirmed defects
detected / confirmed-defect pool of that corpus (percentage)"; the
Overall column equals the Cov column of Table 6b. B2 (TritonBench
default) applies only to TritonBench-G and is reported as a native
anchor in the appendix breakdown of that column.

| Validator | CUDA-L1 | AI-CUDA-Engineer | TritonBench-G | KernelBench-samples | Overall |
|-----------|--------:|-----------------:|--------------:|-----------:|--------:|
| B1u KernelBench inputs, unified oracle | ?/? (? %) | ?/? (? %) | ?/? (? %) | ?/? (? %) | ? % |
| B7 robust-kbench (port) | ?/? (? %) | ?/? (? %) | ?/? (? %) | ?/? (? %) | ? % |
| B8 KernelBenchX (port) | ?/? (? %) | ?/? (? %) | ?/? (? %) | ?/? (? %) | ? % |
| B9 Seeded fuzzing + fp64 ref (port) | ?/? (? %) | ?/? (? %) | ?/? (? %) | ?/? (? %) | ? % |
| B11 Compute Sanitizer | ?/? (? %) | ?/? (? %) | ?/? (? %) | ?/? (? %) | ? % |
| **M-full** | **?/? (? %)** | **?/? (? %)** | **?/? (? %)** | **?/? (? %)** | **? %** |
| **M-dir (site-directed)** | **?/? (? %)** | **?/? (? %)** | **?/? (? %)** | **?/? (? %)** | **? %** |

Across the four corpora, M-full's margin over the strongest external
baseline ranges from ? points (?) to ? points (?), and its coverage
?(does / does not) drop on the Triton-language corpus relative to the
CUDA corpora—?(supporting / qualifying) the claim that the advantage is
stable across generation systems and languages rather than an artifact
of one dataset.

**Table 6b. Aggregate comparison with external validators (equal
budget: 32 candidate invocations/subject).** Cov: coverage of the
discovered-defect pool; Reĉ: estimated population recall
(Horvitz–Thompson weighted; Section 5.1.5). † marks differences vs.
M-full significant at α=0.05 after Holm correction (exact McNemar,
computed on Cov, whose pairing is exact).

| Validator | Cov | Reĉ [95% CI] | Precision | Median TTFC (s) | GPU-s / defect |
|-----------|----:|-------------:|----------:|----------------:|---------------:|
| B1 KernelBench (5-call anchor, legacy oracle) | ? | ? [?, ?] | ? | ? | ? |
| B1u KernelBench inputs, unified oracle | ? | ? [?, ?] | ? | ? | ? |
| B7 robust-kbench (port) | ?† | ? [?, ?] | ? | ? | ? |
| B8 KernelBenchX (port) | ?† | ? [?, ?] | ? | ? | ? |
| B9 Seeded fuzzing + fp64 ref (port) | ?† | ? [?, ?] | ? | ? | ? |
| B11 Compute Sanitizer | ?† | ? [?, ?] | ? | ? | ? |
| **M-full** | **?** | **? [?, ?]** | ? | ? | ? |
| **M-dir (site-directed)** | **?** | **? [?, ?]** | ? | ? | ? |

B10 (LLM self-review) is reported in the appendix with token-based
costing; B11 alarms cover memory/race properties and are additionally
broken down by alarm type in the artifact.

Against the strongest external baseline (B?), M-full's advantage is
? points (OR = ?, p = ?). On fault classes where B9's fp64 reference
outperformed our tolerance oracle (?), a combined configuration
(MutaKernel policies + fp64 reference) reached ? coverage, which we
adopt/reject (?) as a design refinement.

Table 7 isolates *why* MutaKernel outperforms: the 2×2 oracle factorial
(B1, B1u, M-legacy, M-full) separates oracle strictness from input
effects, and the constructed ladder B3–B6 strengthens the baseline one
ingredient at a time. It is deliberately *not* labeled an ablation:
Table 7 builds the baseline *up* toward MutaKernel to show that the gain
is not attributable to budget, tolerance, or generic test diversity
alone, whereas the ablation proper (Table 11) removes components *from*
M-full to establish per-dimension necessity. The shared rows B1u,
M-full, and M-dir permit direct reading across Tables 6a, 6b, and 7.

**Table 7. Controlled baselines and design-increment decomposition
(equal budget: 32 candidate invocations/subject).** Columns and
significance marking as in Table 6b.

| Configuration | Cov | Reĉ [95% CI] | Precision | Median TTFC (s) | GPU-s / defect |
|---------------|----:|-------------:|----------:|----------------:|---------------:|
| B1 KernelBench inputs, legacy oracle (5-call anchor) | ? | ? [?, ?] | ? | ? | ? |
| B1u KernelBench inputs, unified oracle | ? | ? [?, ?] | ? | ? | ? |
| B3 IID-32 | ?† | ? [?, ?] | ? | ? | ? |
| B4 Diversified values | ?† | ? [?, ?] | ? | ? | ? |
| B5 Boundary values | ?† | ? [?, ?] | ? | ? | ? |
| B6 Dtype/mode/config grid | ?† | ? [?, ?] | ? | ? | ? |
| M-legacy: MutaKernel inputs, legacy oracle | ? | ? [?, ?] | ? | ? | ? |
| **M-full** | **?** | **? [?, ?]** | ? | ? | ? |
| **M-dir (site-directed)** | **?** | **? [?, ?]** | ? | ? | ? |

The 2×2 factorial separates the two effects: unifying the oracle alone
(B1→B1u) changes coverage by ? points, and enriching inputs alone under
the legacy oracle (B1→M-legacy) by ?; the headline input/dimension
effect (B1u→M-full, oracle held fixed) is ? points. The strengthening
ladder then decomposes that effect: raising the budget alone (B1u→B3) recovers
? points; undirected value diversity (B3→B4) adds ?; boundary and
context diversity (B4→B5/B6) add ? and ?; fault-directed composition
(B5/B6→M-full) adds ?; and per-subject site direction (M-full→M-dir)
changes coverage by ? points while reducing median TTFC by ? %.

The equal GPU-time analysis mirrors the two-table split: Table 8a
answers whether external tools close the gap when granted the same wall
clock rather than the same call count, and Table 8b answers the same
question for the constructed configurations of the decomposition.
M-full and M-dir are repeated in both sub-tables as the common anchor.

**Table 8a. Equal GPU-time comparison, external group (? s/subject wall
budget).**

| Validator | Cov | Reĉ [95% CI] | Defects found in budget |
|-----------|----:|-------------:|------------------------:|
| B7 robust-kbench (port) | ? | ? [?, ?] | ? |
| B8 KernelBenchX (port) | ? | ? [?, ?] | ? |
| B9 Seeded fuzzing + fp64 ref (port) | ? | ? [?, ?] | ? |
| **M-full** | **?** | **? [?, ?]** | ? |
| **M-dir (site-directed)** | **?** | **? [?, ?]** | ? |

**Table 8b. Equal GPU-time comparison, constructed-configuration group
(? s/subject wall budget).**

| Configuration | Cov | Reĉ [95% CI] | Defects found in budget |
|---------------|----:|-------------:|------------------------:|
| B3 IID random testing | ? | ? [?, ?] | ? |
| B4 Diversified values | ? | ? [?, ?] | ? |
| B5 Boundary values | ? | ? [?, ?] | ? |
| B6 Dtype/mode/config grid | ? | ? [?, ?] | ? |
| **M-full** | **?** | **? [?, ?]** | ? |
| **M-dir (site-directed)** | **?** | **? [?, ?]** | ? |

> **Answer to RQ2.** Under matched budgets MutaKernel attains an
> estimated population recall of ? % vs. ? % for the strongest simple
> baseline and ? % for the strongest external validator, and its
> per-corpus coverage leads on ?(all four / ? of four) natural corpora
> (Table 6a), spanning both CUDA and Triton subjects; ? % of its
> advantage is attributable to fault-directed composition rather than
> test volume or generic diversity. Site direction preserves/improves
> (?) recall while cutting time-to-first-counterexample by ? %.

## 5.4 RQ3: External Validity on an Independently Labeled Corpus

**Seeded-bug corpus (C6).** Table 9 reports detection on gpuemu's 26-op
corpus, whose 10 seeded LLM-style bugs and 16 correct controls were
labeled and released by a third party; no label in this experiment was
produced by us. Given the corpus size, we position this experiment as
*complementary credibility evidence* under fully independent labels—not
as a standalone effectiveness result: we report exact binomial
(Clopper–Pearson) intervals, itemize every bug (detected by which
validators, under which case), and root-cause every miss.

**Table 9. Third-party seeded-bug corpus: recall and false positives.**

| Validator | Bugs detected (/10) | Controls falsely flagged (/16) |
|-----------|--------------------:|-------------------------------:|
| B1 | ? | ? |
| B3–B6 (best) | ? | ? |
| B7 / B8 / B9 (ports) | ? / ? / ? | ? / ? / ? |
| M-full | ? | ? |
| M-dir | ? | ? |

The site fingerprint mapped ? of the 10 seeded bugs to a fault class in
our taxonomy; the remaining ? (e.g., ?) fall outside it and are counted
as `not represented`, consistent with the taxonomy boundary reported in
Section 5.6.

> **Answer to RQ3.** On ground truth established entirely outside this
> paper, MutaKernel detects ?/10 seeded bugs with ?/16 control false
> positives—?(consistent with / better than / worse than) the strongest
> external baselines, corroborating the audited results of RQ1–RQ2.

## 5.5 RQ4: Controlled Fault Study

**Baseline blind spots.** We regenerate all 1,646 first-order fault
probes and re-execute the baseline validator under the corrected
substrate. Table 10 reports per-category detection with the equivalence-
uncertain population made explicit: ? probes are machine-proven
equivalent—a label reserved for two machine-checkable arguments,
byte-identical (unnormalized) source or one of four versioned static
rules whose proof template is released and human-spot-checked; anything
weaker is at most LIKELY_EQUIVALENT—? carry a replayed non-equivalence
witness, and ? remain INCONCLUSIVE after the blinded equivalence audit
(κ = ?).

**Table 10. Baseline detection on controlled probes (rerun).**
<!-- "Probes" 列为生成期常量（同源码+同算子+同降采样种子下确定性可复现），
     已按 V1 生成记录核验（757+702+178+9=1,646；details/*.json 逐条计数）。
     V2 重新生成后须复核一致；其余列全部依赖修正基座重跑与盲审，不得预填。 -->

| Operator category | Probes | Killed | Machine-proven eq. | Witnessed non-eq., survived | INCONCLUSIVE | Detection rate (witnessed) |
|-------------------|-------:|-------:|-------------------:|----------------------------:|-------------:|---------------------------:|
| A: Arithmetic | 757 | 186 | ? | ? | ? | ? % |
| B: GPU parallel | 702 | 337 | ? | ? | ? | ? % |
| C: ML numerical | 178 | 13 | ? | ? | ? | ? % |
| D: LLM patterns | 9 | 3 | ? | ? | ? | ? % |
| Overall | 1,646* | 539 | ? | ? | ? | ? % |

*minus 105 stillborn (compile-failed) probes excluded from all denominators.

Escape-mechanism classification of the ? confirmed blind spots attributes
? % to value-activation failures, ? % to precision masking, ? % to
mode-reachability, ? % to nondeterminism-observation, and ? % to
configuration-reachability (Figure ?), grounding the five stress
dimensions in measured failure modes rather than design intuition.
Counterexample search additionally falsified the equivalence hypothesis
for ? of 453 probes the evidence pipeline had graded LIKELY_EQUIVALENT
(? %), quantifying the residual risk of heuristic equivalence detection;
only specification violations—never tolerance-conforming bitwise
divergences (? cases, reported separately)—count toward any of these
figures.

**Predictive validity.** Using leave-one-corpus-out analysis, fault-class
detection rates measured on probes predict natural-defect detection on the
held-out corpus with Spearman ρ = ? (per-class recall correspondence in
the artifact). Because the 16-operator taxonomy was *designed on* the
1,020-generation development corpus, we report its coverage of that
corpus (? %, double-coded, κ = ?) as descriptive only; the evaluative
taxonomy-coverage number is computed on the *held-out* natural defects
confirmed in C2–C5 (Section 5.1.1, data separation): ? % mapped onto the
taxonomy, ? % explicitly `not represented` (dominated by ?).

> **Answer to RQ4.** The baseline validator misses ? % of witnessed
> non-equivalent probes, concentrated in ML-numerical (? %) and
> LLM-pattern (? %) fault classes; probe-measured blind spots ?(do / do
> not) predict natural-defect detection (ρ = ?), supporting mutation
> analysis as a diagnostic instrument ?(and / but not) as a proxy for
> natural-defect recall.

## 5.6 RQ5: Component Contribution

**Dimension necessity.** Removing one dimension at a time from M-full
(Table 11) reduces discovered-defect coverage by ? (value), ? (dtype),
? (training), ? (repetition), and ? (configuration) points; ? % of kills
are independently confirmed by ≥2 dimensions.

**Table 11. Leave-one-dimension-out (equal budget redistributed).**

| Configuration | Cov | Δ vs. M-full | Sole-detector defects lost |
|---------------|----:|-------------:|---------------------------:|
| M-full | ? | — | — |
| − value | ? | −? | ? |
| − dtype | ? | −? | ? |
| − training | ? | −? | ? |
| − repetition | ? | −? | ? |
| − configuration | ? | −? | ? |

**Site direction.** Because the fault-to-stress map is itself derived
from the C1 probe study, evaluating site direction on C1 with that same
map would be circular. The controlled evaluation therefore uses
*task-level cross-fitting*: the 90 source tasks are partitioned into
? folds; for each fold, a map is built from the probes of the remaining
tasks only and evaluated on the held-out fold's probes (probes of one
task never straddle folds), and results are pooled over folds. Under
cross-fitting, the map closes ? % of witnessed blind spots within the
first k=8 planned cases (closure curve in Figure ?; the non-cross-fitted
curve, an upper bound, is in the appendix). The deployment-grade map
(built on all of C1 and frozen before any C2–C5 execution) is then
evaluated on the natural corpora: on C2–C5 subjects whose fingerprints
contain the relevant sites, M-dir reaches the first counterexample ? ×
faster than M-full at equal budget; on the ? % of confirmed defects
whose fault class had no fingerprint site (fingerprint false negatives),
the 30% general budget recovered ? of ?. We claim predictive power for
the map only where the cross-fitted and frozen-corpus results agree.
Varying the directed-budget fraction (50/50, 90/10) changes coverage by
at most ? points (appendix).

> **Answer to RQ5.** All five dimensions contribute non-redundantly
> (leave-one-out losses of ?–? points; cross-confirmation ? %), and
> site-directed selection converts static fingerprint evidence into a
> ? × median speed-up of defect discovery at ?(no / minor) recall cost.

## 5.7 Cost

Validating one candidate costs a median of ? s (p95 ? s) in deploy mode
(early exit on first violation) and ? s in full-audit mode; cold JIT
compilation accounts for ? % and is amortized across cases. Validating
1,000 candidates costs ? GPU-hours (~$? on-demand cloud), versus ? for
robust-kbench (port) and ? for B9. The budget–recall curve (Figure ?)
saturates at ? invocations, which we adopt as the recommended deployment
budget. A guarantees/scope/cost comparison with formal-verification
approaches (Kuiper, ProofWright, Dubey et al.)—whose proofs cover
different properties than tolerance-based differential testing—appears
in the appendix; the two paradigms are complementary.

## 5.8 Scope of the Main Study and Appendix Material

To keep the evaluation focused, the main body reports five experiment
families: natural false acceptance (RQ1), the budget-matched comparison
(RQ2), external validity (RQ3), the controlled fault study (RQ4), and
component contribution with cost (RQ5, 5.7). The following analyses are
preregistered but reported in the appendix/artifact: unrestricted native
protocols of all external tools; the full per-corpus breakdown of the
external comparison (per-corpus precision, TTFC, cost, and Reĉ
complementing the coverage matrix of Table 6a); the L3/backward
generalization sample (C8); LLM self-review (B10);
directed-budget-fraction sensitivity; the formal-verification paradigm
comparison; and the ADRS-loop repair case study.

## 6 Threats to Validity

**Construct.** Confirmed-defect labels come from a blinded two-annotator
audit (κ = ?); we mitigate residual subjectivity with the external
ground-truth corpus of RQ3, whose labels we cannot influence, and with
the Horvitz–Thompson recall estimator, which frees the headline recall
from the "union of detector alarms" pool (Section 5.1.5)—though the
estimate's variance depends on the all-PASS sampling rate (n = ? per
stratum). Contracts encode judgment in their boundary clauses; the
governance protocol (automatic extraction, dual review, pre-observation
freezing, versioned amendments, conservative INCONCLUSIVE) bounds this
threat, and all contract sources and diffs are released. Fault classes
outside the 16-operator taxonomy (? % of held-out natural defects;
cross-stream ordering, multi-kernel interactions) bound the reach of
site-directed selection; the general budget and the documented taxonomy-
maintenance protocol partially compensate. **Internal.** Paired re-execution under the corrected substrate flipped
3.2 % of analyzable verdicts (2/62) and invalidated the legacy harness's
un-mutated original-kernel controls on 26 of 50 kernels (Table 3); all
reported numbers come from the rerun, and legacy results are cited only
as pilot evidence. The taxonomy, policy library, and fault-to-stress map were
developed and frozen on data disjoint from the test corpora
(Section 5.1.1), and controlled evaluations of the map use task-level
cross-fitting (Section 5.6); the primary comparison holds the judging
oracle fixed across validators and decomposes oracle strictness from
input effects (Section 5.3), so tolerance choices cannot masquerade as
detection power. Ported baselines carry a residual implementation-
fidelity risk that we bound by native-dataset reproduction (deltas: ?)
and released port code. Reference implementations are trusted; rounds
where the reference itself fails its contract are INCONCLUSIVE and never
counted. **External.** Results cover KernelBench-style single-kernel
tasks in CUDA/Triton on a single GPU architecture (NVIDIA A800, Ampere);
architecture-dependent behaviors—tensor-core codepaths, SM-count-
sensitive launch configurations—may differ on other GPUs, and
cross-architecture portability of verdicts is left to replication; the
L3/backward sample (C8, appendix) probes but does not establish
generality beyond this population. **Conclusion.**
All primary comparisons are budget-matched, task-clustered, and
Holm-corrected; INCONCLUSIVE sensitivity bounds are reported alongside
every headline number.

## Data Availability

The artifact contains the frozen corpora manifests and contracts, all run
manifests and observations, replay bundles for every non-PASS observation,
the blinded audit queues, raw annotations, adjudications and agreement
statistics, the fault-to-stress map, all analysis scripts, and a smoke
mode that reproduces every table from released data on a single GPU.
