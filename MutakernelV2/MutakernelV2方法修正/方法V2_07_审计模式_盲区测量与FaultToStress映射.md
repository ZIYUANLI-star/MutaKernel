# 方法 V2-07: 审计模式 Harness — 盲区测量、平反与 FaultToStress 映射（M7）

> **版本**: 2.0-draft-1（2026-07-21）
> **对应代码**: 重写 `scripts/run_stress_enhance.py` → `scripts/run_audit_harness.py`（建议）
> + 新增 `src/audit/{cohort,budget,ripr,mapbuild}.py`
> **上游文档**: 方法V2_03（探针）、方法V2_05（EMD）、方法V2_06（CSE）
> **对应论文位置**: Section 4（Empirical Study）+ Section 5.1（Map Derivation）

---

## 一、问题背景

审计模式是 V2 的"研究活动"侧：它消费探针、EMD 证据与 CSE，产出三样东西
——基线验证器的盲区测量（RQ 证据）、EMD 的平反与校准数据、以及
Mode A → Mode B 的唯一桥接 artifact（FaultToStressMap）。V1 对应物
（Phase II 编排器）有四个结构问题：

1. 队列继承自被状态污染的 Phase I（534 这个数本身不可信，须重跑后
   机械重导）；
2. Tier 分类把"Layer 2 worker 超时"当作"非等价证据"塞进 Tier 1
   （32 个超时探针混入，其中 22 个成为 Tier 1 存活的主体）；
3. 产出止步于 kill 计数与算子×维度矩阵，**没有逃逸机制归类这一层**——
   fault-to-stress 的因果链只存在于论文行文；
4. 审计（Task A）在检测阳性移出后进行，构成 resubstitution 循环。

## 二、本模块拟解决的问题

1. **队列派生**：从修正基座重跑的全量探针结果，按冻结规则机械导出
   审计队列（无人工挑选）；
2. **先验引导预算**：按 EMD 等级分配 CSE 搜索强度（V1 Tier 的规范化）；
3. **盲区测量与平反**：按 §V2-06 五值语义聚合，分层报告；
4. **逃逸机制归类（RIPR）**：把每个确认盲区归因到故障传播链的失败环节；
5. **FaultToStressMap 构建**：聚合 killing case 证据为版本化映射 artifact；
6. **与人工审计的正确时序**：审计队列在 CSE 结果封存后、按规则生成，
   包含全部告警 + 未决 + 阴性抽样（消除循环）。

## 三、方法设计

### 3.1 队列派生（cohort derivation）

```
输入: Phase I 重跑结果（全量 1,646 探针 × 修正基座 × 基线验证协议）
规则（冻结后不得因结果调整）:
  killed_baseline    → 关闭（基线盲区测量的分子侧证据）
  stillborn(编译失败) → 排除出分母（无信号）
  survived           → EMD（M5）
      MACHINE_PROVEN  → 关闭（等价，出分母）
      其余等级         → 进入审计队列 Q
输出: frozen_audit_cohort.json（probe_id 全列 + 每条的 EMD 等级 + 派生规则版本）
```

**禁止**：沿用历史 534 队列；把历史 Phase II/审计标签带入新队列的任何
先验字段（防泄漏，标注盲性要求）。

### 3.2 先验引导预算（Tier 的 V2 形式）

预算档位由 EMD 等级 + 见证情况决定（替代 V1 的 Tier 1/2/3 + 白名单）：

| 档位 | 准入 | value seeds/策略 | 其他维度 | 附加 |
|------|------|-----------------|---------|------|
| W（有见证） | NON_EQUIVALENT_WITNESSED 且见证可复放 | 3 seeds × 去重后全策略 | 全维标准 | tier1_replay 先行 |
| U（未决） | INCONCLUSIVE 或 LIKELY_EQUIVALENT 边缘（q_pre 排序后 50%） | 3 seeds | 全维标准 | — |
| E（高等价先验） | LIKELY_EQUIVALENT（q_pre 排序前 50%） | 5 seeds（最强挑战） | 全维标准 | 平反重点对象 |

要点：(a) **超时/基础设施不再制造档位**——INCONCLUSIVE 是独立报告类，
不冒充"非等价证据"；(b) 高等价先验反而给**更强**预算（E 档 5 seeds），
因为平反一个 LIKELY_EQUIVALENT 的信息价值最高（校准负样本 + 内部效度）；
(c) q_pre 只用于 U/E 分界排序（V2-05 §3.4 的唯一消费点）。

### 3.3 盲区测量与平反的聚合口径

```
对每个探针的 EngineResult:
  ∃ SPEC_VIOLATION →
      cse_killed = true
      若 EMD 等级 == LIKELY_EQUIVALENT → false_equivalent = true（平反）
      记录 first_violation 维度、cross_dimension_confirmations
  仅 EXACT_DIVERGENCE_ONLY → strict_equivalence_falsified = true（单列）
  全 INDISTINGUISHED/有效轮通过 → unfalsified（等级不变，措辞纪律）
报告分层:
  ① 按算子/故障类: 基线杀死率、CSE 补杀率、闭合率
  ② 按 EMD 等级: 平反数 / 各级基数（V1 "264 杀 21" 的规范化）
  ③ INVALID_INPUT / INCONCLUSIVE / not_applicable / extended_contract 全量可见
mutation score: 多分母视图（含/不含各等级），不再声明单一"审计分数"
```

### 3.4 逃逸机制归类（RIPR 归类器）

对每个确认盲区（基线漏检 + CSE SPEC_VIOLATION），沿故障传播链归因
基线失败环节。判据设计为**可机械执行的决策树**（输入 = killing case 元数
据 + EMD 证据 + 人工审计 reason_category）：

```
function ClassifyEscape(probe, engine_result, emd, audit_reason?) -> mechanism:
    kc ← first SPEC_VIOLATION 的 case
    if kc.mode == "config"                        → REACHABILITY_FAILURE
       # 默认配置下故障位点未被触及/未触边界（对应 requires_config_change）
    elif kc.mode == "train"                       → REACHABILITY_FAILURE(mode)
       # eval 固定统计量使故障路径未执行（对应 path_not_triggered）
    elif kc.mode == "repeated"                    → OBSERVATION_FAILURE(nondet)
       # 单次执行无法观察竞态
    elif kc.mode == "eval" 且 kc.parameters.dtype ∈ {fp16, bf16}
                                                   → MASKING_FAILURE(precision)
       # fp32 掩盖精度损失
    elif kc.policy ∈ 数值极端族/边界族/结构族      → ACTIVATION_FAILURE(value)
       # 默认 randn 未激活故障状态（对应 value_insensitive/predicate 边界）
    else                                           → ABSORPTION_FAILURE(tolerance)
       # 差异产生但被容差吸收（EXACT_DIVERGENCE 证据辅助判别）
补充: 人工审计 reason_category 存在时做一致性校验，
      冲突条目转人工仲裁（不静默取一边）
```

机制枚举与缺失观察轴的对应：

| escape_mechanism | 缺失轴 | 导出维度 |
|------------------|--------|---------|
| ACTIVATION_FAILURE | value_distribution | value（+定向策略族） |
| MASKING_FAILURE | dtype/precision | dtype |
| REACHABILITY_FAILURE(mode) | execution_mode | training |
| OBSERVATION_FAILURE(nondet) | repetition | repeated |
| REACHABILITY_FAILURE(config) | batch_configuration | config |
| ABSORPTION_FAILURE | oracle_strictness | exact/tolerance 分离报告 |

### 3.5 FaultToStressMap 构建

```
function BuildMap(all_engine_results, cohort) -> FaultToStressMap:
    for fault_class F:
        probes_F ← 队列中该故障类的非等价探针（含平反后）
        for case c（policy×mode 组合）:
            closure_rate(F,c) = |probes_F 被 c 杀死| / |probes_F 被 c 覆盖执行|
            sole_detector(F,c) = 仅 c 所在维度杀死的探针数
            mean_cost_ms(F,c) = 该 case 平均执行成本
        effective_cases(F) ← closure_rate 降序，附 sole_detector 与成本
        escape_mechanism(F) ← §3.4 归类的众数（多机制则列多条 entry）
        evidence ← 支撑该条目的 counterexample_id 列表
    map_version 与 operator_version / policy_bank_version 联动
```

映射的**受控验证声明**（论文 RQ 证据）：对已知非等价探针集，
map 前 k 个 effective_cases 的累计闭合率曲线 + leave-one-dimension-out
（去掉某维后闭合率损失）——替代 V1 "166/168" 的单点声明。

### 3.6 变异预测效度分析（RQ3 另一半，与 M4 互补）

```
leave-one-dataset-out: 用 D\{d} 数据集的审计结果构建 map，
预测 d 数据集上自然缺陷（M9 确认标签）的检出（按故障类对齐）；
报告按故障类的预测召回与秩相关。负结果照报。
```

### 3.7 与人工审计的时序（消除 V1 循环）

```
CSE 全部完成 → 观测日志封存（digest）
→ M9 生成队列: 全部告警（SPEC_VIOLATION）∪ 全部未决（INCONCLUSIVE/
   unfalsified 且证据边缘）∪ 阴性分层抽样（全 PASS 探针按算子×等级分层）
→ 双盲标注（标注者不见: 哪个维度报警、EMD 等级、q_pre、历史标签）
→ 标签锁定后才允许计算任何"审计确认"口径
```

**禁止**：先移出 CSE 阳性再审计余下（V1 的 368 循环）；用审计标签
回改队列或预算。

## 四、架构与流程

```
Phase I 重跑（M1 基座 + 基线协议）
   │ 冻结派生规则
   ▼
审计队列 Q（+ EMD 等级）
   │ §3.2 预算分档（W/U/E）
   ▼
[M6] CSE 三路搜索（审计模式：跨维不早停）
   │
   ├─ SPEC_VIOLATION ──► 盲区确认 + 平反（§3.3）──► [M5] 回流
   │                                └──► RIPR 归类（§3.4）
   ├─ EXACT_DIVERGENCE_ONLY ──► 单列统计
   └─ 其余 ──► unfalsified / INCONCLUSIVE 分层
   ▼
[M9] 封存后双盲审计（§3.7）
   ▼
FaultToStressMap v2（§3.5）＋ 盲区测量报告 ＋ EMD 校准数据集
```

## 五、桥接接口

| 方向 | 接口 |
|------|------|
| M7 → M6 | `plan`（case 列表 + 预算档 + 去重信息）；`run_audit(…)` |
| M7 → M5 | 平反回流触发；校准数据集（特征=证据向量，标签=是否反证） |
| M7 → M8 | `FaultToStressMap`（唯一桥接 artifact，版本化发布） |
| M7 → M9 | 封存观测日志 + 队列生成请求；接收锁定标签计算确认口径 |
| M7 → 统计 | 多分母 mutation score、闭合率曲线、leave-one-out、预测效度 |

## 六、与 V1 的差异摘要

| 维度 | V1 | V2 |
|------|----|----|
| 队列 | 继承污染的 534 | 重跑后机械派生 + 冻结规则 |
| Tier | 超时混入 Tier 1；Tier 3 白名单魔法 | EMD 等级 + q_pre 排序三档；INCONCLUSIVE 独立 |
| 产出 | kill 计数 + 算子×维度矩阵 | + RIPR 归类 + FaultToStressMap + 校准数据集 |
| 审计时序 | 阳性移出后审 368（循环） | 封存后全告警∪未决∪阴性抽样，双盲 |
| 核心声明 | 166/168、99.82% | 闭合率曲线 + leave-one-out + 审计确认口径（带 CI） |

## 七、实现注意

1. RIPR 决策树的判据依赖 killing case 元数据——CSE 必须完整落盘
   first_violation 与全部维度结果（审计模式不早停的原因之一）；
2. 多维同时杀死的探针，逃逸机制可多标（每维一条），映射条目按维聚合；
3. FaultToStressMap 的 closure_rate 分母是"被该 case 覆盖执行的探针数"
   而非全队列（applicability 差异不折算为无效）；
4. 平反样本预期仍是小样本（历史 21），§3.6 与 M5 校准的样本量声明
   必须诚实；E 档加强预算部分正是为了增加平反检验的功效。
