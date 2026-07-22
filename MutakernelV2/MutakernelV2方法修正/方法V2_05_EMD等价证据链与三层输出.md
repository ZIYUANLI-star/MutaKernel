# 方法 V2-05: EMD 等价证据链与三层输出（M5）

> **版本**: 2.0-draft-1（2026-07-21）
> **对应代码**: `scripts/full_block12.py`（EMD 部分，已改造）+ 新增
> `src/emd/{evidence,grading,calibration}.py`（建议拆分）
> **继承关系**: Layer 0–3 的实现细节（规范化规则、4 条静态规则、
> 动态轮次结构、LLM 提示词）继承 V1 文档 `方法_PhaseI_EMD四层等价检测.md`，
> 本篇定义 V2 的角色重定位、输出重构与校准协议。
> **对应论文位置**: Section 4.x（Equivalence Evidence）+ RQ3 + Threats

---

## 一、问题背景

### 1.1 V1 EMD 的三宗罪（审稿意见的直接来源）

1. **二值标签被当作接近 ground truth**：candidate_equivalent 直接决定
   mutation score 分母（保守/乐观双口径也只是在"信或不信"之间摇摆），
   审稿人 A/B 一致质疑其启发式与主观性；
2. **实践反证了二值判定**：Phase II 在 264 个 candidate_equivalent 里
   杀掉 21 个——EMD 的"等价"判断存在系统性 false equivalent；
3. **伪概率残留**：Tier 3 的 `1 − 1/(N+1)` 置信下界在轮次非独立、
   策略定向构造的前提下不成立。

### 1.2 V2 的角色重定位（两次收缩）

- 第一次收缩：EMD 是**被评估的分诊方法**，不是 ground truth；
- 第二次收缩（"探针价值过滤器"）：在"原始内核为待验证中心、变异体为探针"
  的框架下，EMD 的职责只是**剔除没有测试价值的等价探针 + 为审计预算
  分配提供先验**。硬结论（盲区/平反/杀死）的严格性由机器可复放的反例
  承担，EMD 判错的代价被限定在预算层面。

## 二、本模块拟解决的问题

1. 采集四层等价证据并输出**结构化证据向量**（原始数据，永久保留）；
2. 由证据向量按**机器可查验的准入条件**派生有序等级（论文主口径）；
3. 派生内部排序分数 q_pre（仅用于预算分配，不出现在论文声明中）；
4. 定义**校准研究协议**（RQ3 分析贡献）：评估 EMD 证据对"探针后续被
   反证"的预测效度——置信度是被研究对象，不是被依赖组件。

## 三、方法设计

### 3.1 四层证据采集（继承 + 修正）

| 层 | 内容 | V2 修正 |
|----|------|---------|
| Layer 0 | CUDA 字符串与 Python host 分别规范化（去注释、折叠空白）后精确比对 | 仅**未规范化的源码恒等**可自动定 MACHINE_PROVEN；规范化恒等降为强证据（规范化器自身可能有 bug） |
| Layer 1 | 4 条算子感知静态规则（boundary_unreachable / dead_write / mask_no_reach / dead_host_constant）+ M3 的先验等价 node_type 标记 | 规则命中 = MACHINE_PROVEN 候选，但每条规则须附可独立复核的证明模板；规则集版本化 |
| Layer 2 | 轻量行为探测（更名 lightweight behavioral probing）：100 随机轮 + 算子定向 6 策略 × 2 seeds，original vs mutant **NaN-aware bitwise** | 执行改用 M1 基座（状态控制修正后重跑）；`tested_policies/seeds` 完整落盘（供 CSE 增量去重）；worker 超时 → INCONCLUSIVE 而非"非等价" |
| Layer 3 | LLM 审阅（喂入 Layer 0–2 完整证据链 + 算子语义 + 输入规格） | **opt-in、仅 triage**：verdict/confidence 只进证据向量，**无权改变最终等级**；prompt/model 版本化，原始响应全量留档 |

Layer 2 的去重语义（供 M6/M7 消费）：bitwise 是比 allclose 严格的比较，
Layer 2 已 bitwise 一致的 (policy, seed) 在 CSE 的 value 维度中跳过，
预算让给未测策略；但 **train/dtype/config/repeated 维度不去重**
（执行上下文不同，Layer 2 结论不可迁移）。

### 3.2 证据向量（第一层输出）

Schema 见 V2-00 §3.6 `EMDRecord.evidence`。纪律：证据向量是唯一原始
数据源，等级与分数都是它的纯函数（可离线重放派生，支持敏感性分析）。

### 3.3 有序等级（第二层输出，论文主口径）

| 等级 | 准入条件（机器可查验） |
|------|------------------------|
| EQUIVALENT_MACHINE_PROVEN | 未规范化源码恒等，或 Layer 1 规则命中且证明模板通过独立复核 |
| LIKELY_EQUIVALENT | 非 MACHINE_PROVEN，且 Layer 2 全部计划轮完成并 bitwise 一致，且无任何反证 |
| INCONCLUSIVE | Layer 2 未完成（超时/基础设施）或证据相互冲突 |
| NON_EQUIVALENT_WITNESSED | 存在已复放的行为差异见证（Layer 2 bitwise 分歧或后续 CSE 反例） |

要点：

- 等级单调性：等级只能被**新证据**改变（CSE 反例把 LIKELY_EQUIVALENT
  降为 NON_EQUIVALENT_WITNESSED = 平反）；LLM 输出永远不改等级；
- `provably equivalent` 措辞在论文中仅指 MACHINE_PROVEN；
- 统计口径：mutation score 按等级分层给出多分母视图，
  INCONCLUSIVE 永远可见，不静默剔除。

### 3.4 内部排序分数 q_pre（第三层输入）

```
q_pre = w0·[L0 规范化恒等] + w1·[L1 规则命中] +
        w2·min(1, L2_rounds/112)·[L2 无分歧] + w3·L3_confidence·[L3 判等]
```

- 权重初始拍板（如 0.3/0.3/0.3/0.1），**作用域仅限审计预算分配的排序**
  （方法V2_07 §3.2），任何论文数字不引用 q_pre 的绝对值；
- `q_pre_scope: "internal_ranking_only"` 字段强制声明。

### 3.5 校准研究（RQ3 分析贡献，有条件执行）

目的：评估"EMD 证据能多大程度预测一个存活探针后续被 CSE/审计反证"。

```
数据: 重跑后的新队列，特征 = 证据向量展开（L0/L1 布尔、L2 轮数与
      策略覆盖、L3 confidence…），标签 = 后续是否被反证
      （CSE SPEC_VIOLATION ∨ 人工审计 NON_EQUIVALENT）
切分: 按任务聚类切分 holdout（防同任务泄漏）；
      标签来源与"评估 MutaKernel 的标签"不同源复用（防循环）
模型: logistic + isotonic 校准（两者都报）
报告: AUC、Brier、可靠性图、按算子分组的秩相关；
      负结果照报（"EMD 证据预测力有限"本身是有效发现）
```

前置条件：反证负样本 ≥ 约 50（历史仅 21，重跑后评估是否达标；
不达标则只报告秩相关 + 明示样本量限制）。

### 3.6 平反回流（false-equivalent feedback）

CSE 在 LIKELY_EQUIVALENT 探针上产出 SPEC_VIOLATION 时：

1. 等级改写为 NON_EQUIVALENT_WITNESSED，`falsified_by: counterexample_id`；
2. 该样本进入校准数据集的负样本池；
3. 审计报告的 `false_equivalent_count` 按 EMD 等级分层给出
   （= V1 "264 中杀 21" 的 V2 规范化形式）。

## 四、架构与流程

```
survived 探针（重跑后的新队列）
   │
Layer 0 规范化比对 ──恒等──► MACHINE_PROVEN（终态）
   │
Layer 1 静态规则 ──命中+复核──► MACHINE_PROVEN（终态）
   │
Layer 2 轻量行为探测（M1 基座执行）
   │        ├─ bitwise 分歧 → NON_EQUIVALENT_WITNESSED（携带见证）
   │        └─ 未完成 → INCONCLUSIVE
   ▼
Layer 3 LLM triage（opt-in，只写证据向量）
   ▼
证据向量 ──纯函数──► 等级 + q_pre ──► [M7] 预算分配
                                    ──► [M9] Population B 双盲校准标签
                                    ──► [§3.5] 校准研究（RQ3）
CSE 反例 ──► 平反回流（§3.6）
```

## 五、桥接接口

| 方向 | 接口 |
|------|------|
| M5 → M7 | `EMDRecord`（等级 + q_pre + tested_policies）；等级驱动预算档位 |
| M5 → M6 | Layer 2 `tested_policies/seeds`（value 维度增量去重）；divergence 见证（tier1 replay 输入） |
| M6 → M5 | SPEC_VIOLATION 反例 → 平反回流 |
| M5 → M9 | Population B 队列（双盲等价标注）；标注者不得见 EMD 等级/LLM 答案 |
| M5 → 统计 | 分层 mutation score 多分母视图；校准研究结果 |

## 六、与 V1 的差异摘要

| 维度 | V1 | V2 |
|------|----|----|
| 输出 | 二值（+graded strict/candidate） | 证据向量 → 有序等级 → 校准研究 三层 |
| LLM | Layer 3 可撤销 candidate（影响终态） | 仅 triage，无权改等级 |
| 置信度 | 1−1/(N+1) 伪概率 | 删除；q_pre 仅内部排序；概率化只在校准研究中作为被评估对象 |
| 超时语义 | 视同非等价（污染 Tier 1） | INCONCLUSIVE |
| 平反 | 隐性（Phase II 杀 21 个后改标签） | 显式回流协议 + 分层报告字段 |

## 七、实现注意

1. 等级派生必须实现为对证据向量的**纯函数**（无 IO、无随机），
   并配套"重放派生"脚本用于敏感性分析（改规则/轮数/阈值重派生全队列）；
2. Layer 2 重跑必须走 M1 修正基座——历史 Layer 2 结果（V1 状态污染下
   产生）只作 pilot，不得作为新等级的证据；
3. LLM triage 的敏感性分析（换模型/prompt/轮数的等级不变性）在
   等级不依赖 LLM 的设计下退化为"q_pre 排序稳定性"检查，成本大幅下降；
4. `EQUIVALENT_MACHINE_PROVEN` 的静态规则证明模板须逐条落盘
   （规则、命中位置、为何行为不可达/不可传播），供人工抽查复核。
