# 方法 V2-04: RealismGuard 构造效度验证与故障分类学维护（M4）

> **版本**: 2.0-draft-1（2026-07-21）
> **对应代码**: `src/mutengine/realism_validator.py`、`scripts/validate_realism.py`
> + 新增双人编码工具
> **继承关系**: diff 分析管线（23 条领域正则 + 3 条兜底结构模式 +
> ROOT_CAUSE_TO_OPERATORS 两级映射）继承 V1 文档
> `方法_变异真实性守卫RealismGuard与算子来源.md`，本篇定义 V2 的
> 协议升级与分类学演化闭环。
> **对应论文位置**: Section 4.x（Probe Realism）+ RQ3 + Threats

---

## 一、问题背景

变异研究的构造效度命门：**"你人为设计的探针，LLM 真的会犯吗？"**
V1 的 RealismGuard 用真实迭代历史配对（失败 turn vs 最佳正确 turn 的 diff）
+ 23 条正则的根因分类回答了这个问题，但存在三个 V2 必须补的缺口：

1. **单来源自动分类，无人工校验**：正则首中即返回、优先级遮蔽、
   standalone 路径无兜底——分类准确率未知，审稿人 B 明确要求
   multi-annotator 验证；
2. **量化实验从未落盘**：扫描确认工作区内没有 RealismGuard 的结果数据，
   V1 论文只写了"informed by 1,020 validator-rejected turns"，
   覆盖率数字缺位；
3. **分类学静态**：Reviewer C 问"GPU 架构和 LLM 失败模式演化时分类学
   如何维护"——V1 只有文档级设想。

## 二、本模块拟解决的问题

1. 产出**双人编码、可复核**的探针现实性证据：每个 C/D 算子对应多少
   真实 LLM 错误、覆盖率多少、哪些真实故障类未被覆盖（`not_represented`）；
2. 把 V1 的自动正则分类降级为**预标注建议**（pre-annotation），最终标签
   由人工双编码 + 仲裁产生（对应 FSE 计划的 Population C）；
3. 建立**分类学维护闭环**：新失败模式 → 编码 → 新算子/新策略提案 →
   受控注入验证 → 版本化发布。

## 三、方法设计

### 3.1 语料构建（继承 V1，补齐冻结纪律）

| 数据源 | 构造 | 质量 |
|--------|------|------|
| 迭代历史配对（主力） | 同问题 failed turn × 最佳 correct turn → unified diff | 高（同一 LLM 真实迭代的精确 diff） |
| 最终失败 kernel（补充） | 无正确对照，standalone 源码 + 错误消息 | 低，单独分层报告 |

V2 新增纪律：语料以 (problem_id, turn, kernel sha256) 建 stable ID 清单
并冻结（含 1,020 条 validator-rejected 记录的完整枚举），抽样种子预注册。

### 3.2 编码协议（Population C，双人开放编码）

```
阶段 0  预标注: V1 的 23 正则 + 兜底结构模式跑全量 → 机器建议标签
        （只作为效率辅助，编码者界面默认隐藏，可选查看）
阶段 1  试点: 分层抽 20–30 条（按 level / 语言 / 机器建议类别分层），
        编码者 A、B 用草案码本独立编码
阶段 2  码本修订: 讨论分歧 → 修订类别定义/判据 → 重标试点 → 冻结码本
阶段 3  主编码: 冻结样本（全量或预注册分层样本）双人独立编码；
        每条输出: root_cause ∈ 码本、mapped_operators ⊆ 16 算子 ∪
        {not_represented}、multiple 允许、confidence、rationale
阶段 4  仲裁: A/B 标签锁定后，仲裁者 C 只看分歧条目
阶段 5  报告: 仲裁前 percent agreement、Cohen's kappa、混淆矩阵、
        逐类别一致率、仲裁率、未决数
```

码本初版 = V1 的 20 根因 + `not_represented` + `unknown`，
类别判据必须写成"满足什么 diff 证据才可归入"的操作性定义。

### 3.3 核心输出指标

```
RealismReport_v2:
  coverage_rate_cd     = 被 C/D 算子覆盖的真实错误占比（主指标，分数据源报告）
  coverage_rate_all    = 全 16 算子覆盖占比
  per_operator_realism = {算子: 对应真实错误数}   # 为 0 的算子触发降级评审
  not_represented      = [{root_cause, count, 描述}]  # 分类学的已知边界
  agreement            = {kappa, percent, confusion_matrix}
```

`per_operator_realism = 0` 的算子处置规则（预注册）：从"核心创新"降级为
"对照算子"或移除；`not_represented` 高频类（阈值预注册，如 ≥5 例）触发
新算子提案流程（§3.4）。

### 3.4 分类学维护闭环（回应 Reviewer C）

```
新语料（新 LLM / 新 GPU 架构 / 新数据集的失败样本）
      │  进入 §3.2 编码管线（同一码本，增量编码）
      ▼
not_represented 新高频类 X
      │
      ├─ 1. 算子提案: 定义 X 的 find_sites/apply（遵循 V1 基类协议）
      ├─ 2. 受控验证: 新算子注入探针 → 基线杀死率 + CSE 逃逸机制归类
      │      → FaultToStressMap 增补条目（可能引出新策略需求）
      ├─ 3. 现实性回验: 新算子的 per_operator_realism 必须 > 0
      └─ 4. 版本发布: operator_version、fingerprint_version、
             map_version 三者联动升级，写入变更记录
```

该闭环使分类学成为**受维护协议约束的活组件**而非一次性产物——这是
论文 Discussion 的方法论卖点。

### 3.5 与 RQ3（变异预测效度）的关系

RealismGuard 提供 RQ3 的一半证据（探针 ↔ 真实错误的**分布**对应）；
另一半（探针检出 ↔ 真实缺陷检出的**预测**对应）由 M7 的
leave-one-dataset-out 分析提供（方法V2_07 §3.6）。二者共同回答
"为什么变异研究是必要的"，允许诚实的部分负结果。

## 四、架构与流程

```
KernelBench 迭代历史 + 1,020 rejected turns（冻结清单）
        │
  配对/standalone 语料构建 ──► 机器预标注（V1 正则管线，辅助角色）
        │
  双人编码（阶段 1–4）──► 最终标签（+ agreement 统计）
        │
        ├──► RealismReport_v2（RQ3 证据、Threats 素材）
        └──► not_represented 高频类 ──► 分类学维护闭环 ──► M3 算子集演化
```

## 五、桥接接口

| 方向 | 接口 |
|------|------|
| M4 → M3 | 算子增删/降级决定（版本化）；`ROOT_CAUSE_TO_OPERATORS` 表维护 |
| M4 → M7 | per_fault_class 现实性权重（审计报告中按"现实性加权"给出敏感性视图） |
| M4 → M9 | Population C 的编码记录复用 M9 的标注/仲裁/agreement 工具链与存储格式 |
| M4 → 论文 | coverage 表（RQ3）、not_represented 边界（Threats）、闭环协议（Discussion） |

## 六、与 V1 的差异摘要

| 维度 | V1 | V2 |
|------|----|----|
| 分类方式 | 正则自动分类即终局 | 正则降级为预标注，双人编码 + 仲裁 + kappa |
| 数据状态 | 设计有、数据未落盘 | 冻结语料清单 + 预注册抽样，必须产出数据 |
| 未覆盖类 | uncovered_patterns 列表 | `not_represented` 一等类别 + 新算子触发规则 |
| 分类学演化 | 文档设想 | 操作化闭环 + 三版本联动 |

## 七、实现注意

1. 编码工作量估算：迭代配对样本数百对，双人全量编码约数十人时；
   若受限，预注册分层抽样（按 level × 语言 × 机器建议类别），
   报告抽样权重；
2. 编码者不得看到该样本对应算子在后续实验中的杀死率或论文预期
   （盲性要求同 M9）；
3. 机器预标注与人工终标的混淆矩阵本身是有价值的报告内容
   （量化 V1 自动分类的准确率，顺带回应"启发式"批评）；
4. diff 噪声（变量重命名/重构导致的行级 diff 误报）在码本中给出
   显式处理规则（允许标 `diff_noise` 弃用该样本）。
