# V2 判杀口径与 Oracle 修正思路

> 创建日期: 2026-07-21
> 性质: 方法思路文档（非代码）。代码层实现见 `src/validation/oracle.py`、
> `scripts/_stress_worker.py`、`docs/FSE_CODE_REMEDIATION.md`。

---

## 1. 三判定 + 四类结果的统一语义

所有变异体压力测试（Mode A）统一为 A/B/C 三判定：

- **A** = ok(ref, original, x)：original 在压力输入 x 下满足参考语义
  （结构/shape/dtype/NaN-Inf 位置检查 + 容差比较，均由严格 oracle 执行）；
- **B** = ok(ref, mutant, x)；
- **C** = exact(original, mutant, x)：逐位一致（NaN-aware）。

| A | B | C | 唯一合法解释 | 进入哪些统计 |
|---|---|---|-------------|-------------|
| 通过 | 失败 | （通常不同） | **specification violation**：真缺陷 + genuine validator gap | mutation score、validator gap、一切核心统计 |
| 通过 | 通过 | 不同 | **exact divergence only**：推翻严格实现等价，不推翻契约等价 | 单独报告，不计缺陷；作为"容差内可区分行为"的量化数据 |
| 通过 | 通过 | 相同 | 等价假设的经验性支持（非证明） | 提升等价置信度 |
| 失败 | 失败 | — | 压力输入超出有效测试域，该轮 INCONCLUSIVE | 单独计数，不判杀、不降置信度 |
| 失败 | 通过 | 不同 | 变异意外修复/缓解了 original 的问题（罕见），单独分析 | 不判杀；可作为 original 自身隐藏缺陷的线索 |

（后两行即 2026-07-21 GPT5.6sol 讨论中细化的"情况四/情况五"；情况五虽不计入
validator gap，但在"待验证对象是原始内核"的 V2 框架下有独立价值——它直接提示
original 候选内核可能存在隐藏缺陷，应转入候选内核自身的缺陷核查流程。）

**历史口径必须修正的两处**：

1. `value_stress` 与 `tier1_replay` 中"allclose 通过但 bitwise 不同也判 KILLED"
   要拆成 `exact_divergence` 与 `specification_violation` 两个字段，只有后者计入
   缺陷。历史 166 个 stress kill 里凡 kill_type="bitwise" 的需重新归类——这会改
   动 166/168、99.82% 等数字，重写论文前先跑一遍数据核对影响面。
2. `config_stress` 的 NaN oracle 缺陷：历史实现在 ref 产生 NaN 而 original 有效
   时仍拿含 NaN 的 ref 与 mutant 做 allclose（必假阳）。V2 语义：config 维度下
   ref 无效 → 该轮直接 INCONCLUSIVE，不 fallback、不判杀（original 在
   off-contract batch 下正确性未验证，fallback 也不成立）。

> **2026-07-21 定案更新**：ref-NaN 的最终口径以 `方法V2_01 §3.5`（合并方案：
> 运行期 NaN 位置感知比较 + planning 期契约排除）为权威，本节以下内容为
> 决策过程记录。

## 2. ref-NaN fallback 的收紧

历史主轨道 26%（139/534）的变异体至少一次触发"ref NaN → 用 original 当 oracle"
的回退，累计 1303 次。V2 处理：

- fallback 轮次产生的 kill 一律单独标记；主结果给出 含/不含 fallback 两个口径；
- 更根本的解决：契约的 value-domain 子句应把"会使 ref 产生 NaN/Inf 的输入"划出
  有效测试域，使这类轮次在 planning 阶段就被排除或显式标记为扩展契约测试。

## 3. 三值结果与失败归因

采纳 FSE 改版的 PASS / FAIL / INCONCLUSIVE 三值语义，并强调两条纪律：

- PASS 永远不是证明，禁止 `confirmed_equivalent` 措辞，统一改为
  `unfalsified under the tested domain`；
- 编译失败、超时、OOM、ref 失败、基础设施失败必须是彼此可区分的 INCONCLUSIVE
  子类，任何一类都不得混入缺陷计数或等价证据。

## 4. 严格 oracle 相对历史 oracle 的行为变化（重跑时的预期影响）

新 oracle（`src/validation/oracle.py`）不再做 dtype 强转/扁平化，检查嵌套结构、
stride/alias 拓扑、NaN/Inf 位置，不支持的结构 fail-closed 为 INCONCLUSIVE。预期
影响：

- 历史上因 dtype 强转被掩盖的差异会新增 FAIL；
- 历史上被宽松比较放过的结构不匹配会新增 FAIL 或 INCONCLUSIVE；
- 输出含非 tensor 结构的任务可能整体变 INCONCLUSIVE——重跑后要专门统计这一类，
  必要时为常见结构补充 oracle 支持，而不是把任务悄悄剔除。

## 5. 与论文叙述的对应

- 第 1 节的四类结果表直接成为论文 Study 部分的判定语义表；
- "exact divergence only" 类的规模本身是一个有趣的观测（容差内可区分行为的普遍
  程度），可作为讨论 tolerance 选择的证据；
- INCONCLUSIVE 的所有类别保留在分母表中——这是 FSE 审稿人核查诚实性的第一入口。
