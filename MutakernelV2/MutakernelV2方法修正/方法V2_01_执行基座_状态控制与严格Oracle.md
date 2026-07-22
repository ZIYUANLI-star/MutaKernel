# 方法 V2-01: 执行基座 — 状态控制、输入隔离与严格 Oracle（M1）

> **版本**: 2.0-draft-1（2026-07-21）
> **对应代码**: `src/validation/{types,state,inputs,oracle,executor}.py`
> **上游文档**: 方法V2_00（Schema 权威）
> **对应论文位置**: Section 3.x（Differential Execution Substrate）+ Threats

---

## 一、问题背景

差分测试的全部结论都建立在一个前提上：**观测到的输出差异可以归因于被测
程序本身**。V1 执行基座在四处破坏了这个前提（拒稿后代码审计确证）：

| # | V1 缺陷 | 后果 |
|---|---------|------|
| 1 | Phase I 连续实例化 `Model` 与 `ModelNew`，不重置 RNG、不同步参数 | 带随机初始化参数的任务（L2 大量 Linear/Conv/BN）中 ref 与 mutant **权重不同**，输出差异与变异无关也被计为 kill。旁证：L2 杀死率 89.7% vs L1 49.6% |
| 2 | Phase II `sync_weights` 默认关闭；回退同步按 state_dict **形状顺序**拷贝而非按名 | 权重错位拷贝，差异归因失效 |
| 3 | 参考与候选执行共享同一 input 对象 | 任一侧 in-place 修改污染另一侧 |
| 4 | 旧 oracle 强转 dtype、扁平化部分结构、不支持的结构不 fail-closed、全局统一容差 | dtype 缺陷被掩盖；结构错误漏检；判定不可解释 |

此外，编译失败/超时/OOM/参考失败/候选崩溃被折叠成二值结果，基础设施
故障会污染缺陷计数。

## 二、本模块拟解决的问题

提供 V2 全部动态实验（Mode A 与 Mode B）共用的**唯一**成对执行基座，保证：

1. **同源性**：参考与候选（或 original 与 mutant）在构造、参数、缓冲区、
   RNG、输入四个层面严格一致，观测差异可归因；
2. **隔离性**：每次执行获得独立深拷贝输入；进程级隔离防止 JIT 崩溃/挂起
   污染全局；
3. **严格性**：Oracle 检查完整结构而非仅数值近似；不支持即 INCONCLUSIVE；
4. **三值性**：PASS/FAIL/INCONCLUSIVE + 细分失败类别（见 V2-00 §3.3 枚举）；
5. **可复放**：每个非 PASS 观测生成 fresh-container 可复放的证据包。

## 三、方法设计

### 3.1 RNG 快照与重放（RNGSnapshot）

覆盖四个 RNG 域：Python `random`、NumPy（可选）、Torch CPU、Torch CUDA
（全设备）。

```
class RNGSnapshot:
    capture(include_cuda) -> RNGSnapshot   # 深拷贝四域状态
    restore()                              # 校验 CUDA 设备数不变后恢复

@contextmanager
def replay_rng(snapshot):
    caller = RNGSnapshot.capture()
    snapshot.restore()
    yield
    caller.restore()          # 保证调用方 RNG 不被污染
```

**模型构造协议**（消除缺陷 #1 的核心）：

```
function InstantiatePair(ref_module, cand_module, init_inputs, device):
    snap ← RNGSnapshot.capture()
    ref_model  ← ref_module.Model(*init_inputs)          # 消耗 RNG
    with replay_rng(snap):                                # 回放同一初始状态
        cand_model ← cand_module.ModelNew(*init_inputs)
    report ← StrictStateSync(ref_model → cand_model)      # 双保险（3.2）
    if report is StateSyncError:
        return INCONCLUSIVE(category="state_sync_failure")
    return (ref_model.to(device).eval(), cand_model.to(device).eval())
```

RNG 回放保证"即使双方构造顺序不同也从同一初始熵出发"；严格状态同步
保证"即使双方参数化结构一致性可验证时以 ref 权重为准"。两者叠加，
任何一层失败都显式 INCONCLUSIVE，绝不静默继续。

### 3.2 严格状态同步（StrictStateSync）

**禁止**任何按顺序/按形状的模糊匹配（消除缺陷 #2）：

```
function StrictStateSync(src, dst):
    s ← src.state_dict();  d ← dst.state_dict()
    if keys(s) ≠ keys(d):            raise StateSyncError(差异键列表)
    for key in keys(s):
        校验同 kind（tensor/非 tensor）、同 dtype、同 shape、同 layout
        不符 → raise StateSyncError(key, 详情)
        d[key] ← clone(s[key])       # preserve_format
    dst.load_state_dict(d, strict=True)
    return StateSyncReport(keys_synced, tensor_values_synced)
```

键集不一致（候选用了不同参数化，如 FusedDense）时**不做兜底**：该 subject
的状态相关比较记 INCONCLUSIVE(state_sync_failure)，进入缺失报告，
而不是冒险错拷。这是 fail-closed 原则的直接应用。

### 3.3 输入隔离与逻辑值哈希（消除缺陷 #3）

```
function IsolateInputs(args):
    # 深拷贝参数树（list/tuple/dict/tensor 递归），tensor 拷贝须:
    #  - 保留非稠密 stride、storage offset、重叠视图
    #  - 保留跨参数 alias 拓扑（同 storage 的两个入参拷贝后仍共享新 storage）
    return (isolated_for_reference, isolated_for_candidate, value_hash)
```

`value_hash` = 对参数树的逻辑值（不含 storage 地址）做 SHA-256，
写入 Observation，用于反例复放的一致性校验与盲审证据包。

参考与候选各自拿到独立副本；执行后还要比较**输入的事后状态**
（检测 in-place 副作用差异）与**模型参数/缓冲区的事后状态**
（检测训练模式下的状态轨迹差异）——两者都是 candidate 可观测行为的一部分。

### 3.4 严格 Oracle（消除缺陷 #4）

比较分层进行，任何一层失败即产出结构化 mismatch 记录：

```
function StrictCompare(ref_out, cand_out, oracle_cfg) -> {EQUAL_STRICT, WITHIN_TOL, MISMATCH(detail), UNSUPPORTED}:
    1. 结构层: 嵌套容器类型/长度/键集逐位对应；不支持的值类型 → UNSUPPORTED
    2. 元数据层: shape、dtype、device 完全一致（不做任何隐式 cast）
    3. 特殊值层: NaN/Inf 的位置集合完全一致（NaN-aware）
    4. 精确层: 整数/布尔/复数逐位相等; 浮点在有效掩码上逐位比较 → exact_equal
    5. 容差层: torch.allclose(atol, rtol)（来自契约的 dtype 感知容差，
       禁止硬编码全局 1e-2）
    6. 梯度层(契约启用 backward 时): 三个冻结随机 VJP 的输入梯度同层比较
```

**关键输出对**: `exact_equal`（第 4 层）与 `within_tolerance`（第 5 层）
永远分开记录——这是 V2 判杀口径拆分（SPEC_VIOLATION ≠ EXACT_DIVERGENCE）
的数据基础（见 方法V2_06 §3）。

UNSUPPORTED → INCONCLUSIVE(oracle_unsupported)，重跑后须统计该类规模，
常见结构缺口应补 oracle 支持而非静默剔除任务。

### 3.5 三值结果与失败归因

```
Verdict := PASS | FAIL | INCONCLUSIVE
FAIL 必须满足: reference 执行有效 ∧ 差异可归因于 candidate
INCONCLUSIVE 类别（V2-00 §3.3 枚举）互斥且必填
```

判定归属表：

| 事件 | 结果 |
|------|------|
| candidate 编译失败 / import 失败 | INCONCLUSIVE(candidate_compile_error)（Mode A 中探针编译失败=stillborn，同类） |
| candidate 抛任意 `BaseException`（含 SystemExit） | FAIL(failure_kind=candidate_crash)（崩溃是候选的可观测错误行为） |
| reference 抛异常 | INCONCLUSIVE(reference_failure)，**绝不 fallback 到别的 oracle** |
| 超时 / OOM / GPU 健康检查失败 | INCONCLUSIVE(timeout / oom / gpu_health) |
| oracle 不支持的输出结构 | INCONCLUSIVE(oracle_unsupported) |

**ref-NaN 的处理（2026-07-21 定案，"合并方案"）**：V1 的 ref-NaN fallback
（用 original 输出替代 reference 作 oracle）废除，替代方案分两层：

1. **运行期（主）**: oracle 采用 **NaN 位置感知比较**（`equal_nan=True` +
   NaN/Inf 位置集合逐位对应，§3.4 第 3 层）。reference 产生 NaN 时，
   original 必须逐位置复现同一 NaN 模式才构成有效对照（A=PASS）；
   不复现 → 该轮 INVALID/作废。能据此判杀的唯一情形是
   "original 复现 ref 的 NaN 模式而 mutant 未复现"——这是强证据而非
   V1 那种 `allclose(NaN,·)=False` 的分类性假阳性。
2. **planning 期（辅）**: 契约 value-domain 子句排除已知会使 reference
   产生契约外 NaN/Inf 的输入（`ref_nan_inducing` 排除集，由 pilot 迭代
   填充），减少运行期作废轮的预算浪费。

### 3.6 进程隔离与资源防护

- 每个 CaseConfig 在独立子进程执行（`start_new_session=True`），
  超时 `killpg(SIGKILL)`；主进程零 GPU 操作；
- 子进程启动时检查 GPU 空闲显存 ≥ 512MB，
  `set_per_process_memory_fraction(0.9)`；
- 每 subject 完成后清理 stale CUDA 模块 + `empty_cache` + 健康检查
  （失败重试 1 次后 abort 并记录）；
- 编译器 stdout/stderr 全量落盘（hash 命名），不丢弃；
- 冷/热计时分离：`compile_ms` 与 `candidate_ms` 独立记录
  （供开销分析区分冷路径与稳态成本）。

### 3.7 成对执行主流程（伪代码）

```
function ExecuteCase(subject, case, contract) -> Observation:
    validate_case_against_contract(case, contract)       # 越权 → contract_violation
    (ref, cand) 或 INCONCLUSIVE ← InstantiatePair(…)
    apply_execution_context(ref, cand, case)             # dtype cast / train / rebatch…
    torch.manual_seed(case.seed)
    template ← get_inputs()
    inputs ← PolicyBank.apply(case.policy, template, case.seed)   # M2
    (in_ref, in_cand, value_hash) ← IsolateInputs(inputs)
    ref_out  ← run(ref, in_ref)          # 异常 → reference_failure
    cand_out ← run(cand, in_cand)        # 异常 → FAIL(candidate_crash)
    cmp ← StrictCompare(ref_out, cand_out, contract.oracle)
    cmp_side ← StrictCompare(post_state(in_ref, ref), post_state(in_cand, cand))
    组装 Observation（含 exact_equal、timing、value_hash）
    if verdict ≠ PASS: 生成 replay_bundle 并做一次本地复放自检
    return Observation
```

## 四、与其他模块的桥接

| 方向 | 接口 |
|------|------|
| M2 → M1 | `contract`（oracle 容差、授权 case）；`PolicyBank.apply(policy, template, seed)` |
| M6 → M1 | `ExecuteCase(subject, case, contract)`；审计模式一轮内对 (R,O) 与 (R,M) 各调用一次并额外做 O/M 精确比较 |
| M1 → M6/M7/M8 | `Observation`（V2-00 §3.3）；`CounterexampleBundle` 素材（materialized inputs + value hash） |
| M1 → M9 | replay_bundle 是盲审证据包的原料（政策中立化由 M9 负责） |

## 五、与 V1 的差异摘要

| 维度 | V1 | V2 |
|------|----|----|
| 模型初始化 | 顺序构造，RNG 漂移 | RNG 回放 + 严格按名状态同步，失败 fail-closed |
| 输入 | 共享对象 | 双侧深拷贝 + alias/stride 保真 + 逻辑值哈希 |
| Oracle | allclose + dtype 强转 | 六层严格比较，exact 与 tolerance 分离，UNSUPPORTED fail-closed |
| ref NaN | fallback 用 original 当 oracle（26% subject 触发） | 废除；NaN 位置感知比较（运行期）+ 契约排除（planning 期） |
| 结果 | 二值 | 三值 + 10 类失败归因 |
| 证据 | 无 | 每非 PASS 一个可复放证据包 |

## 六、有效性威胁与实现注意

1. **重跑数字会变**：严格 oracle 与状态控制会翻转部分历史 kill/survive，
   这是修正而非退化；先导实验须量化翻转率（思路文档 04 的 P1）。
2. `nn.Module.to(dtype)` 为 in-place，多 dtype 循环时第二轮从第一轮状态
   cast；数学上与从 fp32 直接 cast 等价，但实现须加注释并测试锁定。
3. 三个 VJP 是抽样而非全 Jacobian 证明；论文措辞限定为
   "three frozen random VJPs"。
4. 子进程模型是冷路径成本；稳态 ADRS 开销需 compile-once 会话或
   在论文中显式分开报告（不为 V2 专门开发 session runner）。
5. 候选代码与可信 oracle 同处一个 Python 信任域；V2 论文不主张
   对抗恶意候选的 reward-hacking 防护（降级为讨论）。
