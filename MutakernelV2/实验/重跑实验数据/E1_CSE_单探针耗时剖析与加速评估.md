# E1 CSE 单探针耗时剖析:为什么重载 lane3/4 一条要 3200-3600 秒

日期:2026-07-28。数据来源:远端 `/root/mk_v2_runs/e1/cse_observations_lane{1,2,3,4}.jsonl` 的 trial 级 `timings_ms`、对正在运行的两个 worker 的实况观测(partial-snapshot 文件 mtime 逐轮计时 + `py-spy dump --nonblocking` 非阻塞栈采样 12 次)、以及远端 CPU 微基准(`nice -n 15`,纯 CPU,未触碰 GPU)。全程只读,未干扰 lane3/4 驱动。

## 结论一句话

**慢的不是 GPU,也不是严格 Oracle,而是每一轮在 CPU 上单线程重新生成十亿级元素的输入张量。** 重载探针 97-99% 的墙钟时间落在 `timings_ms` 覆盖不到的"轮外":`get_inputs()` 的 `torch.rand`(P53 单次 18.8s)、压力策略的全量值生成(25-38s)、双重 `clone_tree` 深拷贝、`_summaries` 的 5 次全量 CPU 归约(4.3s)。GPU 利用率实测 0%,两个 worker 各占约 295% CPU。

## 一、耗时拆解(观测文件实测)

轮内 = Σ trials[].timings_ms.total_ms;轮外 = wall_ms − 轮内。

### 重载 lane3(0.92 显存独占;中位 wall 3601.6s——注意这就是 3600s 超时截断值)

| 探针 | wall (s) | 轮数 | 轮内 (s) | 轮外 (s) | 轮外占比 | 轮内主项 |
|---|---|---|---|---|---|---|
| P53__const_perturb__4 | 3604.6 | 93 | 41.1 | 3563.6 | 98.9% | oracle 38.2s |
| P53__mask_boundary__0 | 3529.8 | 94 | 34.6 | 3495.2 | 99.0% | oracle 31.8s |
| P96__const_perturb__4 | 3604.9 | 87 | 42.4 | 3562.6 | 98.8% | oracle 27.7s |
| P96__index_replace__4 | 3603.2 | 88 | 42.3 | 3560.9 | 98.8% | oracle 28.8s |
| P96__launch_config_mutate__1 | 3601.2 | 90 | 42.3 | 3558.9 | 98.8% | oracle 27.5s |
| **P16__arith_replace__6(特例)** | 2833.6 | 103 | **2675.8** | 157.8 | 5.6% | **candidate 2669.4s** |

### 重载 lane4(并发 0.45;中位 wall 3199.0s)

| 探针 | wall (s) | 轮数 | 轮内 (s) | 轮外 (s) | 轮外占比 |
|---|---|---|---|---|---|
| P20__relop_replace__3 | 3594.3 | 103 | 56.1 | 3538.2 | 98.4% |
| P24__init_modify__1 | 3303.7 | 103 | 58.2 | 3245.6 | 98.2% |
| P27__launch_config_mutate__1 | 3213.4 | 103 | 53.3 | 3160.1 | 98.3% |
| P27__arith_replace__43 | 3394.2 | 103 | 55.9 | 3338.3 | 98.4% |
| P91__mask_boundary__2 | 2084.0 | 103 | 36.5 | 2047.5 | 98.2% |
| P20__arith_replace__18(oracle 偏慢例) | 3513.0 | 103 | 284.3 | 3228.6 | 91.9% |

### 轻载 lane1/2 对照(中位 wall 94.0s / 137.6s)

| 探针 | wall (s) | 轮数 | 轮内 (s) | 轮外 (s) | 轮外占比 |
|---|---|---|---|---|---|
| L1_P1__relop_replace__7 | 105.6 | 103 | 19.8 | 85.7 | 81.2% |
| L1_P12__launch_config_mutate__1 | 54.7 | 103 | 8.5 | 46.2 | 84.5% |
| L1_P10__mask_boundary__0 | 112.6 | 103 | 28.5 | 84.0 | 74.6% |
| L1_P2__relop_replace__3 | 176.4 | 103 | 11.5 | 164.9 | 93.5% |

轮内分段在所有 lane 都很小:state_snapshot/state_sync/rng_capture/input_isolation/cleanup 合计每探针 **0.1-0.5s**(输入隔离 `clone_call_inputs` 是 GPU 侧克隆,每探针累计仅 26-95ms);reference/candidate 前向每探针累计 2-10s;oracle 每探针累计 25-52s(重载,约占 wall 0.7-1.5%)。

## 二、轮外时间的归属(代码 + 实况证据)

读 `scripts/_mutant_worker.py` 与 `scripts/run_e1_probe_study.py::run_worker` 确认,每轮在 `validate_pair`(即 `timings_ms` 计时范围)之外还要做:

1. `_seed_all(seed)` + `template = get_inputs()`:**CPU 单线程 RNG 全量生成**。P53 为 128×4096×4095 = 21.5 亿元素 / 8.6GB;
2. stress 轮:`policy_fn(clone_tree(template), seed)`,其中 policy 内部**又**做一次 `clone_tree` → 每个 stress 轮两次 8.6GB 深拷贝,策略函数再全量生成对抗值并 `copy_` 回视图;
3. `_summaries()`:对输入张量做 min/max/mean/isnan.any/isinf.any 共 5 次全量 CPU 归约;
4. `_normalise_args()`:8.6GB 可分页内存 H2D 搬运(估 1-3s);
5. `_write_partial_snapshot()`:每轮覆写结果 JSON(约 50KB,毫秒级)。

每探针一次性的固定开销:子进程启动 + torch import(实测仅 1.4s,页缓存热)+ CUDA 上下文 + 双模型编译加载(TORCH_EXTENSIONS_DIR 缓存命中)≈ 每探针 20-50s。

**实况证据(运行中的 P53、P91 worker):**

- 12 次 `py-spy dump --nonblocking` 栈采样,**12/12 全部**落在输入生成链:`get_inputs`(torch.rand)、`policy_bank` 策略函数(`_sparse`/`_extreme_magnitude`/`_alternating_sign`)、`clone_tree`;0 次落在前向、oracle 或清理;
- partial-snapshot mtime 逐轮计时:P53 一个 stress 轮 wall 33.3s,其中轮内(timings_ms)仅 0.3s;P91 各 stress 轮 15.9-29.4s,轮内均 0.3s;
- `nvidia-smi` GPU 利用率 0%,两 worker 各约 295% CPU(ps 生命周期均值)。

**CPU 微基准(远端实测,nice -n 15,分量口径)**

| 操作(P53 形状 8.6GB) | 耗时 (s) |
|---|---|
| torch.rand 模板生成 | 18.8 |
| clone_tree 一次 | 1.1 |
| 策略全路径(双 clone+值生成+copy_):extreme_magnitude / sparse / alternating_sign / boundary_last_element / structured_ramp | 29.9 / 38.3 / 36.0 / 25.2 / 9.4 |
| _summaries(5 次全量归约) | 4.3 |
| 对照:torch.rand P27 形状 (6.4GB) / P91 形状 (4.3GB) / 轻载 P1 (67MB) | 16.0 / 8.8 / **0.13** |

按此模型:P53 随机轮 ≈ rand 18.8 + summaries 4.3 + H2D ≈ 25s;stress 轮 ≈ rand 18.8 + 策略路径 10-38 + summaries 4.3 ≈ 33-60s(基准机器带载、nice 降权,实况稳态 33s/轮,比例一致)。40 随机 + 54 stress(P53 跳 3 策略)≈ 2800-3600s,与观测吻合;**lane3 多数探针实为 3600s 超时截断**(P96 只完成 82-90/103 轮,靠 partial-evidence 规则判 STILL_LIKELY_EQUIVALENT),真实完整成本还要更高。

## 三、轻重载 25-40 倍差距的构成

| 分量(每轮) | 轻载 (P1/P12) | 重载 (P53/P27/P91) | 倍数 |
|---|---|---|---|
| 输入生成 torch.rand | 0.13s | 8.8-18.8s | ×70-145 |
| 策略变换+双重克隆(stress 轮) | <0.1s | 10-38s | ×100+ |
| _summaries 归约 | ~0.01s | 4.3s | ×400 |
| 轮内合计(前向+oracle+隔离) | 0.08-0.28s | 0.35-0.56s | ×2-5 |
| **每轮总计** | **0.5-1.6s** | **20-40s** | **×25-60** |

轮数相同(103),固定开销相近(20-50s),所以探针级 94-137s vs 3200-3600s ≈ ×25-40,**全部由 CPU 侧输入体积差解释**(轻载张量 MB 级,重载 4.3-8.6GB)。轻载探针的轮外占比同样高(75-93%),只是绝对值小,且其中固定启动开销占了约一半。

## 四、假设检验

| 假设 | 结论 | 依据 |
|---|---|---|
| H1 CPU 策略输入生成主导 | **成立(主导)** | py-spy 12/12 命中生成链;轮外占 wall 97-99%;微基准 rand 18.8s + 策略 25-38s/轮 |
| H2 严格 Oracle 次要 | **成立** | oracle Σ 25-52s/探针 ≈ 0.7-1.5%(P20 有一条 277s ≈ 8% 的离散例) |
| H3 每轮输入隔离深拷贝是大头 | **不成立(需修正)** | 轮内 input_isolation 是 GPU 克隆,Σ 26-95ms/探针;CPU 深拷贝确实存在但在轮外 stress 路径(双 clone_tree ≈ 2.2s/轮,~6%),大头是"生成"不是"拷贝" |
| H4 worker 重启+模型重建是显著固定开销 | **不成立(重载)** | torch import 1.4s、扩展编译缓存命中,固定 ≈ 20-50s/探针 <1.5%;对轻载占 ~30-50%,是轻载轮外占比高的主因 |
| H5 reference 在 CPU 上跑 | **不成立** | device=cuda,双模型 `.to(device)`;reference_ms 每轮 22-100ms |
| H6 轮间 empty_cache/gc 耗时 | **不成立** | cleanup Σ 52-159ms/探针;empty_cache 仅在 OOM/超时路径触发 |

**另有第二类慢探针(少数)**:P16__arith_replace__6 输入很小,但变异体前向 24.5s/次(原核 ~46ms),103 轮 candidate Σ 2669s 占 wall 94%。这是"变异改变了性能而非正确性"的探针,90s 轮级 watchdog 不触发,属设计使然。

## 五、设计代价 vs 实现低效

**方法学必需、不能省(实测代价很低)**:RNG 捕获回放 + 状态快照/同步(每探针 Σ <0.3s)、GPU 侧输入隔离(Σ <0.1s)、严格 Oracle(Σ 25-52s,~1%)、子进程隔离 + watchdog(20-50s/探针)。40 个随机轮各自独立 seed 的输入生成(P53:40×18.8 ≈ 752s)也是预注册协议的一部分,不能砍轮数。

**实现层面可优化(按收益排序)**:

1. **stress 轮模板缓存**:63 个 stress 轮只用 3 个 seed(base_seed+40+si,si∈{0,1,2}),`_seed_all(seed); get_inputs()` 对相同 seed 产出 bit 级相同的模板,却被重复生成 21 次。按 seed 缓存 3 份模板可省 51-60 次 rand ≈ **P53/P27 类省 800-1130s/探针(25-31%),P96 类(双张量)省 ~900s,P91 类省 ~450s**。改动约 15 行 worker 代码。唯一语义注意点:跳过 get_inputs 会使进入 validate_pair 时的全局 RNG 状态不同——本套内核前向全部确定性(无 dropout),输出不受影响,但 dry-run 时应比对一条探针的 trials 逐轮一致;
2. 去掉外层冗余 `clone_tree`(策略内部已克隆,模板每轮重造无需保护):省 ~1.1s/stress 轮 ≈ 60s/探针(~2%),3 行改动;
3. `_summaries` 降级(pass 轮只记 shape/dtype,fail/首末轮保留全量统计):省 ~4.3s/轮 ≈ 400s/探针(~11%),但**改变证据记录内容**,中途改动有证据链一致性风险,不建议;
4. 策略值生成移到 GPU:会改变 RNG 流 → 输入不同 → 方法学变更,**中途禁止**。

## 六、是否值得现在干预

**现状定量**:lane3 剩余 ≈ 75 探针(102 目标,已完成 ~27),速率 ~1 条/h → 约 75h;lane4 剩余 20 条 → 约 20h(并行);关键路径 lane3 ≈ **3 天**。(lane5 的 8 个内核尚未开跑,不在本次询问范围。)

**若现在应用优化 1+2**:lane3 每探针省约 15-19 分钟,剩余 75 条省 **约 17-20 小时(23-27%)**,关键路径 3 天 → 约 2.3 天;附带收益是消除大部分 3600s 截断——目前 lane3 大量探针只完成 82-94/103 轮、靠 partial-evidence 背书,提速后可拿到完整 103 轮证据。所需操作:改 worker(~18 行)→ 离线 dry-run 一条 P53 探针比对 trials → 杀掉并重启 lane3/4 驱动(逐探针 checkpoint,最多损失 2 条 in-flight ≈ 2h)→ 观测文件里前后探针的 wall_ms 口径断裂需在对账文档标注。

**建议:默认不动。** 理由:(a) 收益 ~20h,但校验 + 重启 + 口径标注本身要消耗半天且引入人为失误面;(b) 改动虽 bit 级等价于输出,但 RNG 全局状态路径有微妙差异,150 条收官阶段不值得为省 1 天承担任何证据链质疑;(c) 3600s 截断的探针已被预注册的 partial-evidence / resource-degraded 规则合法覆盖。**触发干预的条件**:如果 lane5 还要排进同一台机器导致总日程 >5 天、或论文需要"完整 103 轮"的证据完成率指标,则只上优化 1(模板缓存),预期省 17-20h,并按上述流程执行。

## 附:数据与脚本

- 取样/实况/基准脚本(仓库根,只读):`_perf_cse_sample_remote.py`、`_perf_cse_live_remote.py`、`_perf_cse_bench_remote.py` 及对应 `_perf_cse_*_drive.py`;原始输出 `_perf_cse_sample_out.json`、`_perf_cse_live_out.json`。
- 关键代码位置:`scripts/_mutant_worker.py` L473-618(轮循环,get_inputs/clone_tree/policy/_summaries/_normalise_args 均在 timings_ms 之外)、`src/validation/executor.py` L216-380(timings_ms 计时范围)、`src/stress/policy_bank.py` L45-83(policy 内部第二次 clone_tree)。

---

# 第二部分:CPU 侧输入生成能否大幅优化——可行性评估(2026-07-28 12:00)

**直接回答:能。分两档——**

- **现在就能上的"逐 bit 等价"组合**(模板缓存 + 去冗余 clone + summaries 单遍归约):重载探针 **约 2.5-2.7 倍**(墙钟 −55~63%),输出与证据记录逐 bit 不变;
- **GPU 侧生成**(单项加速 100-3700 倍,探针级约 **15 倍**,3600s → 约 240s):**当前战役与收官窗口都不可用**(RNG 流断裂,与主账本 302 条已完成探针及 witness_seed 复现语义不可比),应在 **E3 预注册前**作为基座默认落地。

微基准环境:GPU 测试以 `set_per_process_memory_fraction(0.15)` 限额、2GiB 探尺寸(峰值 ≤5GB,在跑 lane 有 15-23GB 余量),生成/变换均为带宽受限操作、按元素数线性外推;CPU 测试 `nice -n 15`。脚本 `_perf_cse_bench2_remote.py`。

## 一、候选手段逐项评估(按预期收益排序)

### 1. GPU 侧输入生成:单项 100-3700 倍,探针级 ~15 倍——但现在不能上

**实测(2GiB 探尺寸,括号内为线性外推到 P53 全尺寸 8GiB)**:

| 操作 | CPU 实测 | GPU 实测(外推 P53) | 加速比 |
|---|---|---|---|
| torch.rand 生成 | 18.8s (P53) / 9.3s (P91 半尺寸 4GiB) | 0.0025s(0.011s) | **~1700×** |
| mixed_extremes 全变换 | >64-120s(清算表 5.1.1 缩比外推,P53) | 0.0996s(0.43s) | **~150-280×** |
| extreme_magnitude 变换 | ~10s(P53,policy 路径扣除项) | 0.0052s(0.022s) | ~450× |
| 5 项 summaries 归约 | 4.41s (P53) | 0.0475s(0.20s) | 22× |
| H2D 搬运(GPU 生成后归零) | 0.21s/2GiB → ~0.9s/8.6GB | 0 | — |

探针级推算:P53 每轮从 25-33s 降到 ~1.5-2s(轮内 0.5s + 生成 <0.5s + snapshot),103 轮 + 固定开销 ≈ **240s vs 现在 3600s ≈ 15 倍**。轻载探针几乎无感(生成本来 <0.2s/轮)。

**正确性与可比性风险(定性为"当前禁用"的依据,均已实测)**:

- 同 seed 不同数:seed=123 时 CPU MT19937 头 4 个值 `[0.296, 0.517, 0.252, 0.689]`,CUDA Philox 为 `[0.067, 0.945, 0.221, 0.272]`——**输入完全不同**;
- CPU `torch.Generator` 无法驱动 CUDA 生成(实测 `RuntimeError: Expected a 'cuda' device type for generator`),即 policy_bank 现有代码在 GPU 张量上会直接崩,不存在"顺便兼容"的侥幸;
- 可比性语义(读清算表口径):终版归类规则第 1 条规定**重跑数据直接归入主账本判级类别**;FALSIFIED 记录携带 `witness_seed`,其复现语义 = (seed, 生成后端) 二元组,换后端等于换 witness;同一 kernel 的探针族(如 P53 的 10+ 条)现共享"同 seed 同输入"的横向一致性,混用后端会破坏它。所以**不仅现在的 lane3/4 不能换,收官独占窗口的重跑也不能换**——重跑要入主账本。
- 工程风险:policy_bank 21 个策略函数全部硬编码 CPU(`torch.zeros/ones/arange/full` 无 device 参数 + CPU generator),GPU 化需逐个织入 device(~100-150 行,含 `_make_policy`/`_apply_to_tensor`);fancy-index/布尔掩码操作本身全部 GPU 兼容(逐函数核过,仅 `_sparse` 的 `mask.sum().item()` 是一次 D2H 同步,可接受);VRAM 峰值增加 ~17-19GB(模板+values+mask 常驻 GPU),0.92 独占档可容,0.45 并发档需重审配额。

**适用时机:E3/E4 启动前落地,预注册声明 `generation_backend: cuda_philox`。**

### 2. 消除双重 clone:确认安全,收益小(~2%)

代码确证:`STRESS_POLICIES` 的 21 个条目**全部**经 `_make_policy` 包装,第一步即 `result = clone_tree(template_inputs)`,策略从不改写入参;worker 外层 `policy_fn(clone_tree(template), seed)` 的克隆是冗余的——template 每轮重新生成、用后即弃,无人复用。E0 的"共享输入对象污染"教训针对的是 reference/candidate 之间的隔离,由 `validate_pair` 内部的 `clone_call_inputs`(GPU 侧,每轮 0.3ms)保障,与这层无关。收益:实测 clone_tree 1.1s/次 × 54-63 stress 轮 ≈ **60-70s/探针(~2%)**;改动 1 行;逐 bit 等价(克隆不改值)。单独做不值得,搭车做。

### 3. 模板缓存(与 1/2/5 叠加重算):bit 级等价组合的主力

前提再确认(实测):同 seed 重生成逐 bit 相等(`torch.equal=True`);63 个 stress 轮只用 3 个 seed(50040/50041/50042),模板被白白重造 21 遍。叠加后的探针级推算(用实况轮节奏扣减分量):

| | 现在(实测) | 缓存+去clone+aminmax 后 | 削减 |
|---|---|---|---|
| P53 随机轮 | ~25s | 18.8(rand,不可省)+0.4(summaries)+0.9(H2D)+0.5(轮内) ≈ **20.7s** | −17% |
| P53 stress 轮 | ~33s(实况) | 0(模板缓存)+7.4(policy 值生成)+0.4+0.9+0.5 ≈ **9.3s** | −72% |
| P53 探针(40 随机+54 stress) | ~3500-3900s(3600 截断) | ≈ **1360s** | **−61%** |
| P96 探针(截断于 3600,真实 >4100s) | >4100s | ≈ **1600s** | ~−60% |

随机轮的 `torch.rand` 是 40 个独立 seed 的预注册协议,**不可缓存不可省**——这是 bit-exact 组合的天花板(P53 探针压不进 ~1300s 以下)。

### 4. 多线程/分块 CPU 生成:标记不可行

- `torch.set_num_threads` 对 `torch.rand` **无效**(实测 1 线程 9.30s vs 56 线程 11.88s——CPU RNG 填充是串行核,加线程反而略慢);顺带确认输出与线程数无关(bit 相等),即现有数据不存在线程数依赖问题;
- 分块并行(每块独立 generator,8 线程 `uniform_`):3.39s vs 9.30s,仅 **2.7 倍**,且每块独立 seed → **位流必然改变**,破坏 seed→输入 的预注册映射,与 GPU 化同等的可比性代价却只有零头收益。**两条路都不可行/不值得**;若允许改位流,直接上 GPU 生成。

### 5. summaries 单遍归约:意外的高性价比,且可做到逐 bit 等价

实测 `torch.aminmax + mean` 0.41s vs 现行 5 次全量归约 4.41s(**10.7 倍**)。等价性:min/max/mean 数值与现行逐 bit 相同(同一归约核);`has_nan = isnan(min)`(NaN 在 min/max 归约中传播);`has_inf = (max==+inf) | (min==-inf)`;唯一边界是"张量同时含 NaN 和 Inf"时需回退全量 `isinf`(生成的输入按构造无 NaN,回退实际不会触发,但代码要写上)。收益 ~4s/轮 × 94-103 轮 ≈ **370-410s/探针(~11%)**;改动 ~10 行 `_summaries`。

## 二、分场景结论

### 场景 A:当前收官阶段(lane3 剩 ~75 条、lane4 剩 ~18-20 条、lane5 ~54 条待串接)

- **综合最优方案:bit-exact 组合(2+3+5)**。全部逐 bit 等价,GPU 生成与分块并行禁用。
- **能压到多少**:重载探针 −55~63%;剩余关键路径 max(lane3≈75h, lane4+lane5≈65h) ≈ **3 天 → 约 1.2-1.3 天,净省 ~1.7-1.9 天**。附带收益:消除 3600s 截断(P53/P96 全轮完成,减少对 partial-evidence/resource_degraded 兜底的依赖),P96 独占补跑窗口大概率不再需要 5400s 例外。
- **操作成本**(参照清算表 5.1.1 已演练过的停-改-起流程):worker 改 ~30 行 + `pytest` + **dry-run 一条 P53 探针做逐字段 trials 对比**(status/oracle/summaries 数值逐 bit 一致)→ kill lane3/4 驱动 → 守护 AUTORESTART 拉起;账本逐探针原子写入,损失 in-flight ≤2 条(≤2h);全程约半天。wall_ms 口径断裂需在 manifest 加 `input_pipeline: v2_cached` 字段并在对账文档标注(若论文报告单探针成本,分段陈述)。
- **建议:值得动**。相比第一部分"默认不动"的结论,依据变了:实测叠加收益 55-63%(先前保守估计 25-31%),省 ~1.8 天 >> 半天运维成本,且 dry-run 逐 bit 对比可把正确性风险封零、5.1.1 已有同流程成功先例。**若日程完全无压力、且不接受 wall_ms 口径分段,保守替代:现在不动,把该组合放到收官独占窗口再启用(见场景 B)。**

### 场景 B:收官独占窗口(4 条 INCONCLUSIVE 摘 id 重跑 + P96 5400s 专窗)

- **可安全启用:bit-exact 组合(2+3+5)**。判据(读清算表口径得出):重跑结果**直接入主账本**(终版归类规则第 1 条),判级用同一预注册 `grade_cse_evidence`,witness 依赖 `witness_seed` 复现——因此**输入必须逐 bit 相同,"同分布"不够**;该组合恰好满足(同 seed 同生成器同位流,只是不重复算)。
- **不可启用:GPU 生成、分块并行**(位流改变,理由同上)。
- 启用后的窗口预算:P96 族 ~1600s/条(5400s 预算三倍余量,OOM 风险窗口同步缩短),P53 摘 id 重跑 ~1400s/条(带既有 policy skip)。即使场景 A 选择不动,独占窗口是无在跑账本顾虑的落地点,建议至少在此启用。

### 场景 C:E3/E4(复用 `src.validation` 基座跑外部验证器对比)

- **应在 E3 启动前落地:全栈(1+2+3+5),以 GPU 生成为主项**。E3 契约草稿未冻结、预注册未落(E3_就绪状态.md),现在声明 `generation_backend: cuda_philox` 无任何历史包袱;E3 预算制(32 次候选调用/科目)下,若输入生成仍在 CPU,重载科目会重演 E1 的"GPU 空转 97%"——B7/B8/B9 native 复现同样受益。
- **收益**:重载科目单轮 25-33s → ~1.5-2s(**~15 倍**);E1 重载探针等价物 3600s → ~240s。
- **需要的验证**(E3 冻结前完成):
  1. policy_bank device 化的单元测试:21 策略 × (CPU/GPU) 的 shape/dtype/分布断言 + CPU 路径与现版**逐 bit 回归**(保证 E1 数据可复现语义不丢);
  2. 输入一致性校验脚本:同 (seed, backend) 重放逐 bit 相等;跨 backend 明示不等并写入 manifest;witness 复现工具按 (seed, backend) 取生成器;
  3. VRAM 预算断言:生成路径峰值(模板+values+mask ≈ 2.2× 输入体积)计入 lane 配额审查;
  4. `_sparse`/`mixed_extremes`/`sparse_extreme` 三个曾被 P53 跳过的策略在 GPU 上重新计价(实测 GPU mixed_extremes 全尺寸 ~0.43s,**资源契约域违规的根因消失**,E3 可取消 per-kernel 跳过、恢复满 63 stress 轮)。

## 附 2:第二部分数据与脚本

- 微基准脚本:`_perf_cse_bench2_remote.py`(GPU 限额 0.15、2GiB 探尺寸、逐项释放)+ `_perf_cse_bench2_drive.py`;原始输出见本文件表格,全部为 2026-07-28 11:5x 远端实测。
- 依据文档:`E1_INCONCLUSIVE清算表.md`(重跑入账口径、5.1.1 缩比基准与停-改-起先例)、`E3_就绪状态.md`(E3 未冻结、`src.validation` 统一判定)。

---

# 第三部分:bit-exact 优化组合实施记录(2026-07-28 12:52-13:30 部署,导师批准)

按第二部分场景 A 方案实施:**种子模板缓存 + 去冗余外层 clone + summaries 单遍化**,GPU 生成未动(留待 E3)。全程遵守账本零丢失红线,逐 bit 等价由门禁脚本实证后才部署。

## 1. 改动清单

| 文件 | 改动 | 逻辑 |
|---|---|---|
| `scripts/_mutant_worker.py`(md5 `dc3b9d73…`) | ~55 行 | ① `_tensor_summary` 改单遍 `aminmax`+mean,`has_nan=isnan(min)`、`has_inf=(max==+inf)|(min==-inf)`,张量含 NaN 时回退全量 `isinf`(NaN+Inf 边界),非常规 dtype 回退旧路径;② stress 轮:单模板缓存(仅当模板树所有叶子均为浮点张量,守卫函数 `_all_float_tensor_tree`;混合树保持逐轮重生成),去掉外层 `clone_tree`;③ 结果新增 `input_pipeline: "cse_gen_opt_v1"` 字段 |
| `scripts/run_e1_cse_falsify.py`(md5 `86317e42…`) | 8 行 | manifest 与逐条观测记录写入 `input_pipeline` 口径标记 |
| `src/stress/policy_bank.py` | **0 行** | 未动(策略纯函数性质是缓存安全的前提,由测试锁定) |
| `tests/test_gen_opt_bitexact.py` | 新增 21 项 | summaries 新旧逐 bit 对照(16 组含 NaN/Inf/NaN+Inf/int/bool/空张量)、21 策略不改写入参、21 策略输出与模板值无关、守卫函数边界 |
| 判级/门槛/轮数/policy skip | **零改动** | `grade_cse_evidence`、42/63 阈值、`cse_policy_skip.json` 均未触碰 |

## 2. 内存账与缓存策略选择(先算账后定策)

实测 cgroup(docker):限额 **120GiB**,部署前用量 54.5GB(双 worker RSS 32.6+31.6GB)。

- **3 份/seed 缓存(否决)**:lane3(P53/P96 模板 8.6GB)+25.8GB、lane4(P27 类 6.4GB)+19.3GB 稳态新增,叠加 alternating_sign 类策略 ~26GB 瞬时峰,最坏对齐 >120GiB,会复现历史 cgroup SIGKILL;
- **单模板缓存(采用)**:代码审查 + 单元测试证实全部 21 个策略只读模板的 shape/dtype、从不读值、从不改写入参,因此一份模板(首个 stress seed 生成)即可服务全部 63 轮且逐 bit 等价;活跃集 = 旧实现的"每轮一份模板",净峰值变化 ≈ +1 份缓存 −1 次外层 clone ≈ **0**。63 轮省 60 次 get_inputs 的收益与 3 份缓存完全相同。
- seed 轮转是 policy-major(50040/41/42 每 3 轮循环),"只缓存当前 seed 一份"会全程 miss,故用户预案中的该退化档不适用;混合叶子树(非浮点张量/非张量叶)自动回退逐轮重生成(守卫)。

部署后实测:双 worker RSS 峰值瞬时 31+68=**99GB < 120GiB**,无 OOM;历史上两次 cgroup SIGKILL 的问题探针 L1_P20__relop_replace__4 本次 103 轮全程通过。

## 3. 部署前门禁(全绿后才动生产文件)

1. `py_compile`:staged worker/driver/test 通过;
2. staged 单元测试:**21 passed**(用 `MK_WORKER_PATH` 指向 staging 文件,不触生产);
3. **逐 bit 对比脚本**(`_go_bitcompare_remote.py`,CPU-only、nice 15,staging worker + 生产 policy_bank):严格按生产轮序重放新旧两条路径,**341 项张量 `torch.equal`+dtype+shape+stride 全等,336 项 summaries 逐 bit 全等,0 失败**。覆盖:P1 全尺寸(21 策略×3 seed)、P53 缩比 128×512×511(21×3)、P96 双张量+标量积结构缩比(21×3)、混合树守卫回退路径(3 策略)、**P53 全尺寸 8.6GB**(2 策略 + 全尺寸 summaries),外加缓存模板跑完所有轮后与新鲜重生成逐 bit 一致(无污染)。

## 4. 部署流程(5.1.1 先例,账本核对)

- 12:52 账本基线:obs lane3=25 行 / lane4=31 行,completed md5 `b546679a…`/`b4aa25a9…`;
- 备份:`scripts/_mutant_worker.py.bak_20260728_preopt`、`scripts/run_e1_cse_falsify.py.bak_20260728_preopt`;
- 安装 + 原地 py_compile 通过;
- **先杀驱动后杀 worker**(顺序关键:先杀 worker 会让旧驱动把半截结果判成 INCONCLUSIVE 写账):杀旧驱动 90127/90128(lane4)、127575(lane3)及 worker 145249/145848/146267/146272;
- kill 后账本核对:obs 行数与 completed md5 **与基线完全一致**(零丢失、零污染);被杀的 in-flight 探针 L1_P35__arith_replace__16 / L1_P20__relop_replace__4 均未入账;
- 守护 v10.1 于 12:56:37 AUTORESTART(≤60s),新驱动 **lane3=146300、lane4=146306**,日志健康(policy skip 生效、目标数一致 102/28 done、49/31 done),两 lane manifest 均带 `input_pipeline` 口径说明;被杀两条探针以新代码干净重跑;
- 部署后全量 pytest:**257 passed**(含新增 21 项,零回归)。

## 5. 部署后守望(2 条探针实测)

| 探针 | 判级 | wall | 轮数 | 对照 |
|---|---|---|---|---|
| L1_P35__arith_replace__16(lane3) | **FALSIFIED**(witness: structured_ramp / sub 0 / seed 50040 / output_diverged,证据链完整;该策略确定性生成,witness 与管线无关可复现) | 1620s | 71 轮时发现反例提前终止 | 无直接旧对照(证伪提前返回);CSE 对 LIKELY_EQUIVALENT 的证伪正是本阶段目标 |
| L1_P20__relop_replace__4(lane4) | STILL_LIKELY_EQUIVALENT | **1865s** | **103/103 全完成** | 同族旧中位 3300-3600s → **−45%**;且该探针历史上两次 cgroup SIGKILL,本次通关 |

无新增 INCONCLUSIVE(lane3 计数 4 为部署前存量,lane4=0);无 OOM。实测削减 −45~50%(P20 类),低于评估上限 55-63%:残余大头是**策略自身的全尺寸 randn 值生成**(bit-exact 约束下不可省,属第二部分已识别的"随机轮 rand + 策略值生成"天花板);P53/P96 类 stress 轮占比更高,预计削减更接近上限。

## 6. 新收官预计(13:30 起算)

| lane | 剩余 | 单探针预计 | 时长 |
|---|---|---|---|
| lane3(0.92) | 102−29=73 | ~1700-2000s | ~36-40h |
| lane4(0.45) | 49−32=17 | ~1900s | ~9h |
| lane5(串接 lane4 后,自动继承新代码) | ~54 | ~1900s | ~28.5h |
| **关键路径** | max(lane3, lane4+lane5) | | **~37-40h ≈ 1.6 天**(vs 优化前 ~3 天,净省 ~1.4 天) |

预计收官时点:约 2026-07-30 凌晨。

## 7. 口径标注

**2026-07-28 12:56:37(北京时间)为 wall_ms 口径断点**:此后 lane3/4/5 观测记录带 `input_pipeline: "cse_gen_opt_v1"`,墙钟时间不可与此前记录直接比较;输入张量与 summaries 逐 bit 等价、判级语义零改动,判级结果跨断点完全可比。论文若报告单探针成本需分段陈述或以标记字段筛选。

## 附 3:第三部分脚本与产物

- 门禁与部署脚本(仓库根):`_go_bitcompare_remote.py`(逐 bit 对比)、`_new_test_gen_opt_bitexact.py`(单测,已部署为 `tests/test_gen_opt_bitexact.py`)、`_go4_stage_and_gate.py`、`_go5_deploy.py`、`_go6_watch.py`;
- 本地同步:`_dl_mutant_worker.py`、`_dl_run_e1_cse_falsify.py` 已更新为部署版本;远端备份 `*.bak_20260728_preopt`。
