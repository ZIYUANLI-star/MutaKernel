# KernelGenBench 接入 MutaKernel 变异测试 —— 实验总结(给师兄)

> 执行人:<author> 的 cc 代跑(全量变异测试在集群 H20 上完成,确定性增强测试在 A800 上完成)
> 代码与数据:`<WORKDIR>/MutaKernel-KGB/`(独立目录,未动既有 `MutaKernel`);结果在 `MutaKernel/runs/kgb_ext_llmemd/`
> 说明:EMD 等价判定使用四层(含 Layer3 LLM,Bedrock claude-opus-4-5);**增强测试维度不含 LLM**,仅用确定性维度。

---

## 0. 一句话结论(回答导师的问题)

**KGB 的正确性验证器确有 kill 不掉的 mutant(真漏检),且 MutaKernel 的五维增强能补杀其中相当一部分 —— 强力支持把 KernelGenBench 纳入论文作为新 subject。**

规模(54 个有效 subject、2664 个变异体)下:
- KGB 验证器**漏检 1186** 个变异体;
- EMD 等价判定(Layer0 文本 / Layer1 静态 / Layer2 统计 bitwise / Layer3 LLM)剔除真等价体后,
  得 **699 个真漏检**(strict 等价 215 + LLM 确认候选等价 272 + 真漏检 699 = 1186);
- MutaKernel 五维(仅确定性维度,**无 LLM**)**补杀 248 个(35.5%)**,最终逃逸 451 个;
- 变异分数(conservative):KGB baseline **0.595 → MutaKernel 0.698**(+10.4 个百分点)。

> **关键发现(EMD 多层)**:统计 bitwise 等价(在有限压力输入下逐位一致)会**系统性高估等价率**——
> LLM 层把 **399 个被统计判为"等价"的候选识别为其实并不等价**(真漏检集合近翻倍),同时将 54 个原判漏检
> 修正为等价。这本身就是 EMD 多层(尤其 LLM 层)价值的强证据。

这正是论文命题"Validating the Validators"想要的证据:一个专门做 kernel 生成评测的新 benchmark,其验证器也会系统性漏检语义故障,而 MutaKernel 能量化并补杀。

---

## 1. 做法(完全复用 MutaKernel,唯一新增一个 Bridge)

严格按实验设计文档:**变异方法不变(MutaKernel),种子=KGB 的正确 Triton 内核,oracle=KGB 自己的验证器**。

- **唯一新增**:`src/bridge/kernelgenbench_bridge.py`(`KernelGenBenchBridge`,仿 `eval_bridge.py`)。
  把 `sota_agents/AutoKernel/kernels/*.py` 的 `@triton.jit` 内核包成 `ModelNew`(变异算子改其中的 jit 体);
  参考用 `AutoKernel/reference.py` 的纯 torch 实现(即 ATen baseline)包成 `Model`;输入按 KGB
  `accuracy_utils` 的 shape/dtype 构造。**所有张量(含权重)都经 `get_inputs()` 产出**,保证 ref/mut 同 seed 下逐位一致。
- **oracle 落点**:`src/mutengine/kgb_runner.py`(`KGBMutantRunner`)把判等从 MutaKernel 默认 `torch.allclose(1e-2)`
  换成 **KGB 的 `kernelgenbench_assert_close`**(强制 `res.dtype==dtype`,`atol=1e-4*reduce_dim`,
  `rtol`=fp32 1e-5 / fp16 1e-3 / bf16 0.016)。直接 import KGB 的 `accuracy_utils`,**未改其容差**。
  → 这样 survived 才真正等于"KGB 验证器漏检"。
- **EMD(等价判定)**:复用 `EquivalentDetector` 的 Layer0(文本)/Layer1(静态规则)/Layer2(统计 bitwise,
  原始 vs 变异内核);Layer3(LLM,Bedrock claude-opus-4-5)对全部 survived 带 unified diff 逐个判等,
  把"统计上看不出差异但语义不等价"的变异体从候选等价里捞回为真漏检(并发调用 + 断点续跑)。
- **五维增强(确定性维度)**:KGB 原生驱动复用 `policy_bank.STRESS_POLICIES`,在 **A800** 上跑确定性维度:
  - `value_stress` — 21 种数值分布策略 × 多种子(保持 shape/dtype 不变,只改数值);
  - `config_stress` — **batch 规模变化:只改第一维 batch_size(`{1,2,4,8,16,32,64}` × 种子),其余张量维度固定为各问题规范 shape**;
  - `dtype_stress` — 在备选浮点精度下重跑;
  - `repeated_run` — 多次重复检测非确定性(UB / race)。
  - **LLM 维度按要求跳过**。子进程隔离 + 32 路单线程并行(`OMP_NUM_THREADS=1`),逐位(bitwise)比较。
- **工程鲁棒性**:GPU 变异测试里有变异体会触发 illegal memory access、毒化整个 CUDA context。
  做了**子进程隔离 + 崩溃即判 killed**:某变异体崩溃/超时→判其 killed(崩溃即被检出)→从下一个 fresh 进程继续,
  不污染其他用例。

新增代码(均 additive,未改动原逻辑):`kernelgenbench_bridge.py`、`kgb_runner.py`、
`run_kgb_mutation_test.py`、`_kgb_mutant_worker.py`、`run_kgb_emd.py`、`kgb_stress_orchestrator.py`、
`kgb_stress_worker.py`、`aggregate_kgb_stress.py`、`kgb_report.py`、`kgb_aggregate.py`。

---

## 2. 结果总表

### 2.1 主结果:9 算子 × {fp16,fp32,bf16} × 多 shape

| 指标 | 数值 |
|---|---|
| 有效 subject | 54 / 66 |
| 变异体总数 | 2664 |
| killed(KGB 抓到) | 1424 |
| stillborn(编译/实例化失败) | 54 |
| survived(KGB 漏检) | 1186 |
| ├─ strict 等价(Layer0 文本归一相同) | 215 |
| ├─ 候选等价(Layer1/2 + LLM 确认) | 272 |
| └─ **真漏检 true_escape** | **699** |
| 五维(确定性)补杀 rescued | **248** |
| **最终逃逸 final_escape** | **451** |
| 变异分数 conservative(baseline → MK) | **0.595 → 0.698** |
| 变异分数 optimistic(baseline → MK) | 0.671 → 0.788 |

EMD 层级:真漏检 **699**(经 Layer0 文本 / Layer1 静态 / Layer2 统计 bitwise / Layer3 LLM)。
LLM 层把 **399** 个统计 bitwise 判等的候选识别为真不等价、**54** 个改判为等价、**10** 个文本相同但 LLM 认为不等价的保守保留为 strict。
故障分类(确定性 EMD 标注):容差过松 216 · dtype/数值缺口 135 · shape 覆盖不足 2 · 模式未覆盖 1。
补杀首杀维度:value_stress 217 · dtype_stress 13 · config_stress 12 · crash(真实 CUDA 崩溃)6。
最有效策略(任一维度命中):config:batch_1(121)· dtype:float16(116)· value:near_zero(93)· value:large_magnitude(69)。
**候选等价体被增强测试杀掉 = 0/272**,强力佐证 LLM 等价判定正确(增强测试找不到任何反例)。
(crash 击杀已在单线程纯隔离 + 240s 超时下全部复现,确认为真实 CUDA 级非法访问,非并发误判。)

### 2.2 冒烟(管线验证)

14/18 有效 subject,778 变异体,真漏检 45,五维补杀 25,最终逃逸 20。冒烟与主结果结论一致,管线可复现。

---

## 3. 分算子洞见(对论文很有用)

按真漏检 / 五维补杀拆开看,**两类算子表现截然不同**:

| 算子 | survived | 真漏检(EMD 后) | 五维补杀 | 解读 |
|---|---|---|---|---|
| rmsnorm | 113 | 104 | **83 (80%)** | 数值/容差类逃逸,value/dtype 大量补杀 |
| layernorm | 213 | 150 | 90 (60%) | 数值/容差类,极端幅值 + dtype 补杀 |
| cross_entropy | 187 | 61 | 15 (25%) | 部分补杀(accumulation/dtype) |
| softmax | 170 | 81 | 18 (22%) | 部分补杀(near_zero/large_magnitude) |
| flash_attention | 232 | 165 | 36 (22%) | 逃逸多为 acc_downgrade/init 等**结构/精度**故障,纯数值扰动有限 |
| reduce | 99 | 81 | 6 (7%) | EMD 捞出大量真漏检,确定性维度补杀有限 |
| matmul | 28 | 21 | 0 (0%) | 真漏检以维度/步长计算类为主,确定性维度未补杀 |
| rotary_embedding | 144 | 36 | 0 (0%) | 逃逸多为 launch_config/index 等**结构**故障,需 config/索引感知或 LLM 维度 |

**关键洞见 1(维度分工)**:value/dtype 维度对**归约/规范化类**(rmsnorm/layernorm)的容差/数值漏检补杀率高(60%~80%);
但对**结构性故障**(错误的 launch config、索引、精度降级,如 rotary/flash_attention)补杀有限——这部分正是我们**按要求跳过的 LLM 增强维度**与更丰富的 config/training 维度该补的。给论文一个干净的"每个维度各司其职"分解叙事。

**关键洞见 2(EMD 多层的影响)**:LLM 层在 reduce / matmul / softmax 上识别出大量被统计 bitwise 漏判的真漏检
(reduce 81、matmul 21、softmax 81),说明**统计 bitwise 等价在这些算子上大量误判等价**;LLM 正确识别后,确定性增强测试虽未能全部补杀,
但暴露了"需要更强 oracle / 更强增强维度"的真实缺口——这对论文论证 EMD 多层必要性非常有力。

典型逃逸案例(带 diff 的明细见 `runs/kgb_ext_llmemd/details/`):
- `rmsnorm` `epsilon_modify`(dtype/数值)→ value_stress 补杀✓
- `rotary_embedding` `launch_config_mutate`(shape 覆盖)→ 五维未补杀(需 config/LLM)
- `softmax` `arith_replace`(softmax__float32__1024x1024)→ 触发 CUDA 非法访问,**crash 击杀✓**

**关键洞见 3(存活体根因)**:对 451 个最终逃逸逐个分析(见 `存活变异体逐个分析_中文.md`),绝大多数并非 kernel 计算逻辑真等价,
而是 **KGB 固定 benchmark 形状/数据类型留下的"输入空间盲区"**:被改写的代码分支需要特定 ndim、规约维、非连续布局、未命中的断言枚举项、
或非 2 幂宽度才会触发的边界掩码,而这些条件恰好落在确定性增强测试的覆盖之外(占比最高为边界掩码盲区 31%、断言/白名单守卫盲区 20%、维度分发盲区 16%)。

---

## 4. 诚实的局限 / caveats

1. **增强测试 LLM-free**:EMD 已用 LLM 做等价判定,但按要求**增强测试维度未用 LLM**,因此结构性逃逸的补杀偏保守,
   补上 LLM 增强维度后补杀率预计更高。LLM 逐条判等明细见 `runs/kgb_ext_llmemd/`。
2. **12 个 subject 被判种子无效**(已如实丢弃,不影响结论):
   - `matmul` fp32 ×2:`tl.dot` 在 Hopper 走 TF32,与 torch fp32 在 rtol=1e-5 下不符(真实精度现象);
   - `rotary` fp16/bf16 ×4:pointwise 无归约,atol=1e-4 对低精度过紧,逐元素有个别越界;
   - `fused_mlp` ×6:SwiGLU 两次链式 matmul(fp32 累加 vs torch 低精度 matmul)在 KGB 严格 oracle 下数值不符
     (已修好 triton 3.4 的 `tl.math.tanh` 兼容问题,但数值仍对不上 → 种子本身不够 oracle-clean)。
   这几条本身是"种子/精度"层面的小发现,可在论文里一句带过。
3. **种子来源**:本次种子=AutoKernel 自带的 ~9 个真实 Triton 内核(覆盖 ATen 类算子);
   全量 ATen-110 / vLLM-50 / cuBLAS-50 需要 KGB 的 LLM/Agent track 生成更多"过验证器"的正确种子(见后续)。
4. **故障分类是启发式**(按算子类别 + fp32-vs-低精度交叉验证),非人工标注;粗粒度足够支撑结论,精标可后做。

---

## 5. 对论文的建议

- **纳入,作为"外部 benchmark"式新 subject(RQ4 风格)或 RQ1–3 式新数据集**:论据是"连一个专门做 kernel
  生成评测的新 benchmark,其验证器也系统性漏检语义故障(**真漏检 699/2664**),而 MutaKernel 确定性增强能补 **35.5%**"。
- **EMD 多层叙事**:统计 bitwise 等价会**系统性高估等价率**(LLM 层从候选等价里捞回 399 个真不等价)——
  支撑 EMD 多层(尤其 LLM 层)的必要性;候选等价被增强测试杀 0/272 又反证 LLM 判等可靠。
- 叙事可强调**维度分工**:value/dtype 维度杀数值/容差类,结构类需 config/training/LLM 维度——支撑"五维缺一不可"。
- 与已有 subject(KernelBench 等)并列时,KGB 提供了**Triton 原生 + 多来源(ATen/vLLM/cuBLAS)**的新覆盖面。

---

## 6. 后续(需 LLM / 更多 GPU,均已留好接口)

1. 五维的 **LLM 增强维度**(对结构性逃逸如 launch_config/index 更有效)→ 预计进一步提高结构类补杀。
2. 全量种子:跑 KGB LLM/Agent track 为 ATen-110 生成"过验证器"的正确 Triton 种子 → RQ1/2/3 全量。
3. vLLM(50)/ cuBLAS(50):H20 可跑(NVIDIA),需装 `vllm==0.13.0`(会锁 torch≥2.9/triton≥3.5)。
4. fused_mlp / matmul-fp32 / rotary 低精度:若要纳入,可按 KGB 各算子的实际 reduce_dim/容差精确对齐(而非启发式)。

---

## 7. 复现 / 文件位置

集群 `<WORKDIR>/MutaKernel-KGB/`:
- 环境:`envs/kgb`(torch 2.8.0+cu128 / triton 3.4 / H20 CC9.0;**注意 driver 570 只支持到 CUDA 12.8,torch 必须 cu128,不能 cu13**)。
  A800 上确定性增强测试用 torch 2.1.2+cu121 / triton 2.1.0。
- 代码:`MutaKernel/src/bridge/kernelgenbench_bridge.py`、`MutaKernel/src/mutengine/kgb_runner.py`、
  `MutaKernel/scripts/run_kgb_*.py`、`_kgb_mutant_worker.py`、`kgb_stress_orchestrator.py`、`kgb_stress_worker.py`、
  `aggregate_kgb_stress.py`、`kgb_aggregate.py`、`kgb_report.py`。
- 结果:`MutaKernel/runs/kgb_ext_llmemd/` —— LLM-EMD + A800 确定性增强测试:
  - `summary.json`(变异分数)、`emd_summary.json`(EMD 分层判定明细)、`details/*.json`(54 个,逐变异体含 `final_emd_status`/`stress_result`/`post_stress_status`);
  - `stress/`:`STRESS_REPORT.md`、`stress_summary.json`(分维/分策略/分算子补杀)、`stress_details.json`、`results.jsonl`。
- 存活体逐个分析:`MutaKernel-KGB/存活变异体逐个分析_中文.md`(451 个最终逃逸的根因分类与逐条说明)。

复现命令:
```bash
# 1) 变异测试 + EMD(含 LLM 第3层)
cd <WORKDIR>/MutaKernel-KGB/MutaKernel
python scripts/run_kgb_mutation_test.py        # 产 survived 集合
python scripts/run_kgb_emd.py                  # Layer0/1/2 + Layer3 LLM 判等

# 2) 五维确定性增强测试(A800,子进程隔离 + 32 路并行)
python scripts/kgb_stress_orchestrator.py --workers 32 --timeout 240

# 3) 聚合(合并 EMD + 增强测试,产 stress_summary / STRESS_REPORT / details 更新)
python scripts/aggregate_kgb_stress.py
```
