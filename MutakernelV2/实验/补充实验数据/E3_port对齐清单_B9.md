# B9 Seeded differential fuzzing（gpuemu, Sarkar 2026）协议移植对齐清单

- 上游工件：Skelf-Research/gpuemu，commit `696b510bc21f8036bcee395749b7fa2c2b4baf2a`；语料快照 gpuemu-corpus `2f153103`（本地 `external/C6_gpuemu/`）
- 协议参照实现：`gpuemu-corpus/drivers/_p1lib.py`（daemon/local 双 oracle）、`gpuemu_corpus/data/_refkit.py`（fp64 参考协议）、`crates/gpuemu-daemon/src/validator.rs`（比较语义）
- 移植实现：`src/experiments/external_ports/b9_seeded_fuzzing.py`；调度：`scripts/run_e3_external.py plan/run --baseline b9`
- 登记：`configs/external_baselines.json` → `seeded-fuzzing-style-port-v1`（本次新增）

## 逐条款对齐

| # | 上游协议条款（出处） | 我们的实现 | 状态 |
|---|---------------------|-----------|------|
| 1 | op-schema-aware 采样：形状从 op_schema 各维 candidates 列表抽取（`_p1lib.sample_case`） | schema 源改为科目契约：固定形状按契约冻结值（等价于单 candidate 维度）；契约 batch adapter 存在时其 `allowed_values` 即维度 candidates（`plan_b9_port(batch_values=...)`），逐案例轮转 | 已对齐（schema 来源不同，语义一致） |
| 2 | dtype 从 op 声明列表抽取（meta.json `dtypes`） | 契约浮点 dtypes ∩ 已校准集合 {float32, float16}，逐案例轮转；不在校准集内的 dtype 显式拒绝（不静默丢弃） | 已对齐 |
| 3 | 取值分布：uniform [-10, 10]（`_p1lib.make_inputs`，"mirror the Rust fuzzer's value range"） | `b9_uniform10`：同分布，浮点张量逐种子重采，shape/dtype 保持 | 已对齐 |
| 4 | fp64 CPU 参考：参考实现以 float64 计算后**舍回 kernel 输出 dtype**（`_refkit.py` "correctly-rounded ideal"） | `run_fp64_cpu_reference`：参考模块 `.cpu().double()`，浮点输入升 float64，输出 `.to(candidate_output.dtype)` | 已对齐 |
| 5 | 逐 (op,dtype) 校准绝对容差：p95-of-controls × 1.5（外部资源获取清单登记的校准协议；语料 meta.json 容差即其产物，如 softmax_triton fp32 1e-5 / fp16 2e-2） | `calibrate_tolerance(errors, quantile=0.95, factor=1.5)` 原样实现；控制内核校准跑完成前使用 `B9_DEFAULT_TOLERANCES`（fp32 1e-5 / fp16 2e-2，语料 meta 惯例值）并在观测记录中写明所用容差 | 已对齐（校准执行阻塞于 GPU/控制运行） |
| 6 | 比较语义（validator.rs / `_p1lib.compare`）：先 shape，再 NaN、Inf，再 \|err\|>tol 计数；输出 error stats（max/mean abs、rel、分位数） | `compare_to_fp64` + `error_stats` 逐语义移植（failure_kind ∈ {shape, nan, inf, tolerance}；p50/p90/p99 分位数） | 已对齐（ULP 距离见"不支持"） |
| 7 | 预算：上游按 `--iters` 每 (strategy, kernel) 固定迭代数 | port 行恰好 32 次候选调用（fp64 参考执行为参考侧，不计候选预算——与所有其他行同口径） | 已对齐（预算匹配是蓝图要求） |
| 8 | native 语料复现（reproduction delta B9）：在 C6 26-op 语料上复现论文 10/10 caught、16/16 clean | 未执行——`run_e3_external.py c6` 装载器与任务表就绪，执行阻塞于 GPU（Triton kernel 需要设备） | 待办（阻塞于 GPU） |

## 明确不支持条款

1. **Rust 守护进程的逐位一致用例流**：我们镜像文档化的分布（uniform[-10,10] × schema 抽样），不复刻其字节级 RNG 流；上游自己的 local oracle 也如此声明（"not bit-identical to the Rust fuzzer"）。
2. **daemon IPC 协议**（gpuemu.sock、get_test_batch/submit_output）：port 在进程内直接完成采样与判定，无守护进程。
3. **layout/strides 元数据 fuzzing**（上游 case 的 `layout` 字段）：超出契约授权 layout 的布局变异不执行；契约内 noncontiguous 已由 M-full 维度覆盖（不同表行）。
4. **ULP 距离统计**（validator.rs ErrorStats 的 max/mean ULP）：判定不依赖 ULP（判定只用绝对容差 + NaN/Inf），port 的 error stats 省略 ULP 两列；如复现需要可后补。
5. **nan_injected / adversarial 值分布策略**（p3_strategies.py 的策略消融）：属上游论文的策略消融实验，不属 B9 基线协议本体。
