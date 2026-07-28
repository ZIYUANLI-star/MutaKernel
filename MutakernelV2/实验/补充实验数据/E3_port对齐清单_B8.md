# B8 KernelBenchX 协议移植对齐清单

- 上游工件：BonnieW05/KernelBenchX，commit `fd4192293bf9a8c645327a9d46aa1e807f1f9cf2`（本地 `external/B8_kernelbenchx/`）
- 移植实现：`src/experiments/external_ports/b8_kernelbenchx.py`；调度：`scripts/run_e3_external.py plan/run --baseline b8`
- 登记：`configs/external_baselines.json` → `kernelbenchx-style-port-v1`

## 逐条款对齐

| # | 上游协议条款（出处） | 我们的实现 | 状态 |
|---|---------------------|-----------|------|
| 1 | standard 输入族：`rand_tensor(mode="standard")` = N(0,1)（`utils/data_utils.py`） | `b8_standard`：对科目模板输入的浮点张量按种子重采 N(0,1)，shape/dtype/layout 保持（与 M-full 策略库同一包装器，杜绝静默 no-op） | 已对齐 |
| 2 | outlier 输入族：N(0,1)，以 `outlier_prob=0.001` 概率将元素重采为 N(0,1)×`outlier_scale=50`（冻结参数） | `b8_outlier`：同分布、同默认参数（0.001 / 50），逐种子确定性 | 已对齐 |
| 3 | uniform 输入族：U(low, high)，默认 (-1,1)（少数 golden 测试使用） | `b8_uniform` 已实现并注册；32 次预算布局默认不占位（上游主打 standard/outlier），需要时可换布局 | 已对齐（实现可用，默认不入预算） |
| 4 | boundary 输入：每任务 golden 文件手写的 `test_case_*` 边界用例（`boundary_inputs_where_defined`） | **不支持**（见下）——手写用例只为上游自己的 176 任务定义，不可泛化移植到 C2–C5 科目；native 模式保留 | 显式不支持 |
| 5 | dtype-aware oracle：fp16/bf16 rtol=atol=5e-3、fp32/fp64 1e-5（`EVAL/1_exe_acc.py::_default_tol`），`torch.testing.assert_close(equal_nan=True)` | port 行用统一判定管线；上游默认容差以 `B8_NATIVE_DTYPE_TOLERANCES` 常量原样留档，供 native 锚点行与复现对差使用 | 已对齐（oracle 统一是蓝图要求） |
| 6 | 单种子整文件重放（`KERNELBENCHX_SEED` + `_seed_all` 双侧同种子） | 逐案例种子（case seed）双侧同 RNG 状态（`validate_pair` 的 RNG 回放），语义等同上游"两侧同种子执行再比较" | 已对齐 |
| 7 | 任务框：`data/kernelbenchx_v1.json` 176 任务清单，不 glob 目录 | native 模式将以该清单为冻结任务框（配置已登记）；port 模式作用于我方冻结科目，任务框不适用 | 已对齐 |
| 8 | 预算：上游整文件跑一遍（无预算概念） | port 行恰好 32 次候选调用：16 standard + 16 outlier | 已对齐（预算匹配是蓝图要求） |
| 9 | native 数据集复现：对 `metrics/`、examples 榜单数据对差（reproduction delta B8） | 未执行——GPU 工作，待 A800 释放 | 待办（阻塞于 GPU） |

## 明确不支持条款

1. **手写 per-task boundary 用例**（golden 文件 `test_case_1..N`）：仅对上游 176 任务存在，无法在不重新人工编写的情况下移植到 C2–C5 科目；port 的"边界"维度由我方 M-full 边界策略族独立承担（不同表行）。
2. **cosine/L1/RMSE 自定义精度阈值**（`precision_thresholds` per-task 覆盖 + `precision_metric`）：port 行由统一 oracle 取代；native 行保留上游原判。
3. **kernel 导出与 AST 卫生门**（`impl_must_export_kernel`、`check_triton_validity`）：绑定上游 Triton 文件布局，与我方科目接口无对应物。
