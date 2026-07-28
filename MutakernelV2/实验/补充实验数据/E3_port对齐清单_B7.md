# B7 robust-kbench 协议移植对齐清单

- 上游工件：SakanaAI/robust-kbench，commit `078f5bab29934a822268d59a4e707d449abf9b4e`（本地 `external/B7_robust_kbench/`）
- 移植实现：`src/experiments/external_ports/b7_robust_kbench.py`；调度：`scripts/run_e3_external.py plan/run --baseline b7`
- 登记：`configs/external_baselines.json` → `robust-kbench-style-port-v1`
- 口径：port 行与 native 行永不同表行；port 走统一判定管线 + 统一 32 次候选调用预算

## 逐条款对齐

| # | 上游协议条款（出处） | 我们的实现 | 状态 |
|---|---------------------|-----------|------|
| 1 | multi-init：按 `config_forward.json` 的 `multi_init_configs` 多次重建模型（`ConfigTask.get_configs`、`kernel_task.py`） | 计划参数 `init_seed`∈{0,1}（backward 布局）或 {0..3}（forward-only 布局）；执行器在 `torch.manual_seed(init_seed)` 下对 reference 与 candidate 同种子重建实例 | 已对齐（init 配置字典 → init 种子的映射见"差异"§） |
| 2 | multi-input：每 init 下多组 `get_inputs(**input_config)` 新采样（`filter/forward.py`、`primitives/evaluate.py`） | policy `iid` + 独立 seed（用科目任务自己的 `get_inputs` 分布，与上游同源） | 已对齐 |
| 3 | 每个 forward 正确性 trial 调用候选 2 次（`external_baselines.json` 冻结参数 `candidate_calls_per_forward_trial: 2`） | mode `repeated`、`repeat_count=2`，`TestCaseSpec.candidate_run_cost=2`——两次调用都计入预算（port 计费规则原文） | 已对齐 |
| 4 | forward 与 backward 双向比较（`run_kernel.py --backward`、`func_backward.py` AutogradFunction） | 契约授权 backward 的科目：mode `train` 案例，比较输出 + 对全部浮点输入叶子的 VJP（`grad_outputs=ones`）；不授权则全预算给 forward | 已对齐（限契约授权范围） |
| 5 | 统计输出过滤器（`run_filter.py`）：output_range / output_std / output_axes / input_impact，阈值一律 0.01 | `filter_output_range/std/axes/input_impact` 纯函数逐条移植，同阈值 0.01，语义为任务协议健康检查（与上游一致，不改判候选） | 已对齐 |
| 6 | oracle：`torch.allclose(atol=rtol=1e-5)`（native 冻结参数） | port 行替换为统一判定管线（`src.validation.compare_outputs`，dtype-aware）；1e-5 仅保留在 native 锚点行 | 已对齐（蓝图 §5.3 oracle 统一要求） |
| 7 | 预算：上游无预算上限（5 trials × 每 trial 2 次 + eval 重复） | port 行恰好 32 次候选调用：backward 科目 2 init × 4 draw × 2 (fwd) + 2 init × 8 draw × 1 (bwd)；forward-only 4 init × 4 draw × 2 | 已对齐（预算匹配是蓝图要求，非上游行为） |
| 8 | native 数据集复现（§5.1.2 port fidelity）：在 `tasks/` 11 任务 + `highlighted/results.csv` 上重放并对差 | 未执行——GPU 工作，待 A800 释放后跑（reproduction delta B7） | 待办（阻塞于 GPU） |

## 明确不支持条款

1. **LLM sanity 过滤器**（`filter_llm_sanity`，claude-3-7-sonnet 判冗余/低效）：外部 LLM 依赖，port 不执行；`run_output_filters` 返回值中显式置 `None`。
2. **NCU / clang-tidy 剖析与 speedup 测量**（`prof_cuda_kernel`、`eval_cuda_kernel`）：性能测量在正确性 port 范围之外。
3. **上游 .cu 任务接口**（`forward.cu`/`backward.cu` 编译装载）：port 直接驱动科目自身的候选模块；.cu 接口仅在 native 模式使用。

## 与上游的其余差异（非条款级）

- 上游 multi_init/multi_input 配置是**任务相关的构造参数字典**；我们的冻结科目接口没有等价的参数空间，故映射为"init 种子 × 输入种子"网格——生成机制不同、协议意图（多初始化 × 多输入的稳健性检验）保持。
- 上游 warmup/timing 循环不移植（同条款 2 说明，正确性无关）。
