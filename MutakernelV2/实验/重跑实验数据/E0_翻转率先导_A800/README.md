# E0 翻转率先导实验（Table 3）— A800 运行记录

> 启动日期: 2026-07-21
> 远程环境: AutoDL 容器 `autodl-container-8da74ab132`，NVIDIA A800 80GB PCIe，
> driver 590.48.01，CUDA 12.1（nvcc 12.1.r12.1），PyTorch 2.1.2+cu121，
> Python 3.10.8（/root/miniconda3），112 vCPU。
> 驱动脚本: `scripts/run_e0_flip_rerun.py`（配对双臂设计）
> 分析脚本: `scripts/analyze_e0.py`

## 1. 实验设计（与蓝图 Table 3 的对应）

同一探针在**同一 GPU 上背靠背跑两臂**：

- **legacy 臂**: 冻结的 pre-fse-rework runner（`src/mutengine/mutant_runner_legacy.py`，
  逐字节取自 tag `pre-fse-rework-20260719`）——无 RNG 回放、无状态同步、共享输入；
- **corrected 臂**: 修正基座 runner（RNG 回放 + 严格按名状态同步 + 输入隔离 +
  严格 oracle + 三值结果）。

两臂协议完全相同（5 组随机输入、atol=rtol=1e-2、eval、seed 42，与 V1 一致）。
**主指标 = legacy vs corrected 的配对翻转率**（隔离基座效应，消除硬件/环境漂移
混杂）；corrected vs historical 为次要指标（含环境漂移）。

**original-kernel 控制门控**：每个 kernel 先跑"未变异原始内核"双臂控制；任一臂
非 survived → 该 kernel 全部探针标 `original_control_ok=false`，不计入可归因
翻转统计（保留在日志中）。该门控在 A800 上立刻发挥作用——例如 L1_P100 的
原始内核在 A800 上本身就 illegal memory access（V1 跑在 3090/H20 上），
其探针的任何判定都不可归因于变异。

**抽样**: 分层（level × reference 是否含状态参数 × 历史 killed/not-killed），
每层 30，seed 20260721，共 180 探针；驱动按 probe_id hash 分片、断点续跑。

## 2. 执行过程记录（含三个被冒烟发现并修复的问题）

| 时间(UTC) | 事件 |
|-----------|------|
| 12:08 | smoke(2 探针)首跑：发现 **修复#1** —— candidate 的 illegal memory access 毒化 CUDA 上下文后，`caller_rng.restore()` 在 finally 中再抛异常，把已判定的 FAIL(killed) 覆盖成 WorkerCrash(stillborn)。修复：`mutant_runner.py` 的 finally 改为记录而不传播（判定已定、进程即退）。 |
| 12:17 | smoke 复跑通过：corrected 臂正确输出 killed（"candidate execution failed while the reference succeeded"） |
| 12:23 | 首次先导（单进程）启动；后改 4 分片并行 |
| 13:13–13:19 | 4 分片下大量控制 unknown —— 诊断（`_diag_e0` 逐 trial）确认 **修复#2**：并发 worker 无显存配额互相挤爆（单 worker 吃 56GB，其余 OOM→INCONCLUSIVE）。修复：worker 加 `MK_GPU_MEMORY_FRACTION` 每进程显存上限；最终决策改回**串行单分片**（run4） |
| 13:20 | **修复#3**：`pkill -f run_e0_flip_rerun.py` 自匹配杀掉自己的 SSH 会话 + nohup 未断开 stdin 导致通道挂起——改用不自匹配模式 + `</dev/null` 子 shell 包裹 |
| 14:12 起 | **run4（串行，权威运行）** 稳定推进：控制在串行下恢复 survived（如 L1_P96 从并发下的 unknown → 串行 survived），配对观测开始积累 |

三个修复本身就是 E0 的价值证明：进程隔离 + 三值结果 + 控制门控让每一类
基础设施故障都被显式分类，而不是污染判定。

## 3. 数据落盘位置

- `interim_20260721/`：截至 21 日晚的全部中间产物（run2/run3/run4 各分片的
  `run_manifest.json`（含环境指纹、协议、抽样计划、worker sha256）、
  `observations.jsonl`（逐探针双臂记录：时间戳、wall_ms、kill_seed、错误摘要）、
  `original_controls.json`、`worker_logs/`（非正常轮的完整 stdout/stderr）、
  驱动日志 `e0_*_s*.log`）；
- 远程权威运行目录: `/root/mk_v2_runs/e0_run4_s0/`（断点续跑，checkpoint =
  `completed.json`）；
- **重要甄别口径**: run2/run3/pilot 的观测大多 `original_control_ok=false`
  （并发 OOM 污染控制），只作过程留档；**Table 3 只从 run4（串行）完成后的
  数据计算**。

## 4. 完成后的收集与分析命令

```powershell
# 本地（Windows，需 $env:REMOTE_PW）
python _remote_e0.py status run4     # 查看进度（180 探针）
python _remote_e0.py collect         # 打包下载全部结果
python scripts/analyze_e0.py "MutakernelV2/实验/重跑实验数据/E0_翻转率先导_A800/<新目录>/e0_run4_s0"
# 输出分层翻转表（Wilson 95% CI）+ table3_summary.json → 填入蓝图 Table 3
```

## 5. 诚实性备注

- Table 3 在 run4 完成并经 `analyze_e0.py` 聚合前保持 "?"（拒绝用不完整数据预填）；
- A800 与 V1 的 3090/H20 架构不同，corrected-vs-historical 翻转含架构漂移，
  论文只把 **paired（legacy vs corrected，同机同时）** 作为基座效应的主证据；
- 控制失败率本身是有价值的副产品数据（原始内核的架构可移植性问题）。
