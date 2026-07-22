# E1 探针研究（RQ4）执行手册 —— run5 结束后待执行清单

状态（2026-07-22）：CPU 阶段已在远程完成并验证；GPU 阶段按下列顺序执行。
远程环境：`/root/mk_v2`，python=`/root/miniconda3/bin/python`，
`PATH=/usr/local/cuda/bin:$PATH`，`TORCH_EXTENSIONS_DIR=/root/mk_v2_runs/torch_ext`。

## 已完成（CPU，run5 期间）

- 探针重新生成：`/root/mk_v2_runs/e1/probes/` + `probe_manifest.json`
  （1,646 = A757 + B702 + C178 + D9；90/90 内核与 V1 历史探针 ID 逐一吻合；
  9 个 machine-proven，全部 `dead_host_constant` v1.0；
  静态规则内容版本与探针 ID digest 已入 manifest）。
- E2 契约提取（KernelBench L1+L2）与 C2–C5 stable-ID 收集框（见
  `补充实验数据/collection_frames/` 与远程 `/root/mk_v2_runs/e2/`）。
- C6 gpuemu 语料 dry-run：26 op = 16 对照 + 10 seeded bug，文件齐全。
- 远程 CPU 测试套件 131 passed。

## run5 结束后按序执行（GPU）

确认 run5 已退出（`ps aux | grep run_e0_flip_[r]erun` 为空）后：

1. **GPU 冒烟**（约 10 分钟，验证 worker 在当前驱动可编译执行）
   ```bash
   cd /root/mk_v2 && PATH=/usr/local/cuda/bin:$PATH \
     TORCH_EXTENSIONS_DIR=/root/mk_v2_runs/torch_ext \
     /root/miniconda3/bin/python -u scripts/run_e0_flip_rerun.py \
     --details-dir phase1_details --kernelbench-root KernelBench \
     --out-dir /root/mk_v2_runs/e1_gpu_smoke --mode smoke --limit 2 --timeout 300
   ```
2. **E1 baseline（B1 协议在 corrected 基座重放；含逐内核对照门控）**
   ```bash
   cd /root/mk_v2 && ( nohup env PATH=/usr/local/cuda/bin:$PATH \
     TORCH_EXTENSIONS_DIR=/root/mk_v2_runs/torch_ext MK_GPU_MEMORY_FRACTION=0.9 \
     /root/miniconda3/bin/python -u scripts/run_e1_probe_study.py \
     --phase baseline --kernelbench-root KernelBench \
     --out-dir /root/mk_v2_runs/e1 --timeout 420 \
     > /root/mk_v2_runs/e1_baseline.log 2>&1 < /dev/null & )
   ```
   断点续跑：重复同一命令即可（baseline_completed.json）。
3. **E1 equiv（幸存者等价证据）**
   ```bash
   ... scripts/run_e1_probe_study.py --phase equiv \
     --kernelbench-root KernelBench --out-dir /root/mk_v2_runs/e1 --timeout 600
   ```
4. **E1 map（CPU，可立即跟跑）**：fault-to-stress map + 任务级 k 折 crossfit +
   逃逸机制分类
   ```bash
   CUDA_VISIBLE_DEVICES= ... scripts/run_e1_probe_study.py --phase map \
     --out-dir /root/mk_v2_runs/e1
   ```
5. **CSE 证伪 LIKELY_EQUIVALENT**
   ```bash
   ... scripts/run_e1_cse_falsify.py --out-dir /root/mk_v2_runs/e1 \
     --kernelbench-root KernelBench --timeout 900
   ```
6. **盲审等价队列导出（CPU）**
   ```bash
   CUDA_VISIBLE_DEVICES= ... scripts/export_e1_blind_equiv_queue.py \
     --e1-dir /root/mk_v2_runs/e1 --output-dir /root/mk_v2_runs/e1_blind \
     --salt <保密盐值>
   ```
7. **E3-C6 / B11**（在 E1 之后或穿插）：
   - C6 GPU runner 尚需实现（loader 已就绪，TODO 见 `scripts/run_e3_external.py`）。
   - B11 冒烟：`scripts/b11_compute_sanitizer.py --subject-id <id> --out-dir ... -- <replay 命令>`。

## 注意事项（E0 教训固化）

- 对照门控失败的内核：其全部探针记 `excluded_control_failed` 并带分类原因，
  不进 Table 10 分母，但保留在观测记录中可审计。
- 两类正当 INCONCLUSIVE（`state_sync_nonbijective` / `cuda_invalid_configuration`）
  在 `baseline_summary.json` 单列分层；若 stateful 内核对照大面积
  state-sync 失败，说明 E0 已知缺口未修复，先修基座再重跑。
- 一切串行；不并发多 shard，除非确认显存预算（MK_GPU_MEMORY_FRACTION）。
