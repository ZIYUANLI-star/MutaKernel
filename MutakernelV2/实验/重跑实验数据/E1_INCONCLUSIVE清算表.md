# E1 全阶段 INCONCLUSIVE 清算表(第一轮,2026-07-27 14:44 北京时间)

## 终版归类规则(2026-07-27 15:10 增补,导师确认口径)

1. 补跑后得到有效判级的探针,其实验数据**归入对应判级类别**(LIKELY / WITNESSED / FALSIFIED 等),中间态 INCONCLUSIVE 记录仅归档留痕,不进入任何最终统计;
2. 只有**穷尽对症补跑配置后仍无法定级**的探针才计入终版 INCONCLUSIVE(蓝图 Table 10 的 INCONCLUSIVE 列语义 = 盲审后仍无法定级,与此一致);
3. 任何"归因为环境/资源/预算"的 INCONCLUSIVE 都不允许直接进入终版——必须至少经历一次对症配置的重试;只有归因为"结构性不可判"(如状态同步非双射)或"重试后确定性复现且严格 Oracle 仍无法健全定级"的才算穷尽。

数据源:远端 /root/mk_v2_runs/e1 全量扫描(活动观测 + 归档),只读盘点脚本 `_inc_inventory_remote.py` / `_inc_inventory2_remote.py`;重判试算用远端现行部署的预注册判级 `grade_cse_evidence`(40/40 随机 + >=42/63 定向 + 零 divergence;resource_degraded:>=77 有效轮 + >=30 随机过 + >=42 压力过,作废轮全为环境事件)。

## 0. 摘要

| 项 | 数 |
|---|---|
| 现存 INCONCLUSIVE 记录(含归档) | 45 条(CSE 归档 44,equiv 现存 1) |
| CSE 归档唯一探针 | 39(双档重叠 5 条) |
| 其中已闭环(重跑/重判) | 4 |
| 仍需重跑(已全部在 lanes 3/4/5 队列) | 35 |
| 本轮新增重判挽救 | 0(唯一达标者 L1_P20__relop_replace__3 前任已重判入账,本轮试算独立复核一致) |
| equiv 历史超时归档(timeout_v1/v2) | 121 条记录 / 120 唯一探针,全部已在合并文件闭环(106 LIKELY + 14 WITNESSED),非存量 |

归因分布(CSE 归档 44 条记录):

- 预算不足(1800s 截断):36 条
- 显存配额(0.45 轮内 OOM 作废):3 条
- 主机内存(cgroup SIGKILL 提前终止):2 条
- 争用作废(三道并发显存耗尽)+超预算:2 条
- 显存配额(5 轮 OOM 作废,余轮达标):1 条
- 合法拒判(equiv):1 条(保留 INCONCLUSIVE,不重跑)

## 1. equiv 阶段现存 INCONCLUSIVE(1 条)

| probe_id | inconclusive_class | 证据 | 归因 | 处置 |
|---|---|---|---|---|
| L1_P95__relop_replace__5 | other_unknown | worker 崩溃:CUDA illegal memory access(wall 61s,无可用轮级证据) | 待定:崩溃可能是变异体真实缺陷(越界访问),也可能是环境瞬态 | **按终版归类规则改为重试**:CSE 收官后独占窗口重跑一次;若确定性复现且参考端健康,按严格 Oracle 判 SPEC_VIOLATION 归入实锤非等价;若瞬态消失则正常判级;两次重试后仍无法定级才计终版 INCONCLUSIVE |

## 2. CSE 归档逐条清算(44 条记录,按归档文件)

dry 列 = 按现行预注册判级对归档轮级证据的离线重判试算(未写账)。

### 2.1 lane0 超时归档(36 条,1800s 预算)

| probe_id | class | wall | 轮级证据摘要 | 归因 | 重判试算 | 现状 |
|---|---|---|---|---|---|---|
| L1_P16__arith_replace__6 | timeout | 1800s | 60轮 随机过40/40 压力过20/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 已重跑 lane3→STILL_LIKELY_EQUIVALENT |
| L1_P19__arith_replace__3 | timeout | 1803s | 58轮 随机过40/40 压力过18/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane3 |
| L1_P19__relop_replace__1 | timeout | 1803s | 59轮 随机过40/40 压力过19/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane3 |
| L1_P20__arith_replace__18 | timeout | 1802s | 60轮 随机过40/40 压力过20/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 已重跑 lane4→STILL_LIKELY_EQUIVALENT |
| L1_P20__relop_replace__3 | timeout | 1802s | 59轮 随机过40/40 压力过19/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 已重跑 lane4→STILL_LIKELY_EQUIVALENT |
| L1_P20__relop_replace__7 | timeout | 1801s | 56轮 随机过40/40 压力过16/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane4 |
| L1_P20__relop_replace__4 | timeout | 1806s | 58轮 随机过40/40 压力过18/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane4 |
| L1_P20__const_perturb__11 | timeout | 1802s | 58轮 随机过40/40 压力过18/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane4 |
| L1_P20__mask_boundary__0 | timeout | 1802s | 59轮 随机过40/40 压力过19/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane4 |
| L1_P21__relop_replace__3 | timeout | 1803s | 59轮 随机过40/40 压力过19/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane3 |
| L1_P21__relop_replace__0 | timeout | 1803s | 60轮 随机过40/40 压力过20/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane3 |
| L1_P21__relop_replace__5 | timeout | 1803s | 60轮 随机过40/40 压力过20/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane3 |
| L1_P21__const_perturb__2 | timeout | 1802s | 59轮 随机过40/40 压力过19/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane3 |
| L1_P21__mask_boundary__0 | timeout | 1801s | 58轮 随机过40/40 压力过18/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane3 |
| L1_P22__arith_replace__20 | timeout | 1801s | 69轮 随机过40/40 压力过29/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 已重跑 lane5→STILL_LIKELY_EQUIVALENT |
| L1_P22__relop_replace__6 | timeout | 1802s | 72轮 随机过40/40 压力过32/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P22__relop_replace__8 | timeout | 1802s | 73轮 随机过40/40 压力过33/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P22__const_perturb__21 | timeout | 1801s | 73轮 随机过40/40 压力过33/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P22__const_perturb__0 | timeout | 1802s | 74轮 随机过40/40 压力过34/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P22__const_perturb__6 | timeout | 1801s | 74轮 随机过40/40 压力过34/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P22__index_replace__10 | timeout | 1802s | 72轮 随机过40/40 压力过32/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P22__mask_boundary__0 | timeout | 1801s | 72轮 随机过40/40 压力过32/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P22__mask_boundary__1 | timeout | 1801s | 73轮 随机过40/40 压力过33/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P23__const_perturb__3 | timeout | 1802s | 73轮 随机过40/40 压力过33/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P23__const_perturb__2 | timeout | 1802s | 74轮 随机过40/40 压力过34/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P23__const_perturb__1 | timeout | 1801s | 72轮 随机过40/40 压力过32/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P23__index_replace__6 | timeout | 1802s | 74轮 随机过40/40 压力过34/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P23__mask_boundary__0 | timeout | 1801s | 71轮 随机过40/40 压力过31/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P23__mask_boundary__1 | timeout | 1802s | 74轮 随机过40/40 压力过34/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P23__init_modify__0 | timeout | 1801s | 70轮 随机过40/40 压力过30/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P23__init_modify__1 | timeout | 1802s | 74轮 随机过40/40 压力过34/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane5 |
| L1_P24__relop_replace__7 | timeout | 1801s | 70轮 随机过40/40 压力过30/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane4 |
| L1_P24__const_perturb__9 | timeout | 1802s | 70轮 随机过40/40 压力过30/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane4 |
| L1_P24__const_perturb__4 | timeout | 1801s | 72轮 随机过40/40 压力过32/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane4 |
| L1_P24__const_perturb__13 | timeout | 1802s | 72轮 随机过40/40 压力过32/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane4 |
| L1_P24__index_replace__12 | timeout | 1802s | 72轮 随机过40/40 压力过32/63 整探针超时 | 预算不足(1800s 截断) | 不达标 | 待跑 lane4 |

注:L1_P20__relop_replace__3 在两档各有一条记录;其闭环不是重跑,而是对 envfix 档那次 103 轮记录的离线重判入账(见 2.2),lane4 观测中的记录与 envfix 归档 wall/证据逐项一致(wall 3594s、98 有效轮、随机 35、压力 63)。

### 2.2 envfix 归档(8 条,三道并发资源耗尽)

| probe_id | class | wall | 轮级证据摘要 | 归因 | 重判试算 | 现状 |
|---|---|---|---|---|---|---|
| L1_P96__arith_replace__12 | out_of_memory | 1876s | 49轮 随机过0/40 压力过0/63 作废49(环境) | 显存配额(0.45 轮内 OOM 作废) | 不达标 | 待跑 lane3 |
| L1_P96__arith_replace__14 | out_of_memory | 1879s | 51轮 随机过10/40 压力过11/63 作废30(环境) | 显存配额(0.45 轮内 OOM 作废) | 不达标 | 待跑 lane3 |
| L1_P96__relop_replace__8 | out_of_memory | 2518s | 60轮 随机过0/40 压力过0/63 作废60(环境) | 显存配额(0.45 轮内 OOM 作废) | 不达标 | 待跑 lane3 |
| L1_P20__relop_replace__3 | out_of_memory | 3594s | 103轮 随机过35/40 压力过63/63 作废5(环境) | 显存配额(5 轮 OOM 作废,余轮达标) | STILL(达标) | 重判入账(resource_degraded)→STILL_LIKELY_EQUIVALENT |
| L1_P20__relop_replace__7 | other_unknown | 1110s | 44轮 随机过40/40 压力过4/63 | 主机内存(cgroup SIGKILL 提前终止) | 不达标 | 待跑 lane4 |
| L1_P20__relop_replace__4 | other_unknown | 1357s | 49轮 随机过40/40 压力过9/63 | 主机内存(cgroup SIGKILL 提前终止) | 不达标 | 待跑 lane4 |
| L1_P22__relop_replace__6 | timeout | 3604s | 102轮 随机过1/40 压力过62/63 作废39(环境) 整探针超时 | 争用作废(三道并发显存耗尽)+超预算 | 不达标 | 待跑 lane5 |
| L1_P22__relop_replace__8 | out_of_memory | 2348s | 71轮 随机过0/40 压力过29/63 作废42(环境) | 争用作废(三道并发显存耗尽)+超预算 | 不达标 | 待跑 lane5 |

双档重叠探针(计 1 次):L1_P20__relop_replace__3, L1_P20__relop_replace__4, L1_P20__relop_replace__7, L1_P22__relop_replace__6, L1_P22__relop_replace__8

## 3. 重排队完整性对账(14:35 快照)

| 项 | 值 |
|---|---|
| lane0 拆分目标(全局) | 206 |
| 已完成(completed 并集∩目标) | 7 |
| 待跑 | 199(lane3=98, lane4=47, lane5=54) |
| 完成+待跑 | 206 = 206 ✓ |
| 待跑未被 lanes3/4/5 覆盖 | 0 ✓ |
| 活动观测跨道重复 probe_id | 0 ✓ |
| completed 文件 | lane0=3, lane1=123, lane2=124, lane3=1, lane4=2, lane5=1 |
| 归档探针在队列 | 36 条超时档:4 已闭环 + 32 在队;8 条 envfix:1 已闭环 + 7 在队;无遗漏,无需补队 |

lanes 1/2 已收官(obs 123+124,DONE 标记在),全局 CSE 目标 453 条 LIKELY_EQUIVALENT = lane0 拆分 206 + lanes1/2 247。

## 4. 对症配置矩阵(仍需重跑的 35 条唯一探针)

| 归因类 | 待跑数 | 所在 lane | 对症配置 | 覆盖判断 | 兜底预案 |
|---|---|---|---|---|---|
| 预算不足(1800s 截断) | 28(纯超时类) | 3/4/5 | 3600s 预算 + budget_exhausted 部分证据判级 | 已覆盖:历史 1800s 完成 56–74 轮,3600s 外推 112–148 轮 ≥ 103 计划轮;判 STILL 只需 40+42 过。先例:L1_P16/P20 重跑 2834/3513s 全轮完成 | 若个别再超时,由部分证据判级兜底 |
| 显存配额(P96 族) | 3 | lane3 队首(串行 0.92≈75.3GB) | 串行独占显存 + 3600s | 基本覆盖但余量最小:P96 equiv 中位 953s×3.22≈3069s,且历史 0.45 下全轮 OOM(6GiB 分配失败),0.92 下应可分配 | **专门补跑安排**:若 lane3 跑 P96 再 INC(超时/OOM),CSE 收官后独占窗口重跑:无并发、0.92、`--timeout 5400`、`--lane 3 --lane-tag p96_5400`(需先从 lane3 obs 归档该 INC 记录并从 cse_completed_lane3.json 摘除 id,lane3 停机后操作) |
| 主机内存(cgroup SIGKILL) | 2(P20_relop_4/7) | lane4 | 两道并发上限(host RAM 83/120GB)+ 守护 v10.1 串接 lane5 | 已覆盖:SIGKILL 源于三道并发时代主机内存耗尽;现两道并发 + lane5 串接 | 守护 AUTORESTART + INCCOUNTS 告警 |
| 争用作废(P22 族显存) | 2(P22_relop_6/8) | lane5(0.45,lane4 完成后串接) | 0.45 两道并发 + resource_degraded 判级 | 大概率覆盖:先例 L1_P22__arith_replace__20 在 0.45 下 102/103 轮、随机 40/40 全过判 STILL;历史 OOM 发生在三道并发碎片期 | 若再 INC:转 lane3 尾部(0.92)或并入 P96 独占窗 |
| 合法拒判(equiv) | 0(不重跑) | — | — | 预注册口径保留 INCONCLUSIVE | — |

注:35 条唯一待跑 = 28(纯超时)+ 3(P96)+ 2(主机内存)+ 2(争用作废)= 32(超时档)+ 3(P96 仅在 envfix 档);P20_relop_4/7、P22_relop_6/8 同时在两档(双归因),按更针对性的归因行归类。

## 5. 看护升级(已上线)

- 远端守护升级 v9 → **v10.1**(14:41 部署,pidfile 流程:停 91232/91887 → 改 → 起 91971;两道驱动 PID 89920/90128 全程不变):进度行由 `cse lane L obs=N` 改为 `cse lane L obs=N inconclusive=I`,I = 该 lane 观测文件中 INCONCLUSIVE 计数,每 30 分钟写入 e1_cse_laneL.log。v9 备份于 e1_guard_cse_split_v9.bak.sh。
- 本地看护链路不变:_watch_cse_cycle.py 每 10 分钟独立统计各 lane INCONCLUSIVE 输出 INCCOUNTS 行,_watch_cse.py 对增量报 ALERT_INCONCLUSIVE_GROWTH。远端进度行与本地 INCCOUNTS 构成双通道。
- 当前基线:lane3=0,lane4=0,lane5=0(任何 >0 即告警并触发第二轮清算)。

## 5.1 运行期新增 INCONCLUSIVE(第二轮清算,2026-07-28 凌晨)

| probe_id | lane | class | 证据 | 归因 | 处置 |
|---|---|---|---|---|---|
| L1_P96__launch_config_mutate__0 | 3 | timeout | 3602s 超时,79 有效轮(随机 40/40 过,定向差 3 轮未达 42) | P96 族单轮成本高,3600s 近失(同族其余 10 条均达标判级) | 已在预案内:并入 CSE 收官后 P96 独占窗口(5400s、无并发、0.92) |
| L1_P53__arith_replace__17 | 3 | timeout(实为轮级挂起) | wall 2195s **未**超总预算;第 59 轮 `mixed_extremes` 触发 90s 轮级看门狗后 worker 卡死提前退出;已完成 58 轮(随机 40/40 全过,定向 18/42) | 见下【定性结论】:**原内核同挂**,`mixed_extremes` 对 P53 超大输入的**输入生成**超预算,与变异无关,合法拒判 | 记录不动(不改账);已上线 per-kernel 策略跳过(P53 跳 mixed_extremes/sparse/sparse_extreme);收官窗口从账本摘除本 id 后带跳过重跑,即可正常定级 |
| L1_P53__relop_replace__4 | 3 | other_unknown(实为轮级挂起) | wall 1954s **未**超总预算;同签名:第 59 轮 `mixed_extremes` 触发 90s 轮级看门狗后 worker 卡死提前退出;已完成 58 轮(随机 40/40 全过,定向 18/42) | 同上:原内核同挂,`mixed_extremes` 输入生成超预算,合法拒判 | 同上(记录不动;跳过机制已上线;收官摘 id 重跑) |
| L1_P53__const_perturb__0 | 3 | timeout(实为轮级挂起) | wall 2671s **未**超总预算;同签名:随机 40/40 全过,定向 18 过后 `mixed_extremes` 连续 3 轮 90s 看门狗超时(round_timeout)随后 worker 卡死 | 同上:原内核同挂,`mixed_extremes` 输入生成超预算,合法拒判 | 同上(记录不动;跳过机制已上线;收官摘 id 重跑) |

### 5.1.1 挂起主体定性结论(2026-07-28,原内核对照 + 缩比基准)

**判定:合法拒判(策略输入超出资源契约域),与变异无关。** 三条 P53 INCONCLUSIVE 同签名,均非变异体非终止,而是 `mixed_extremes` 策略对 P53 超大输入的**输入生成**本身超出轮级看门狗/cgroup 内存预算。证据链:

1. **原内核对照(0.18 显存配额、300s/150s 硬超时,不干扰在跑 lane3/lane4)**:对**未变异 P53 原内核**跑同款 `mixed_extremes`(worker 同 seed 50040+si):
   - si=0:输入生成阶段挂起 >150s(连 "input ready" 都未打印);
   - si=1:输入生成阶段被 cgroup OOM 于 92s SIGKILL(-9);
   - si=2:输入生成耗时约 120s 后完成,GPU forward 仅 0.003s。
   → **原内核与变异体表现完全一致**;GPU 前向永远秒回,挂点在 CPU 输入生成,与内核语义/变异无关。
2. **根因(读 `src/stress/policy_bank.py`)**:P53 输入 `torch.rand(128,4096,4095)` = 21.5 亿 float32 元素(单张量 8.6GB)。`_mixed_extremes` 对整张量做布尔掩码 fancy-index 散射两次(`values[mask]*=1e4` / `values[~mask]*=1e-4`,各约 10 亿元素),峰值 CPU 内存 30–40GB,在 120GB cgroup + 并发双 lane 下逼近 OOM 且远超 90s 轮级预算。`sparse`/`sparse_extreme` 同类(布尔掩码散射 + `mask.sum()` 遍历 21 亿 bool)。
3. **缩比基准(1/16 尺寸、峰值 ~0.5GB,安全)外推全尺寸生成耗时**:`sparse`≈115s、`mixed_extremes`≈64s(实测已 >120s/OOM,外推为下界)、`sparse_extreme`≈23s;其余 18 个策略均为廉价向量化(<40s,且生产中 P53 已实测通过前 6 个)。仅这 3 个做全张量布尔掩码散射。

**已实施处置(自动继续跑,未停机):**

- **代码**:`scripts/run_e1_cse_falsify.py` 新增 per-kernel 策略跳过(`load_policy_skip` 自动发现 `<out>/cse_policy_skip.json`;逐探针 `stress_policies` 剔除被跳策略;`grade_cse_evidence` 接收缩减后的 `stress_rounds_planned`)。**判级语义未放松**:stress 通过阈值仍为绝对值 `CSE_MIN_STRESS_ROUNDS=42`,计划轮 63→54,要求通过率由 42/63=66.7% 升至 42/54=77.8%(更严);跳过轮既不计过也不计败(不入 `trials`),不污染 resource_degraded 兜底。远端 CPU `pytest` 51 项全过(新增 `tests/test_cse_policy_skip.py` 6 项 + 既有 CSE/超时判级用例无回归)。
- **配置**:`/root/mk_v2_runs/e1/cse_policy_skip.json`:`{"L1_P53": ["mixed_extremes","sparse","sparse_extreme"]}`,含完整根因/证据留痕。
- **驱动重启(安全,账本完整)**:备份旧驱动 → 上传新代码/配置/测试 → py_compile + pytest 通过 → kill lane3 驱动(PID 89920)及其 worker(看护 91971、lane4 驱动 90128 全程不变);看护 v10.1 于 05:54:19 AUTORESTART 带新代码/配置拉起(新驱动 PID 127575,日志打印 "per-kernel policy skip active for ['L1_P53']",manifest 记 `policy_skip`)。被 kill 的 `const_perturb__4` 无账本条目(账本仅在探针完成时原子写入),遂带跳过逻辑干净重跑(新 cfg 仅 18 策略、已排除 3 个,已核对),避免再产 1 条废 INCONCLUSIVE。3 条已记录 INCONCLUSIVE 的账本条目**未改动**。

基线更新:lane3 现存运行期 INCONCLUSIVE = P96__launch_config_mutate__0(预算类,预案内)+ 3 条 P53(mixed_extremes 合法拒判,跳过机制已上线);均不进终版统计。lane3/4/5 剩余 P53:仅 **lane3 有 P53**(lane4/lane5 计划中 P53=0);活动队列剩余 7 条 P53(const_perturb__4 重跑中、__7、index_replace__8/__12、mask_boundary__0/__3/__5)将带跳过逻辑正常跑;另 3 条已记录 INCONCLUSIVE 的 P53 待收官窗口摘 id 重跑。哨兵 R1 阈值(4h 内 ≥3)—— 3 条 P53 均系同一 mixed_extremes 系统性成因、已对症闭环,不作为独立新增异常升级。

## 5.2 输入管线 bit-exact 优化部署留痕(2026-07-28 12:56,导师批准)

**变更**:CSE worker 输入生成管线优化(`cse_gen_opt_v1`):stress 轮单模板缓存(仅全浮点张量树,守卫回退)+ 去冗余外层 clone_tree + summaries 单遍 aminmax。**输入张量与 summaries 逐 bit 等价**(部署门禁:341 项张量 torch.equal + 336 项 summaries 逐 bit 对照全绿,覆盖 P1 全尺寸 / P53、P96 缩比全 21 策略×3 seed / P53 全尺寸 8.6GB / 混合树守卫路径);**判级语义零改动**(grade_cse_evidence、42/63 阈值、policy skip 均未触碰)。详见《E1_CSE_单探针耗时剖析与加速评估.md》第三部分。

- **代码**:`scripts/_mutant_worker.py`(~55 行)、`scripts/run_e1_cse_falsify.py`(8 行,manifest+记录加 `input_pipeline` 标记);policy_bank 零改动。新增 `tests/test_gen_opt_bitexact.py` 21 项;远端 CPU pytest **257 全过**无回归。备份 `*.bak_20260728_preopt`。
- **驱动重启(安全,账本完整)**:kill 前后 obs 行数(lane3=25/lane4=31)与 completed md5 逐字节一致;**先杀驱动(90127/90128/127575)后杀 worker**(避免旧驱动把半截结果写成 INCONCLUSIVE);守护 v10.1 于 12:56:37 AUTORESTART 拉起新驱动 lane3=146300 / lane4=146306;被杀 in-flight 探针(P35__arith_replace__16、P20__relop_replace__4)无账本条目,干净重跑。
- **部署后首两条探针**:P35__arith_replace__16 → **FALSIFIED**(witness structured_ramp/sub0/seed50040,证据链完整);P20__relop_replace__4(历史两次 cgroup SIGKILL 户)→ STILL_LIKELY_EQUIVALENT,**103/103 全轮完成**,wall 1865s(同族旧中位 3300-3600s,−45%)。无新增 INCONCLUSIVE,无 OOM(双 worker RSS 峰 99GB < 120GiB)。
- **口径**:**2026-07-28 12:56:37 为 wall_ms 断点**,此后记录带 `input_pipeline: "cse_gen_opt_v1"`,墙钟不可与之前直接比较;判级不受影响、跨断点可比。§4 对症配置矩阵中"3600s 预算"的覆盖判断只会更宽松(单探针成本 −45% 以上),P96 独占窗口(5400s)预计不再必要,届时按实际情况裁定。
- **收官预计更新**:lane3 剩 73 条 ≈ 36-40h;lane4 剩 17 条 ≈ 9h;lane5(54 条,串接自动继承新代码)≈ 28.5h;关键路径 ≈ **1.6 天**(原 ~3 天)。

## 6. 红线遵守

- 判级语义未改:重判试算只调用远端现行部署的预注册 `grade_cse_evidence`;本轮 0 条新写账。
- 已完成 STILL/FALSIFIED/WITNESSED 记录未动;归档文件原样保留。
- 守护修改走停-改-起,期间 lane3/4 驱动不受影响(PID 快照核对一致)。
- 未 git commit。
