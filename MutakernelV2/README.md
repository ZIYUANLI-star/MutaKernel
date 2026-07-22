# MutakernelV2 — V2 版本工作区

> 创建日期: 2026-07-21
> 用途: 存放 MutaKernel V2（FSE 投稿版本）的补充实验数据、重跑实验数据，以及 V2 的方法修正思路文档。

## 目录结构

```
MutakernelV2/
├── README.md                    ← 本文件
├── MutakernelV2方法修正/         ← V2 方法更新思路与方法学设计文档（不含代码）
│   │
│   │  ―― 思路文档（决策依据与演化记录）――
│   ├── 00_V2总体叙事与投稿定位.md
│   ├── 01_判杀口径与Oracle修正思路.md
│   ├── 02_EMD等价置信度与人工审计思路.md
│   ├── 03_实验矩阵_预算匹配基线与统计思路.md
│   ├── 04_历史数据有效性分级与重跑优先级.md
│   ├── 05_反例搜索引擎统一框架与按内核变异引导.md
│   │
│   │  ―― 方法学设计文档（V2 各模块权威规格，可据此从零构建代码）――
│   ├── 方法V2_00_总体架构与模块桥接.md          ← 模块划分 + 全部模块间 Schema（权威）
│   ├── 方法V2_01_执行基座_状态控制与严格Oracle.md   ← M1
│   ├── 方法V2_02_任务契约与输入策略库.md          ← M2
│   ├── 方法V2_03_变异探针生成与位点指纹.md        ← M3
│   ├── 方法V2_04_RealismGuard与分类学维护.md      ← M4
│   ├── 方法V2_05_EMD等价证据链与三层输出.md       ← M5
│   ├── 方法V2_06_反例搜索引擎与三路差分判定.md     ← M6
│   ├── 方法V2_07_审计模式_盲区测量与FaultToStress映射.md ← M7
│   ├── 方法V2_08_验证模式_位点定向候选验证器.md    ← M8
│   └── 方法V2_09_人工审计与统计分析桥接.md        ← M9
└── 实验/                        ← V2 实验总目录
    ├── FSE实验设计_审稿人视角的实验数据蓝图.md   ← 论文 Evaluation 章节 camera-ready 底稿（英文，数据以 ? 占位，跑完直接填表）
    ├── 附_实验设计内部说明_非论文稿.md          ← 内部设计依据（基线取舍理由、成本预估、审稿关切映射，不随论文发布）
    ├── 补充实验数据/             ← V2 新增实验的数据（预算匹配矩阵、外部基线、人工审计标注等）
    └── 重跑实验数据/             ← 历史实验（Phase I/II、RQ4 等）的重跑数据
```

## 与历史目录的关系

- 历史实验数据（`内部变异实验_RQ1_RQ2_RQ3_RQ5/`、`外部Benchmark差分测试_RQ4/`、
  `修复实验_CUDA-Agent_TaskD/`、`实验_LLM真实错误性归类/`）**保持不可变**，
  只作为 pilot 证据引用，不得原地修改（见 `docs/LEGACY_RESULT_VALIDITY.md` 的保存策略）。
- V2 的每一份新数据必须带 run manifest（commit、环境指纹、契约版本、种子、预算），
  与历史 checkpoint 严格分目录，不混合汇总。
- 代码层面的修正记录在 `docs/FSE_CODE_REMEDIATION.md`；
  本目录下的方法修正文档记录的是**方法学思路**，两者互补。

## 关键上游文档索引

| 文档 | 内容 |
|------|------|
| `docs/FSE_REVISION_PLAN.md` | FSE 改版总计划（RQ 重构、工作包 P0–P5、go/no-go 门槛） |
| `docs/LEGACY_RESULT_VALIDITY.md` | 历史数字（939/534/166/222/104 等）的有效性分级登记 |
| `docs/FSE_CODE_REMEDIATION.md` | 代码修复的逐文件记录与审稿意见映射 |
| `docs/FSE_EXPERIMENT_RUNBOOK.md` | 实验执行操作规程（Gate 0–8） |
| `docs/HUMAN_CALIBRATION_PROTOCOL.md` | 四个人工审计人群（A/B/C/D）的标注协议 |
| `docs/MANUAL_VERIFICATION_TODO.md` | 尚未完成的人工验证工作清单 |
| `docs/RELATED_BASELINE_INTEGRATION.md` | robust-kbench / KernelBenchX / ProofWright 集成协议 |
| `configs/fse_strategy_matrix.json` | 预注册的 32 调用预算匹配策略矩阵 |
