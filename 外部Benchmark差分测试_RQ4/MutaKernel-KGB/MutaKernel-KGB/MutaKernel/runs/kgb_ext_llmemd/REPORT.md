# KGB 变异测试 · EMD Layer-3（LLM 语义等价判定）重跑报告

> 本次重跑只在 **等价变异体检测（EMD）** 中新增了第 3 层 —— 用 **Claude Opus 4.5**
> 对全部 1186 个存活变异体做语义等价性二次审查；**增强测试（五维压力）维度未引入任何
> LLM 分析**（按要求保持纯确定性差分），因此 killed / survived / stillborn 的原始
> 变异结果完全沿用既有 GPU 跑批数据，未重跑 GPU。

## 1. 背景与本次改动

原 `kgb_ext` 的 EMD 只跑到 Layer 0–2：

| 层 | 机制 | 判定 |
|----|------|------|
| L0 文本归一化 | CUDA-aware 源码归一化后比较 | STRICT_EQUIVALENT |
| L1 静态规则 | boundary_unreachable / dead_write / mask_noreach | STRICT_EQUIVALENT |
| L2 动态 bitwise | 100 随机 + 算子定向压力输入，逐位比对 | CANDIDATE_EQUIVALENT |
| **L3 LLM（本次新增）** | **Opus 4.5 语义等价审查** | 确认等价 / 推翻为真漏检 |

代码中 `EquivalentDetector` 早已声明 “Layer 3 LLM：对判为等价的变异体做二次审查（外部调用，
本模块不实现）”。本次即补齐这一外部层。

**判定准则（喂给 LLM 的定义）**：当且仅当对算子正常输入域内的**任意**输入（任意 shape /
dtype / 取值范围），变异体与原核产生数值相同的输出（仅允许不可避免的浮点重结合误差）时，
才判为等价；只要**存在**某一合法输入使输出可观测地不同，即判为**非等价（真漏检）**。

## 2. 头条结果

| 指标 | 原确定性 EMD（L0–L2） | 叠加 LLM（L0–L3） |
|------|----------------------|-------------------|
| 存活变异体总数 | 1186 | 1186 |
| 判为等价体合计 | **832** | **487** |
| ├ strict_equivalent | 215 | 215 |
| └ candidate_equivalent | 617 | 272 |
| **真漏检 true_escape** | **354** | **699** |
| 变异分（保守，仅扣 strict） | 0.5946 | 0.5946 |
| 变异分（乐观/LLM 校正，扣全部等价体） | **0.8009** | **0.6707** |
| 原始变异分（不扣等价体）| 0.5456 | 0.5456 |

**核心发现**：确定性统计 EMD **高估了等价体数量**。LLM 语义审查把 832 个“等价体”中的
**399 个候选等价体翻转为真实漏检**，同时把 354 个确定性漏检中的 **54 个确认为等价体**。
净效果：真漏检 **354 → 699（接近翻倍）**，等价体 **832 → 487**。
据此校正后的变异分从被高估的 **0.80 降到更可信的 0.67**。

> 含义：L2 在“100 随机 + 压力输入逐位一致”就判等价，会对**边界对齐 / 特定 shape / 越界 UB**
> 敏感的变异产生**假等价**——这些变异在被测输入上恰好一致，但语义上并不等价。Opus 4.5 的
> 语义分析能精确指出触发差异的输入条件，把它们纠正回真漏检。这反而**强化**了论文主旨：
> 即便是带定向压力的差分测试，也存在盲区。

## 3. 重分类明细（LLM 相对确定性 EMD）

| 重分类 | 数量 | 说明 |
|--------|------|------|
| candidate_equivalent → true_escape | **399** | L2 误判等价、LLM 判非等价（假等价被纠正） |
| survived(true_escape) → equivalent | **54** | L2 判漏检、LLM 判其实等价 |
| strict_equivalent 冲突（LLM 判非等价但保留 strict） | 10 | L0/L1 文本/静态等价为权威，记录冲突备查 |
| 缺 LLM 判定（回退确定性） | 0 | 1186 全部成功，零错误 |

置信度：LLM 对 1186 条判定**全部为 high 置信度**（等价 477 条、非等价 709 条均 high）。

## 4. 分算子（八类）真漏检对比

| 算子 | 存活 | 等价(确定性) | 真漏检(确定性) | 等价(LLM) | 真漏检(LLM) | 漏检变化 |
|------|------|------|------|------|------|------|
| softmax | 170 | 139 | 31 | 89 | 81 | +50 |
| rmsnorm | 113 | 33 | 80 | 9 | 104 | +24 |
| layernorm | 213 | 126 | 87 | 63 | 150 | +63 |
| reduce | 99 | 96 | 3 | 18 | 81 | **+78** |
| matmul | 28 | 28 | 0 | 7 | 21 | **+21** |
| cross_entropy | 187 | 154 | 33 | 126 | 61 | +28 |
| flash_attention | 232 | 162 | 70 | 67 | 165 | **+95** |
| rotary_embedding | 144 | 94 | 50 | 108 | 36 | **−14** |

`reduce`、`matmul` 跳变最显著：原本几乎全判等价（L2 在被测 shape 上 bitwise 一致），LLM 揭示
其中大量是边界/形状相关的真实缺陷。`rotary_embedding` 反向（−14），说明 LLM 也会把部分被
L2 判为漏检的变异确认为等价，并非单向加码。

## 5. 典型“假等价被纠正”案例（均 high 置信度）

- **`L0_P27__relop_replace__0`（reduce）**：`col_offsets < reduce_size` → `<= reduce_size`。
  当 `reduce_size` 为 `BLOCK_SIZE` 整数倍时，掩码多放进越界元素 → 越界读取。被测 shape 恰好
  未触发，故 L2 判等价；LLM 指出触发条件 → 真漏检。
- **`L0_P36__relop_replace__0/2/3`（matmul）**：A/B/K 维加载掩码 `< X` → `<= X`，当 X 为分块
  大小整数倍时越界读 garbage。属真实越界缺陷。
- **`L0_P27__arith_replace__10`（reduce）**：`total_rows = outer*inner` → `outer/inner`。仅在
  3D 张量、非末维归约时不同；测试输入均为 2D（inner=1），`*1==/1` 逐位一致 → L2 假等价。

> 说明性注记：部分“漏检”（如越界读 garbage 后在特定输入上恰好 bitwise 一致）属于在 KGB 与
> MutaKernel 压力输入均未覆盖的输入/内存态下才显现的差异。按“任意合法输入”的严格等价定义判为
> 非等价是正确的，这正是论文要刻画的盲区，报告中据实标注。

## 6. 产物清单（对齐 CUDA-L1 完整性）

```
runs/kgb_ext_llmemd/
├── summary.json        # 54 核 + 整体三种变异分（保守/乐观/LLM 校正）
├── emd_summary.json    # LLM 前后等价/漏检对比、重分类、分算子明细
├── checkpoint.json     # 完成度标记（54 核 / 2664 变异体 / 1186 LLM 判定）
└── details/*.json      # 54 个，每个变异体含 original_code、mutated_code、
                        #   equiv_detail.layer0/1/2/3 与 final_emd_status
```

`details/*.json` 中每个存活变异体新增：
```json
"equiv_detail": { "layer0": "...", "layer1": null, "layer2": {...},
  "layer3": { "llm_equivalent": true|false, "confidence": "high",
              "change_summary": "...", "reasoning": "...", "model": "claude-opus-4-5" } },
"final_emd_status": "killed|survived|stillborn|strict_equivalent|candidate_equivalent"
```

原始 `runs/kgb_ext/` 完整保留未改动；LLM 原始判定流水账见
`runs/kgb_ext/llm_emd/llm_verdicts.jsonl`（1186 条）。

## 7. 运行参数

- 模型：`us.anthropic.claude-opus-4-5-20251101-v1:0`（Bedrock，区域 us-west-2，bearer-token API key）
- 扩展思考：关闭（等价判定用结构化 JSON 推理；max_tokens=2048）
- 并发 5，1186 条用时约 1250s，零失败；断点续跑（按 uid 去重）
- 输入：每个变异体的 unified diff + 原核 + 变异核全文
