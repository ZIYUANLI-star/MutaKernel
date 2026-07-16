# KGB Smoke (9 ops x fp16/fp32, ATen baseline) —— 实验报告

> 结果目录:`runs/kgb_smoke`  ·  LLM-free 运行(EMD 仅 L0/L1/L2,五维仅确定性维度)

## 0. 一句话结论

- **KGB 验证器存在漏检**,且 MutaKernel 五维能补杀其中一部分 —— 支持把 KGB 纳入论文作为新 subject。

- 变异分(mutation score, killed/(total−stillborn−strict_eq))= **54.1%**;真漏检 **45** 个,五维补杀 **25** 个,最终逃逸 **20** 个。


## 1. 总表(RQ-KGB-1/2/3)

| 指标 | 数值 |
|---|---|
| 有效种子 (subject) | 14 / 18 |
| 被测 kernel 数 | 14 |
| 变异体总数 total | 778 |
| killed(KGB 验证器抓到) | 413 |
| stillborn(编译/实例化失败) | 14 |
| survived(KGB 漏检,EMD 前) | 351 |
| ├─ 真等价体 equivalent(EMD 剔除) | 306 |
| └─ **真漏检 true_escape** | 45 |
| 五维补杀 rescued | 25 |
| **最终逃逸 final_escape** | 20 |
| mutation score | 0.5406 |

## 2. 真漏检故障分类(RQ-KGB-2)

| 故障类别 | 数量 |
|---|---|
| tolerance_too_loose | 26 |
| dtype_or_numerics_gap | 19 |

## 3. 五维增强补杀分解(RQ-KGB-3)

- 补杀率 = rescued / true_escape = 25/45 = **55.6%**

| 维度 | 补杀命中数 |
|---|---|
| value_stress | 25 |
| config_stress | 1 |
| dtype_stress | 1 |

> 说明:training_stress 对这些无状态纯函数内核标记 N/A;LLM 维度按本次决策跳过。


## 4. 分 subject 明细

| subject | total | killed | survived | stillborn | score |
|---|---|---|---|---|---|
| softmax__float16__128x512 | 31 | 12 | 18 | 1 | 0.400 |
| softmax__float32__128x512 | 31 | 12 | 18 | 1 | 0.400 |
| rmsnorm__float16__128x512 | 30 | 17 | 12 | 1 | 0.586 |
| rmsnorm__float32__128x512 | 30 | 17 | 12 | 1 | 0.586 |
| layernorm__float16__128x512 | 49 | 24 | 24 | 1 | 0.500 |
| layernorm__float32__128x512 | 49 | 25 | 23 | 1 | 0.521 |
| reduce__float16__128x1024 | 26 | 14 | 11 | 1 | 0.560 |
| reduce__float32__128x1024 | 26 | 14 | 11 | 1 | 0.560 |
| matmul__float16__256x256x256 | 42 | 34 | 7 | 1 | 0.829 |
| cross_entropy__float16__256x512 | 48 | 17 | 30 | 1 | 0.362 |
| cross_entropy__float32__256x512 | 48 | 16 | 31 | 1 | 0.340 |
| rotary_embedding__float32__1024x64 | 120 | 47 | 72 | 1 | 0.395 |
| flash_attention__float16__1x2x128x64 | 124 | 83 | 40 | 1 | 0.675 |
| flash_attention__float32__1x2x128x64 | 124 | 81 | 42 | 1 | 0.658 |

## 5. 典型逃逸案例(最多 8 例)

- **softmax__float16__128x512** · `arith_replace`(A) @L42 · 故障类别=`tolerance_too_loose` · 五维补杀✓(value_stress)
- **softmax__float32__128x512** · `arith_replace`(A) @L42 · 故障类别=`tolerance_too_loose` · 五维补杀✓(value_stress)
- **softmax__float32__128x512** · `stab_remove`(C) @L42 · 故障类别=`dtype_or_numerics_gap` · 五维补杀✓(value_stress)
- **cross_entropy__float32__256x512** · `acc_downgrade`(C) @L40 · 故障类别=`dtype_or_numerics_gap` · 五维未补杀

## 6. 范围与后续

- 本次:LLM-free;subject = AutoKernel 9 算子 × dtype × KGB shape 配置(ATen baseline)。

- 后续(需 LLM / 更多 GPU):EMD Layer3 + 五维 LLM 维度;KGB LLM/Agent track 生成 ATen-110 全量种子;vLLM(50)/cuBLAS(50)。

