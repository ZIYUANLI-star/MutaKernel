# KGB Extended (9 ops x fp16/fp32/bf16 x multi-shape, ATen baseline) —— 实验报告

> 结果目录:`runs/kgb_ext`  ·  LLM-free 运行(EMD 仅 L0/L1/L2,五维仅确定性维度)

## 0. 一句话结论

- **KGB 验证器存在漏检**,且 MutaKernel 五维能补杀其中一部分 —— 支持把 KGB 纳入论文作为新 subject。

- 变异分(mutation score, killed/(total−stillborn−strict_eq))= **54.6%**;真漏检 **354** 个,五维补杀 **163** 个,最终逃逸 **191** 个。


## 1. 总表(RQ-KGB-1/2/3)

| 指标 | 数值 |
|---|---|
| 有效种子 (subject) | 54 / 66 |
| 被测 kernel 数 | 54 |
| 变异体总数 total | 2664 |
| killed(KGB 验证器抓到) | 1424 |
| stillborn(编译/实例化失败) | 54 |
| survived(KGB 漏检,EMD 前) | 1186 |
| ├─ 真等价体 equivalent(EMD 剔除) | 832 |
| └─ **真漏检 true_escape** | 354 |
| 五维补杀 rescued | 163 |
| **最终逃逸 final_escape** | 191 |
| mutation score | 0.5456 |

## 2. 真漏检故障分类(RQ-KGB-2)

| 故障类别 | 数量 |
|---|---|
| tolerance_too_loose | 216 |
| dtype_or_numerics_gap | 135 |
| shape_coverage_gap | 2 |
| pattern_uncovered | 1 |

## 3. 五维增强补杀分解(RQ-KGB-3)

- 补杀率 = rescued / true_escape = 163/354 = **46.0%**

| 维度 | 补杀命中数 |
|---|---|
| value_stress | 163 |
| dtype_stress | 3 |
| config_stress | 1 |
| repeated_run | 1 |

> 说明:training_stress 对这些无状态纯函数内核标记 N/A;LLM 维度按本次决策跳过。


## 4. 分 subject 明细

| subject | total | killed | survived | stillborn | score |
|---|---|---|---|---|---|
| softmax__float16__128x512 | 31 | 12 | 18 | 1 | 0.400 |
| softmax__float16__2048x256 | 31 | 12 | 18 | 1 | 0.400 |
| rmsnorm__float16__512x1024 | 30 | 16 | 13 | 1 | 0.552 |
| layernorm__float16__128x512 | 49 | 24 | 24 | 1 | 0.500 |
| layernorm__float16__2048x256 | 49 | 25 | 23 | 1 | 0.521 |
| reduce__float16__256x4096 | 26 | 14 | 11 | 1 | 0.560 |
| matmul__float16__256x256x256 | 42 | 34 | 7 | 1 | 0.829 |
| cross_entropy__float16__256x512 | 48 | 17 | 30 | 1 | 0.362 |
| flash_attention__float16__1x2x128x64 | 124 | 83 | 40 | 1 | 0.675 |
| softmax__float32__128x512 | 31 | 12 | 18 | 1 | 0.400 |
| softmax__float32__2048x256 | 31 | 11 | 19 | 1 | 0.367 |
| rmsnorm__float32__512x1024 | 30 | 16 | 13 | 1 | 0.552 |
| layernorm__float32__128x512 | 49 | 25 | 23 | 1 | 0.521 |
| layernorm__float32__2048x256 | 49 | 25 | 23 | 1 | 0.521 |
| reduce__float32__256x4096 | 26 | 14 | 11 | 1 | 0.560 |
| cross_entropy__float32__256x512 | 48 | 16 | 31 | 1 | 0.340 |
| rotary_embedding__float32__1024x64 | 120 | 47 | 72 | 1 | 0.395 |
| flash_attention__float32__1x2x128x64 | 124 | 81 | 42 | 1 | 0.658 |
| softmax__bfloat16__128x512 | 31 | 12 | 18 | 1 | 0.400 |
| softmax__bfloat16__2048x256 | 31 | 12 | 18 | 1 | 0.400 |
| rmsnorm__bfloat16__512x1024 | 30 | 16 | 13 | 1 | 0.552 |
| layernorm__bfloat16__128x512 | 49 | 24 | 24 | 1 | 0.500 |
| layernorm__bfloat16__2048x256 | 49 | 24 | 24 | 1 | 0.500 |
| reduce__bfloat16__256x4096 | 26 | 14 | 11 | 1 | 0.560 |
| matmul__bfloat16__256x256x256 | 42 | 34 | 7 | 1 | 0.829 |
| cross_entropy__bfloat16__256x512 | 48 | 17 | 30 | 1 | 0.362 |
| flash_attention__bfloat16__1x2x128x64 | 124 | 83 | 40 | 1 | 0.675 |
| softmax__float16__1024x1024 | 31 | 10 | 20 | 1 | 0.333 |
| rmsnorm__float16__128x512 | 30 | 17 | 12 | 1 | 0.586 |
| rmsnorm__float16__2048x256 | 30 | 17 | 12 | 1 | 0.586 |
| layernorm__float16__512x1024 | 49 | 24 | 24 | 1 | 0.500 |
| reduce__float16__128x1024 | 26 | 14 | 11 | 1 | 0.560 |
| reduce__float16__1024x2048 | 26 | 14 | 11 | 1 | 0.560 |
| matmul__float16__512x256x512 | 42 | 34 | 7 | 1 | 0.829 |
| cross_entropy__float16__1024x1024 | 48 | 15 | 32 | 1 | 0.319 |
| flash_attention__float16__2x4x256x64 | 124 | 87 | 36 | 1 | 0.707 |
| softmax__float32__1024x1024 | 31 | 9 | 21 | 1 | 0.300 |
| rmsnorm__float32__128x512 | 30 | 17 | 12 | 1 | 0.586 |
| rmsnorm__float32__2048x256 | 30 | 17 | 12 | 1 | 0.586 |
| layernorm__float32__512x1024 | 49 | 24 | 24 | 1 | 0.500 |
| reduce__float32__128x1024 | 26 | 14 | 11 | 1 | 0.560 |
| reduce__float32__1024x2048 | 26 | 14 | 11 | 1 | 0.560 |
| cross_entropy__float32__1024x1024 | 48 | 15 | 32 | 1 | 0.319 |
| rotary_embedding__float32__256x128 | 120 | 47 | 72 | 1 | 0.395 |
| flash_attention__float32__2x4x256x64 | 124 | 85 | 38 | 1 | 0.691 |
| softmax__bfloat16__1024x1024 | 31 | 10 | 20 | 1 | 0.333 |
| rmsnorm__bfloat16__128x512 | 30 | 16 | 13 | 1 | 0.552 |
| rmsnorm__bfloat16__2048x256 | 30 | 16 | 13 | 1 | 0.552 |
| layernorm__bfloat16__512x1024 | 49 | 24 | 24 | 1 | 0.500 |
| reduce__bfloat16__128x1024 | 26 | 14 | 11 | 1 | 0.560 |
| reduce__bfloat16__1024x2048 | 26 | 14 | 11 | 1 | 0.560 |
| matmul__bfloat16__512x256x512 | 42 | 34 | 7 | 1 | 0.829 |
| cross_entropy__bfloat16__1024x1024 | 48 | 15 | 32 | 1 | 0.319 |
| flash_attention__bfloat16__2x4x256x64 | 124 | 87 | 36 | 1 | 0.707 |

## 5. 典型逃逸案例(最多 8 例)

- **softmax__float16__128x512** · `arith_replace`(A) @L42 · 故障类别=`tolerance_too_loose` · 五维补杀✓(value_stress)
- **rmsnorm__float16__512x1024** · `arith_replace`(A) @L32 · 故障类别=`tolerance_too_loose` · 五维补杀✓(value_stress)
- **rmsnorm__float16__512x1024** · `acc_downgrade`(C) @L28 · 故障类别=`dtype_or_numerics_gap` · 五维补杀✓(value_stress)
- **rmsnorm__float16__512x1024** · `epsilon_modify`(C) @L45 · 故障类别=`dtype_or_numerics_gap` · 五维补杀✓(value_stress)
- **rotary_embedding__float32__256x128** · `launch_config_mutate`(B) @L85 · 故障类别=`shape_coverage_gap` · 五维未补杀
- **rotary_embedding__float32__256x128** · `launch_config_mutate`(B) @L86 · 故障类别=`shape_coverage_gap` · 五维未补杀
- **softmax__bfloat16__1024x1024** · `broadcast_unsafe`(D) @L64 · 故障类别=`pattern_uncovered` · 五维未补杀

## 6. 范围与后续

- 本次:LLM-free;subject = AutoKernel 9 算子 × dtype × KGB shape 配置(ATen baseline)。

- 后续(需 LLM / 更多 GPU):EMD Layer3 + 五维 LLM 维度;KGB LLM/Agent track 生成 ATen-110 全量种子;vLLM(50)/cuBLAS(50)。

