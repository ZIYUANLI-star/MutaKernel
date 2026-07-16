# MutaKernel 增强差分测试报告 (KGB · 无 LLM 维度)

> 在 EMD-LLM 分类之后，对全部**非严格存活体**施加五维增强差分测试
> （value/config/dtype/repeated；**不含 LLM 维度**），在 A800 上以子进程隔离并行执行。

## 1. 输入集合
- 真漏检 (true escape, LLM 确认): **699**
- 候选等价 (candidate equivalent, 作清白对照): **272**
- 严格等价 (文本归一化相同, 跳过): 215
- KGB baseline 已杀: 1424 · stillborn: 54

## 2. 核心结果
- **增强测试补杀真漏检: 248/699 = 35.5%**
- 增强后仍存活的真漏检: 451
- 候选等价体被增强测试杀掉(矛盾/反证): **0/272** → 0 矛盾即强力佐证等价分类正确

## 3. 变异分数演化 (conservative / optimistic)
| 指标 | KGB baseline | MutaKernel (baseline+增强) |
|---|---|---|
| 已杀变异体 | 1424 | 1672 |
| conservative 分数 | 0.5946 | 0.6981 |
| optimistic 分数 | 0.6707 | 0.7876 |

## 4. 按击杀维度 (首杀维度归因)
| 维度 | 首杀数 |
|---|---|
| value_stress | 217 |
| dtype_stress | 13 |
| config_stress | 12 |
| crash | 6 |

## 5. 按击杀策略 (任一维度命中策略)
| 维度:策略 | 命中变异体数 |
|---|---|
| config_stress:batch_1_s0 | 121 |
| dtype_stress:float16 | 116 |
| value_stress:near_zero | 93 |
| value_stress:large_magnitude | 69 |
| dtype_stress:bfloat16 | 53 |
| value_stress:structured_ramp | 21 |
| value_stress:near_overflow | 20 |
| config_stress:batch_2_s0 | 17 |
| config_stress:batch_1_s1 | 16 |
| dtype_stress:float32 | 11 |
| value_stress:extreme_magnitude | 6 |
| config_stress:batch_1_s2 | 5 |
| value_stress:all_negative | 5 |
| config_stress:batch_8_s0 | 5 |
| value_stress:sparse | 3 |
| config_stress:batch_8_s1 | 2 |
| config_stress:batch_32_s0 | 2 |
| config_stress:batch_64_s2 | 2 |
| config_stress:batch_8_s2 | 2 |
| config_stress:batch_2_s1 | 1 |
| config_stress:batch_16_s1 | 1 |
| config_stress:batch_4_s0 | 1 |
| config_stress:batch_32_s1 | 1 |
| config_stress:batch_64_s1 | 1 |
| config_stress:batch_4_s2 | 1 |

## 6. 按算子 (escape 补杀 / escape 总数)
| 算子族 | 补杀 | 真漏检总数 | 补杀率 |
|---|---|---|---|
| flash_attention | 36 | 165 | 22% |
| layernorm | 90 | 150 | 60% |
| rmsnorm | 83 | 104 | 80% |
| softmax | 18 | 81 | 22% |
| reduce | 6 | 81 | 7% |
| cross_entropy | 15 | 61 | 25% |
| rotary_embedding | 0 | 36 | 0% |
| matmul | 0 | 21 | 0% |

## 7. 说明
- crash 击杀已在**单线程纯隔离**下复核全部复现，确认为真实 CUDA 级崩溃（非并发误判）。
- 增强测试每个变异体在独立子进程中运行，OOB 崩溃只杀其自身进程，不污染其他用例。
- 运行参数: workers=32(单线程/worker), timeout=240s, 比较口径=逐位一致(bitwise)。
