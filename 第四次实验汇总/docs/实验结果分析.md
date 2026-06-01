# 第四次实验结果分析：CUDA-Agent 优化 Kernel 差异测试

> 日期：2026-05-10
> 数据集：BytedTsinghua-SIA/CUDA-Agent（commit d7b79785）
> 测试框架：MutaKernel 5 维压力差异测试

---

## 一、实验概况

| 项目 | 数值 |
|------|------|
| 数据集来源 | BytedTsinghua-SIA/CUDA-Agent |
| 原始条目 | 246（L1: 99, L2: 99, L3: 48） |
| 转换成功 | 222（108 纯 PyTorch + 114 CUDA→load_inline） |
| 实际完成 | 176 |
| 编译失败跳过 | 46 |
| 检测到差异 | **104 (59.1%)** |
| 运行时间 | 42.5 小时（单 H20 GPU） |

---

## 二、各 Level 增强检测率

| Level | 描述 | 完成 | 跳过 | 有差异 | 增强检测率 | baseline-pass 可靠差异 |
|-------|------|------|------|--------|-----------|---------------------|
| L1（基础） | 单算子优化 | 87 | 1 | 38 | **43.7%** | 37 |
| L2（中级） | 融合多步算子 | 62 | 32 | 46 | **74.2%** | 46 |
| L3（高级） | 复杂网络结构 | 27 | 11 | 20 | **74.1%** | 18 |
| **总计** | — | **176** | **46** | **104** | **59.1%** | **101** |

**关键发现**：Level 2/3（复杂优化）的增强检测率（74%）远高于 Level 1（44%），说明越复杂的 CUDA 融合优化越容易在边界条件下表现出行为差异。

---

## 三、5 维压力测试维度触发分析

| 维度 | 触发 kernel 数 | 总差异数 | 说明 |
|------|--------------|---------|------|
| training_stress | 94 | 2053 | 训练模式下梯度计算暴露差异 |
| value_stress | 90 | 1701 | 极端数值输入暴露计算差异 |
| repeated_run | 15 | 30 | 非确定性行为（随机性/并行归约顺序） |
| dtype_stress | 11 | 29 | 低精度下精度损失放大 |
| config_stress | 4 | 10 | 不同配置参数下行为不一致 |

- **value_stress** 和 **training_stress** 是主要差异来源（各覆盖 90+ kernel），说明 CUDA-Agent 的优化在极端数值和训练模式下最容易产生功能性差异
- **repeated_run** 触发 15 个 kernel（8.5%），这些优化引入了非确定性行为

---

## 四、差异严重程度分布

| 差异数范围 | kernel 数 | 占比 |
|-----------|----------|------|
| 0（无差异）| 72 | 41% |
| 1–10（轻微）| 23 | 13% |
| 11–30（中等）| 10 | 6% |
| 31–50（显著）| 33 | 19% |
| 51+（严重）| 38 | 22% |

超过 40% 的有差异 kernel 属于"严重"级别（51+ discrepancies），表明 CUDA-Agent 的部分优化在多种测试条件下均产生不一致结果。

---

## 五、编译失败分析

46 个 kernel 编译/运行失败（跳过），主要分布在 L2（32 个）和 L3（11 个）：
- L2/L3 的优化 kernel 大量使用 `import cuda_extension`（预编译 .so），转换为 `load_inline` 后部分因依赖复杂的 binding_registry 宏、硬编码维度等原因无法编译
- L1 几乎无失败（仅 1 个），因其优化多为纯 PyTorch 实现

---

## 六、与其他数据集对比

| 数据集 | 来源 | 完成数 | 增强检测率 | 主要差异维度 |
|--------|------|--------|-----------|-------------|
| CUDA-L1 | KernelBench Level 1 | 231 | 25.4% | value_stress, training_stress |
| AI CUDA Engineer | Sakana AI | 222 | 25.4% | value_stress, training_stress |
| TritonBench-G | Triton 社区 | 138 | 11.1% | value_stress |
| **CUDA-Agent** | **字节/清华 SIA** | **176** | **59.1%** | **value_stress, training_stress** |

**CUDA-Agent 的增强检测率（59.1%）是所有数据集中最高的**，约为 CUDA-L1 和 AI CUDA Engineer 的 2.3 倍，TritonBench-G 的 5.3 倍。

可能原因：
1. CUDA-Agent 的优化更激进（TF32 启用、算子融合、自定义 CUDA kernel）
2. 优化过程中可能未充分验证边界条件下的数值一致性
3. Level 2/3 的多步融合优化累积了精度偏差

---

## 七、典型案例

### 高差异 kernel 示例
- `cuda_agent__L2_T63`：68 处差异（value_stress + training_stress + repeated_run），GEMM+BN 融合优化
- `cuda_agent__L1_T82`：78 处差异，卷积+激活融合
- `cuda_agent__L3_T22`：56 处差异，DenseNet 结构优化

### 零差异 kernel 示例
- `cuda_agent__L1_T0`：非对称卷积，优化保持完美一致性
- `cuda_agent__L1_T4`：3D 卷积，简单 TF32 优化无精度影响

---

## 八、结论

1. CUDA-Agent 产生的优化 kernel 在压力测试下差异率极高（59.1%），表明其优化策略在数值稳定性方面存在显著风险
2. 复杂度越高的优化（L2/L3）风险越大（74% vs 44%），融合多步运算时精度损失会累积
3. MutaKernel 的 5 维测试框架能有效发现传统单元测试无法覆盖的行为差异，尤其是 value_stress 和 training_stress 维度
4. 建议 CUDA 优化工具在生成 kernel 后，增加极端数值和训练模式下的验证环节

---

## 九、文件位置

| 文件 | 路径 |
|------|------|
| checkpoint | `第四次实验汇总/results/checkpoint.json` |
| summary | `第四次实验汇总/results/summary.json` |
| 逐 kernel 详情 | `第四次实验汇总/results/details/` |
| 本文档 | `第四次实验汇总/docs/实验结果分析.md` |
| 数据转换脚本 | `scripts/prepare_cuda_agent.py` |
| 注册表 | `external_benchmarks/cuda_agent/registry.json` |
| SLURM 日志 | `slurm/logs/mk-cuda-agent_680783.out` |
