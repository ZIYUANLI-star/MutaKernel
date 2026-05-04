# MutaKernel V3.0 实验改进方案

> **基于第二次实验结果的系统性分析与文献调研**
> **目标**: 针对当前 365 个存活变异体（123 个非等价存活 + 242 个候选等价），提出可落地的改进方向，显著提升变异杀死率
> **参考数据来源**: `第二次实验完整报告.md`、`EquivalentDetection_V2_Plan.md`、`StressEnhance_Plan.md`

---

## 一、当前实验瓶颈定量分析

### 1.1 总体现状

| 指标 | 数值 |
|------|------|
| 保守变异分数 (Phase 1+2) | 75.22% (1108/1473) |
| 乐观变异分数 (Phase 1+2) | 90.01% (1108/1231) |
| Phase 2 新增杀死 | 169 / 534 = 31.65% |
| **仍存活（非等价）** | **123** |
| **候选等价（未被杀死）** | **242** |

### 1.2 存活变异体分布——薄弱环节

**按 Tier 分析（Phase 2 存活）：**

| Tier | 总数 | 存活 | Kill Rate | 问题 |
|------|------|------|-----------|------|
| Tier 1 (L2 发现 bitwise 差异) | 151 | **23** | 84.8% | Layer 2 已找到差异，但差异在 allclose 容差内 |
| Tier 2 (LLM 认为可杀) | 119 | **100** | 16.0% | LLM 判断可杀，但实际未能杀死，杀伤率极低 |
| Tier 3 (候选等价) | 264 | **242** | 8.3% | 大量候选等价未被验证 |

**按算子分析（Phase 2 存活数最多的算子）：**

| 算子 | 进入增强 | 存活 | Kill Rate | 分析 |
|------|---------|------|-----------|------|
| relop_replace | 105 | 86 | 18.1% | 关系运算替换，边界条件难以触发 |
| mask_boundary | 85 | 62 | 27.1% | 掩码边界变异，GPU 线程调度导致不可达 |
| const_perturb | 96 | 61 | 36.5% | 常量扰动，部分常量为死代码 |
| arith_replace | 72 | 55 | 23.6% | 算术替换，浮点精度掩盖差异 |

**LLM 迭代分析效率极低：**

| 指标 | 数值 |
|------|------|
| LLM 触发变异体数 | 368 |
| LLM 杀死数 | 3 |
| LLM 杀死率 | 3/368 = **0.8%** |

### 1.3 核心瓶颈总结

1. **allclose 容差过宽**：固定 atol/rtol=1e-2 掩盖了微小但真实的数值差异（23 个 Tier 1 幸存者）
2. **输入策略是静态的**：21 种策略无反馈循环，无法根据执行结果自适应调整
3. **LLM 协议低效**：3 轮迭代 + 单模型 + 无执行反馈，ROI 极低
4. **精度维度不足**：仅测试 float16/bfloat16，未覆盖 TF32、FP64 等 CUDA 特有精度模式
5. **无梯度利用**：模型可微分但未利用梯度信息定向生成对抗性输入
6. **无覆盖率反馈**：不知道哪些 CUDA 分支未被覆盖

---

## 二、改进方向一：梯度引导的对抗性输入生成

### 2.1 文献依据

**GRIST** (Yan et al., ICSE 2021, Purdue):
- 首个利用梯度反向传播暴露深度学习程序数值 Bug 的动态技术
- 在 63 个真实 DL 程序上检测到 78 个 Bug（含 56 个未知 Bug），30 分钟内完成
- 核心思想：对目标数值操作定义 loss，通过梯度下降驱动输入向触发异常的方向移动

### 2.2 核心思路

MutaKernel 的 original 和 mutant 都是 PyTorch 模块，天然支持自动微分。我们可以直接利用梯度信息**最大化输出分歧**：

```
损失函数: L(x) = -||f_original(x) - f_mutant(x)||_2
梯度更新: x_{t+1} = x_t - lr * ∇_x L(x_t)
约束条件: x 保持在算子对应的合理值域范围内
```

### 2.3 具体方案

1. 对每个存活变异体，加载 original 和 mutant 模型
2. 用当前 value_stress 最佳策略的输出作为初始种子 `x_0`
3. 前向传播：分别计算 `y_orig = f_original(x)` 和 `y_mut = f_mutant(x)`
4. 计算散度损失：`loss = -torch.norm(y_orig - y_mut, p=2)`
5. 反向传播：`loss.backward()`，获取 `x.grad`
6. 更新输入：`x = x - lr * x.grad`，投影回合法范围
7. 迭代 50-100 步，检查每步的 allclose 结果
8. 如果 allclose 失败（差异超过容差）→ 判定 KILLED

### 2.4 预期效果

| 目标群体 | 数量 | 预期新增杀死 | 原因 |
|---------|------|------------|------|
| Tier 1 存活 | 23 | 10-15 | 已有 bitwise 差异，梯度可放大至超过 allclose 容差 |
| Tier 2 存活 | 100 | 5-15 | 梯度可找到 LLM 无法用语义推理发现的数值敏感点 |
| arith_replace 存活 | 55 | 10-20 | 算术操作最适合梯度放大 |

**实现难度**: 中等（需处理 CUDA JIT 编译 + 梯度计算的兼容性）

### 2.5 关键参考

```
@inproceedings{yan2021grist,
  title={Exposing Numerical Bugs in Deep Learning via Gradient Back-Propagation},
  author={Yan, Ming and Chen, Junjie and Zhang, Xiangyu and Tan, Lin},
  booktitle={ICSE},
  year={2021}
}
```

---

## 三、改进方向二：多级自适应容差级联

### 3.1 文献依据

**LLM4FP** (arxiv:2509.00256, 2025):
- 使用 bitwise 不等作为浮点不一致的判定标准
- 检测到 Varity 2.5 倍的浮点不一致

当前 MutaKernel 使用**固定**的 `allclose(atol=1e-2, rtol=1e-2)` 作为杀死判定标准，这个容差相对宽松。

### 3.2 核心思路

用**逐级收紧的容差级联**替代单一阈值，在多个精度级别报告变异分数：

| 级别 | 容差 | 含义 |
|------|------|------|
| Level 0 | allclose(atol=1e-1) | 极宽松（基线） |
| Level 1 | allclose(atol=1e-2) | 当前标准 |
| Level 2 | allclose(atol=1e-3) | 收紧 10 倍 |
| Level 3 | allclose(atol=1e-4) | 收紧 100 倍 |
| Level 4 | allclose(atol=1e-5) | 收紧 1000 倍 |
| Level 5 | bitwise_identical | 最严格（Layer 2 已有数据） |

### 3.3 价值

1. **学术价值**：论文中展示多级容差下的变异分数曲线，读者可自行判断合理阈值
2. **实际价值**：Level 2-4 的"近失杀"(near-miss) 变异体是梯度引导（方向一）的最佳目标
3. **23 个 Tier 1 存活变异体**在 Level 5 (bitwise) 已经失败，在 Level 2-4 可能也失败

### 3.4 实现方式

对已有的 534 个 detail JSON 文件重新计算（无需重新执行测试，只需重新比较输出）：
- 从 detail 文件中提取 original 和 mutant 的输出 tensor
- 对每个输出对用 5 个级别的容差重新判定
- 生成多级容差变异分数报告

**实现难度**: 低（纯后处理，不需要重新运行测试）

---

## 四、改进方向三：科学调试式 LLM 协议

### 4.1 文献依据

**Scientific Debugging for Mutation Testing** (Krodinger et al., arxiv:2503.08182, 2025):
- 使用"假设→实验→结论"的科学调试循环生成杀死变异体的测试
- LLM 一致性优于搜索式测试生成工具 Pynguin
- 关键发现：迭代细化是 LLM 测试生成成功的关键因素

**Meta ACH** (Harman et al., ICSE 2025):
- Meta 规模化部署的 mutation-guided LLM 测试生成系统
- LLM 等价检测 Agent：precision 0.79, recall 0.47（结合预处理后 0.95/0.96）
- 工程师接受率 73%

### 4.2 当前 LLM 协议的问题

当前 Phase 2 的 LLM 迭代分析：
- **仅 3 轮**迭代（太少）
- **单模型**（仅 DeepSeek-R1）
- **反馈信息不足**：只告知"未杀死"，不告知数值散度大小
- **无假设驱动**：LLM 没有明确的推理框架

### 4.3 改进方案：科学调试协议

```
Round 1: 假设形成
  输入: 变异代码 diff + 算子语义 + 所有已试策略及结果 + 数值散度报告
  LLM 任务: 分析变异的语义影响，形成"如何杀死"的假设
  输出: 假设描述 + 具体测试输入生成代码

Round 2-10: 实验-观察-细化循环
  执行: 运行 LLM 生成的测试输入
  反馈给 LLM:
    - 是否杀死 (pass/fail)
    - allclose 散度值 (max_diff, mean_diff)
    - 哪些 tensor 元素差异最大
    - original 和 mutant 的中间状态（如果可用）
  LLM 任务: 基于观察结果修正假设，生成新的测试输入
  终止条件: 杀死 OR 达到 10 轮 OR LLM 判定等价并给出证明
```

### 4.4 多模型集成

| 模型 | 角色 | 优势 |
|------|------|------|
| DeepSeek-R1 | 主推理模型 | 强推理链（现有） |
| GPT-4o | 辅助模型 | 代码生成质量高 |
| Claude 3.5 Sonnet | 辅助模型 | 长上下文理解 |

策略：三个模型独立生成测试输入，取**并集**（任一杀死即判定杀死）。

### 4.5 预期效果

| 目标群体 | 数量 | 当前杀死 | 预期新增杀死 |
|---------|------|---------|------------|
| Tier 2 存活 (LLM 认为可杀) | 100 | 0 | 10-20 |
| Tier 3 存活 (候选等价) | 242 | 0 | 5-10 |

**实现难度**: 中等（LLM API 调用 + 执行环境）

### 4.6 关键参考

```
@article{krodinger2025scientific,
  title={Scientific Debugging for Killing Mutants with LLMs},
  author={Krodinger, L. and others},
  journal={arXiv preprint arXiv:2503.08182},
  year={2025}
}

@inproceedings{harman2025ach,
  title={Mutation-Guided LLM-based Test Generation at Meta},
  author={Harman, Mark and others},
  booktitle={ICSE},
  year={2025}
}
```

---

## 五、改进方向四：跨精度差分测试

### 5.1 文献依据

**LLM4FP** (arxiv:2509.00256, 2025):
- LLM 引导的浮点程序生成，检测跨编译器/优化级别的数值不一致
- 不一致率 29.33%，是现有工具 Varity 的 2.5 倍

### 5.2 当前 dtype_stress 的局限

当前 `dtype_stress` 仅测试：
- float32 → float16
- float32 → bfloat16

未覆盖的 CUDA 特有精度模式：

| 精度模式 | 说明 | 潜在影响 |
|---------|------|---------|
| TensorFloat-32 (TF32) | Ampere+ 默认 matmul 精度，10-bit mantissa | 大量 matmul kernel 受影响 |
| FP64 (double) | 上转精度，放大微小差异 | 可暴露 1e-7 级差异 |
| torch.autocast (AMP) | 自动混合精度 | 不同操作用不同精度 |
| cudnn.benchmark | 不同卷积算法 | 非确定性差异 |
| cudnn.deterministic | 确定性模式 | 与 benchmark 对比 |

### 5.3 具体方案

```python
PRECISION_CONFIGS = [
    {"matmul_tf32": True,  "cudnn_tf32": True},   # TF32 开启（默认）
    {"matmul_tf32": False, "cudnn_tf32": False},   # TF32 关闭
    {"dtype": torch.float64},                       # FP64 上转
    {"autocast": True, "dtype": torch.float16},     # AMP float16
    {"autocast": True, "dtype": torch.bfloat16},    # AMP bfloat16
    {"cudnn_benchmark": True},                      # cudnn 自动选算法
    {"cudnn_deterministic": True},                  # 确定性模式
]
```

对每个配置：original 和 mutant 在相同精度环境下执行，比较输出。

### 5.4 预期效果

| 目标算子 | 存活数 | 预期新增杀死 | 原因 |
|---------|--------|------------|------|
| cast_remove | 12 | 3-5 | 精度变化直接影响类型转换行为 |
| epsilon_modify | 2 | 1-2 | epsilon 在低精度下影响更大 |
| arith_replace | 55 | 5-10 | FP64 可放大算术差异 |

**实现难度**: 低（在现有 worker 框架中添加精度配置参数）

---

## 六、改进方向五：覆盖率引导的模糊测试循环

### 6.1 文献依据

**cuFuzz** (NVIDIA, 2026):
- 首个实用化的 CUDA 覆盖率引导模糊测试工具
- 使用 NVBit 动态插桩收集 device-side 分支覆盖
- 发现 43 个未知 Bug（含 19 个商业库中的 Bug）
- 持久模式提升吞吐量

**CuFuzz** (arxiv:2601.01048, 2025):
- 将 CUDA 程序转换为 CPU 程序以启用 AFL 模糊测试
- PREX 优化实现平均 32x 加速

### 6.2 核心思路

当前 MutaKernel 的 21 种策略是**静态选择**的——不管测试结果如何，始终按固定顺序执行。引入覆盖率反馈后，可以**自适应**地选择更可能覆盖新分支的输入：

```
覆盖率引导循环:
1. 用 NVBit 插桩 mutant CUDA kernel
2. 执行当前策略，收集 device-side 分支覆盖
3. 对输入 tensor 施加 AFL 风格的变异（字节翻转、算术扰动、拼接）
4. 执行变异后输入，检查是否触发新的 device-side 分支
5. 保留触发新分支的输入到语料库
6. 对语料库中的输入检查 allclose 判定
7. 重复直到覆盖率饱和
```

### 6.3 预期效果

- **主要目标**：sync_remove、launch_config_mutate 等与 GPU 并行语义相关的算子
- 这些算子的杀死往往依赖于触发特定的线程交互路径，覆盖率引导可系统性地探索这些路径
- **预估新增杀死**：15-30 个

### 6.4 实施挑战

- NVBit 集成需要 NVIDIA GPU 和较新的驱动
- JIT 编译的 CUDA kernel 插桩需要额外处理
- 吞吐量受 NVBit 运行时开销影响（约 67% 降低）

**实现难度**: 高（NVBit 集成 + AFL 循环适配）

### 6.5 关键参考

```
@inproceedings{nvidia2026cufuzz,
  title={Hunting CUDA Bugs at Scale with cuFuzz},
  author={NVIDIA Research},
  year={2026}
}
```

---

## 七、改进方向六：进化式策略合成

### 7.1 文献依据

**EvoGPT** (arxiv:2505.12424, 2025):
- LLM 生成初始测试种群 + 进化算法优化
- 在 Defects4J 上覆盖率和变异分数均提升约 10%
- 关键发现：**显式强制多样性**（多温度 + 多指令）是关键

**SQUMUTH** (Springer, 2025):
- 松鼠搜索算法 (SSA) 生成高阶变异体
- 在 Validator 程序上变异分数达 97.27%

**MUTGEN** (arxiv:2506.02954, 2025):
- Mutation-guided LLM 测试生成
- 将变异反馈直接嵌入 prompt，显著优于 EvoSuite

### 7.2 核心思路

当前 21 种策略是**人工设计**的。用进化算法 + LLM 自动合成新策略：

```
1. 初始种群: 21 种现有策略的参数化表示
   - 每种策略 = (分布类型, 缩放因子, 稀疏度, 偏移, 正负比例)

2. 适应度函数: 该策略在验证子集上新杀死的变异体数

3. 遗传操作:
   - 交叉: 组合两个策略的参数（如 near_zero 的缩放 + structured_ramp 的模式）
   - 变异: 扰动参数（缩放因子 ×0.1~×10, 稀疏度 0~99%）
   - 选择: 锦标赛选择，保留杀死最多新变异体的策略

4. LLM 辅助:
   - 分析存活变异体的代码模式
   - 提议全新的策略概念（如"阶梯状递增+周期性尖峰"）
   - 将自然语言策略描述翻译为 policy_bank 代码

5. 算子定向进化: 为每个高存活率算子维护独立的策略种群
```

### 7.3 预期效果

| 目标算子 | 存活数 | 预期新增杀死 | 进化方向 |
|---------|--------|------------|---------|
| relop_replace | 86 | 8-15 | 更精确的边界命中策略 |
| const_perturb | 61 | 5-10 | 针对循环上界/grid 配置的策略 |
| arith_replace | 55 | 5-10 | 针对运算差异最大化的值域 |
| mask_boundary | 62 | 5-8 | 更多样的边界模式 |

**实现难度**: 中等

---

## 八、改进方向七：动态 Shape/配置探索

### 8.1 文献依据

**ATTest** (arxiv:2602.13987, 2025):
- 约束感知的 tensor 测试生成框架
- Agent 驱动的七阶段流水线，提取 tensor 约束并迭代生成-验证-修复
- PyTorch/TensorFlow 上平均分支覆盖率 55.6%

### 8.2 当前 config_stress 的局限

当前 config_stress 仅变化 `batch_size`（1, 2, 4, 8, 16, 32, 64），其他维度完全固定。但 CUDA kernel 的行为高度依赖于：

| 配置维度 | 当前覆盖 | 潜在影响 |
|---------|---------|---------|
| batch_size | 7 个值 | 改变 grid 大小 |
| 序列长度 | 未覆盖 | 改变 shared memory 使用模式 |
| 通道数 | 未覆盖 | 改变 warp 内线程映射 |
| 空间分辨率 | 未覆盖 | 改变 block 分解策略 |
| 奇数/素数维度 | 未覆盖 | 触发非对齐访问 |

### 8.3 具体方案

1. 用 LLM 分析每个 kernel 的 CUDA 代码，提取 shape 约束（哪些维度可变、范围）
2. 生成多样化的 shape 配置：
   - 素数维度（如 7, 13, 17, 31）：触发不同的 warp 对齐
   - 非 2 的幂次（如 33, 65, 129）：边界处理差异
   - 极小维度（1, 2, 3）：退化情况
   - 与原始 shape 的比值为非整数（如 1.5x, 0.7x）
3. 对每个 shape 配置执行 original vs mutant 比较

### 8.4 预期效果

- **主要目标**：launch_config_mutate（13 存活）、mask_boundary（62 存活）、index_replace（47 存活）
- **预估新增杀死**：5-10 个

**实现难度**: 中等（需要 per-kernel shape 约束分析）

---

## 九、改进方向八：变异体感知的差分执行分析

### 9.1 文献依据

**DiffGAN** (TSE 2025):
- GAN 引导的差分测试输入生成
- 生成 4 倍于 SOTA 基线的触发输入
- 使用 NSGA-II 多目标优化：多样性 + 散度

### 9.2 核心思路

对每个存活变异体进行**细粒度执行追踪比较**：

```
1. 中间值收集:
   - 在 original 和 mutant 的每个 CUDA kernel 启动点插入 hook
   - 记录每个 kernel 的输入/输出 tensor 值

2. 散度定位:
   - 找到 original 和 mutant 输出首次出现差异的最早 kernel
   - 记录散度的量级、位置（哪些 tensor 元素）

3. 反向切片:
   - 从散度点反向追踪，确定哪些输入元素影响了散度
   - 构建输入→散度的因果链

4. 定向输入构造:
   - 仅修改影响散度的输入元素
   - 最大化该子空间内的散度

5. 传播分析:
   - 判断识别出的散度是否能传播到最终输出
   - 是否能超过 allclose 容差
```

### 9.3 预期效果

- **主要目标**：23 个 Tier 1 存活变异体（已知存在 bitwise 散度）
- **预估新增杀死**：5-10 个

**实现难度**: 高（需要 CUDA kernel 执行追踪基础设施）

---

## 十、优先级排序与实施建议

### 10.1 按投入产出比排序

| 优先级 | 改进方向 | 目标群体 | 预估新增杀死 | 实现难度 | 建议阶段 |
|--------|---------|---------|------------|---------|---------|
| **P0** | 方向二：多级容差级联 | 全部 365 | 10-20（含报告价值） | 低 | 立即实施 |
| **P1** | 方向四：跨精度差分测试 | cast/epsilon/arith | 5-15 | 低 | 第一批 |
| **P2** | 方向一：梯度引导对抗输入 | 23 Tier1 + 100 Tier2 | 15-30 | 中 | 第一批 |
| **P3** | 方向三：科学调试 LLM 协议 | 100 Tier2 + 242 Tier3 | 10-20 | 中 | 第一批 |
| **P4** | 方向六：进化策略合成 | relop/const_perturb | 10-25 | 中 | 第二批 |
| **P5** | 方向七：动态 Shape 探索 | launch/mask/index | 5-10 | 中 | 第二批 |
| **P6** | 方向八：差分执行分析 | 23 Tier 1 | 5-10 | 高 | 第二批 |
| **P7** | 方向五：覆盖率引导模糊测试 | 全部 365 | 15-30 | 高 | 第三批 |

### 10.2 预期总体效果

| 场景 | 新增杀死估计 | 新保守分数 | 新乐观分数 |
|------|-----------|----------|----------|
| 仅 P0+P1（低成本） | 15-35 | 76.2-77.6% | 91.2-92.8% |
| P0-P3（第一批全部） | 40-85 | 77.9-81.0% | 93.3-97.0% |
| 全部 P0-P7 | 65-140 | 79.6-84.7% | 95.3-100% |

### 10.3 实施路线图

```
第一阶段（1-2 周）—— 快速见效
├── P0: 多级容差级联（纯后处理，1-2 天）
├── P1: 跨精度差分测试（扩展 dtype_stress，3-5 天）
└── P2: 梯度引导（核心算法开发，5-7 天）

第二阶段（2-4 周）—— 深度改进
├── P3: 科学调试 LLM 协议（prompt 重构 + 多模型，5-7 天）
├── P4: 进化策略合成（GA 框架 + LLM 集成，5-7 天）
└── P5: 动态 Shape 探索（shape 约束提取 + 配置生成，3-5 天）

第三阶段（4-8 周）—— 基础设施级改进
├── P6: 差分执行分析（中间值追踪框架，7-10 天）
└── P7: 覆盖率引导模糊测试（NVBit 集成 + AFL 循环，10-14 天）
```

---

## 十一、完整参考文献

| 编号 | 文献 | 会议/期刊 | 年份 | 核心贡献 |
|------|------|---------|------|---------|
| [1] | Yan et al., "Exposing Numerical Bugs in Deep Learning via Gradient Back-Propagation" | ICSE | 2021 | 梯度引导数值 Bug 触发 |
| [2] | NVIDIA, "Hunting CUDA Bugs at Scale with cuFuzz" | - | 2026 | CUDA 覆盖率引导模糊测试 |
| [3] | Krodinger et al., "Scientific Debugging for Killing Mutants with LLMs" | arXiv | 2025 | 假设-实验-结论式 LLM 杀变异体 |
| [4] | Harman et al., "Mutation-Guided LLM-based Test Generation at Meta" | ICSE | 2025 | 工业规模 LLM 变异引导测试生成 |
| [5] | LLM4FP, "LLM-Guided Floating-Point Program Generation" | arXiv | 2025 | LLM 引导浮点不一致检测 |
| [6] | EvoGPT, "Leveraging LLM-Driven Seed Diversity" | arXiv | 2025 | LLM + 进化算法混合测试生成 |
| [7] | MUTGEN, "Mutation-Guided Unit Test Generation with LLM" | arXiv | 2025 | 变异反馈嵌入 LLM prompt |
| [8] | Xia et al., "Fuzz4All: Universal Fuzzing with Large Language Models" | ICSE | 2024 | LLM 作为通用模糊测试引擎 |
| [9] | Tian et al., "LLMs for Equivalent Mutant Detection" | ISSTA (Distinguished Paper) | 2024 | LLM 等价变异体检测 F1 提升 35.69% |
| [10] | ATTest, "Agent-Driven Tensor Testing for DL Library Modules" | arXiv | 2025 | 约束感知 tensor 测试生成 |
| [11] | DiffGAN, "Differential Testing of DNNs for Image Analysis" | TSE | 2025 | GAN+NSGA-II 差分测试输入生成 |
| [12] | CuFuzz, "Hardening CUDA Programs through Transformation and Fuzzing" | arXiv | 2025 | CUDA→CPU 转换 + AFL 模糊测试 |
| [13] | SQUMUTH, "Squirrel Search Based Algorithm for High Order Mutant Generation" | Springer | 2025 | 搜索式高阶变异体生成 |
