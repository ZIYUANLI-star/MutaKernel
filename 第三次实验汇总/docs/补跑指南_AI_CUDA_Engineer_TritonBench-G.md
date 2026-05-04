# 补跑指南：AI CUDA Engineer + TritonBench-G 增强测试

> 目标：在师弟的机器上把 **AI CUDA Engineer**（229 kernel）和 **TritonBench-G**（180 kernel，其中 142 可测）这两个外部数据集的 5 维增强测试跑完。
>
> **CUDA-L1 已有的实验结果完全不会受影响。**

---

## 一、整体流程（一条命令版）

把师兄发来的 `MutaKernel` 整个项目放到师弟机器上后，**只需要这一条命令**：

```bash
cd <MutaKernel 根目录>
bash 第三次实验汇总/run_scripts/run_补跑_sakana_tritonbench.sh
```

**完事**。脚本会自动完成：环境检查 → 数据完整性检查 → 已有结果保护性检查 → 冒烟测试（每数据集 2 个 kernel）→ 全量补跑 → 汇总打印。

中断了？再敲同一条命令，自动从断点续跑（不会重跑已完成的 kernel）。

---

## 二、为什么这次能跑通（与第一次失败的差异）

### 2.1 数据集已经在师兄电脑上完成"重新适配"

| 数据集 | 第一次失败原因 | 这次的修复 |
|-------|-------------|-----------|
| **AI CUDA Engineer** | 用 `PyTorch_Code_Module` 当参考，参数签名跟 CUDA 的 `forward(...)` pybind 对不上 | 改用 `PyTorch_Code_Functional`，并把 `module_fn` 的函数体替换成 `_ext.forward(*args, **kwargs)` 桩函数；renaming `Model → ModelNew`。`__init__` / `forward` 签名 100% 对齐 |
| **TritonBench-G** | `forward()` 参数数量与 `get_inputs()` 返回数量不匹配（6 NameError + 42 TypeError）；`autograd.Function` 类的 `forward` 被错误识别为顶层 wrapper | 重写 `_find_wrapper_callable`：autograd.Function 用 `ClassName.apply` / 别名；从原 TritonBench `test_*` 函数中挖出真实输入；`Model.forward`、`get_inputs()`、`ModelNew.forward` 三方对齐 |

CPU-only AST sanity check 结果（无需 GPU 也能跑，可在师弟电脑上重跑验证）：

```
AI CUDA Engineer: 229 kernel，全部 OK，0 bug
TritonBench-G:    180 kernel，全部 OK，0 bug
                  ├─ pytorch_ref: 26  （有 PyTorch 等价参考，做差分测试）
                  ├─ self_ref:   116 （无 PyTorch 等价；同一 Triton kernel 自参考，
                  │                    repeated_run 维度仍可检测非确定性）
                  └─ identity:    38 （恒等通路；run_fullscale_diff_test.py 自动过滤）
```

### 2.2 完全离线、不需要网络

- AI CUDA Engineer：`registry.json` (1.6 MB) + `problems/*.py` (229 个) + `parquet/*.parquet`（备份用，跑实验不必读）已全部在本地。
- TritonBench-G：`registry.json` + `problems/*.py` + `kernels/*.py` + `TritonBench_G_v1.json`（源数据）已全部在本地。
- 师弟**不需要**重新跑 `prepare_external_datasets.py`，**不需要**联网。

### 2.3 关于 `MutaKernel/cuda_extensions/` 这个文件夹

这其实**不是数据集**，而是 PyTorch JIT 编译 CUDA 扩展时留下的中间产物（`build.ninja`、`main.cpp`、`cuda.cu`）。运行测试时 `torch.utils.cpp_extension.load_inline()` 会在 PyTorch 自己的临时目录里再生成一份新的，这个老的目录用不上。

> **可以直接删掉，不影响任何东西。** 师兄已经在交付包里清理了它，你看到没有就对了。

---

## 三、师弟的电脑上要做的准备

### 3.1 必备依赖（如果之前跑过 CUDA-L1，应该已经有了）

```bash
pip install torch triton          # 必须
# pip install pandas pyarrow       # 不必须（这两个只在重新 prepare 时用，registry 已生成不必再 prepare）
```

师弟电脑上之前跑 CUDA-L1 已经装好了 torch；triton 装一下就行（CUDA-L1 不依赖它）。

### 3.2 GPU 架构

脚本会自动检测 GPU 架构（用 `torch.cuda.get_device_capability()`）。如需强制指定（极少数情况）：

```bash
bash 第三次实验汇总/run_scripts/run_补跑_sakana_tritonbench.sh 8.6   # RTX 3090 / A6000
bash 第三次实验汇总/run_scripts/run_补跑_sakana_tritonbench.sh 8.0   # A100
bash 第三次实验汇总/run_scripts/run_补跑_sakana_tritonbench.sh 8.9   # RTX 4090
bash 第三次实验汇总/run_scripts/run_补跑_sakana_tritonbench.sh 9.0   # H100
```

---

## 四、命令行参数

| 参数 | 作用 |
|------|------|
| `--skip-smoke` | 跳过冒烟测试，直接进全量。续跑时建议加这个，节省 5-10 分钟 |
| `--only sakana` | 只跑 AI CUDA Engineer |
| `--only tritonbench` | 只跑 TritonBench-G |
| `8.6` / `8.9` / `9.0` 等 | 强制指定 GPU 架构（一般不需要） |

例：

```bash
# 第一次正式跑（含冒烟测试）
bash 第三次实验汇总/run_scripts/run_补跑_sakana_tritonbench.sh

# 中断后续跑
bash 第三次实验汇总/run_scripts/run_补跑_sakana_tritonbench.sh --skip-smoke

# 想先把 sakana 跑完再跑 tritonbench
bash 第三次实验汇总/run_scripts/run_补跑_sakana_tritonbench.sh --only sakana
bash 第三次实验汇总/run_scripts/run_补跑_sakana_tritonbench.sh --only tritonbench --skip-smoke
```

---

## 五、断点续跑机制（重要）

`scripts/run_fullscale_diff_test.py` 内置了 checkpoint 机制：

- 每跑完一个 kernel，结果立刻写入 `第三次实验汇总/results/{数据集}/checkpoint.json`
- 重新启动时，自动读取 checkpoint，跳过 `kernel id` 已存在的所有项
- 所以**任何中断**（断电、Ctrl+C、网络掉线、机器重启、OOM）都能恢复

```
第三次实验汇总/results/
├── cuda_l1/                  ← 师弟之前跑完的，本脚本绝对不动
│   ├── checkpoint.json
│   ├── summary.json
│   └── details/*.json
├── ai_cuda_engineer/         ← 本次补跑写入
│   ├── checkpoint.json       ← 续跑用
│   ├── summary.json          ← 全跑完才写
│   └── details/*.json        ← 每个 kernel 一份详细 JSON
└── tritonbench_g/            ← 本次补跑写入
    ├── checkpoint.json
    ├── summary.json
    └── details/*.json
```

### 5.1 看进度（任何时候都可以跑）

```bash
python -c "
import json, os
base = '第三次实验汇总/results'
for ds in sorted(os.listdir(base)):
    cp = os.path.join(base, ds, 'checkpoint.json')
    if os.path.isfile(cp):
        with open(cp, encoding='utf-8') as f:
            data = json.load(f)
        completed = sum(1 for v in data.values() if isinstance(v, dict) and v.get('status') == 'COMPLETED')
        with_disc = sum(1 for v in data.values() if isinstance(v, dict) and v.get('total_discrepancies', 0) > 0)
        skipped = sum(1 for v in data.values() if isinstance(v, dict) and v.get('status') == 'SKIPPED')
        print(f'  {ds}: completed={completed}  with_disc={with_disc}  skipped={skipped}  total_in_checkpoint={len(data)}')
"
```

---

## 六、推荐用 tmux 跑（防止 SSH 掉线）

```bash
# 进入项目目录
cd <MutaKernel 根目录>

# 启动 tmux
tmux new -s exp

# 在 tmux 里跑
bash 第三次实验汇总/run_scripts/run_补跑_sakana_tritonbench.sh

# 断开 tmux：Ctrl+B 然后按 D
# 重连查看：tmux attach -t exp
```

预计耗时（参考师兄之前的 CUDA-L1：250 kernel 跑了约 14 小时）：

| 数据集 | 可测 kernel 数 | 预计总耗时（RTX 4090） |
|-------|--------------|-------------------|
| AI CUDA Engineer | 229 | ≈ 12-15 小时 |
| TritonBench-G | 142（identity 38 个自动跳过）| ≈ 7-9 小时 |
| **合计** | 371 | **≈ 20-24 小时** |

显存方面：每个 kernel 都是用子进程跑的，跑完即释放，所以 24 GB 显存（RTX 3090/4090）已经绰绰有余。

---

## 七、注意事项

1. **绝对不要**手动删除 `第三次实验汇总/results/` 下面的任何文件——`checkpoint.json` 是续跑命脉。
2. **绝对不要**用 `第三次实验汇总/run_scripts/run_all_experiments.sh`（旧版本一次性跑全部数据集，里面有"清 checkpoint 重跑冒烟"的逻辑，会**清掉 cuda_l1 的已有结果**）。请用本目录下的 `run_补跑_sakana_tritonbench.sh`。
3. 本脚本如果检测到 `cuda_l1/checkpoint.json` 存在，会在 Phase 3 打出明确的 "已有结果，本脚本不会触碰" 提示，可以放心。
4. 如果某个 kernel 跑了超过 3 小时（默认 `KERNEL_TIMEOUT=10800` s）还没结束，主调度会强行终止当前 kernel 并继续下一个，不会卡死整个流程。
5. 跑完后，`第三次实验汇总/results/{ai_cuda_engineer,tritonbench_g}/summary.json` 是给师兄做后续分析的最终产物，把这两个目录连同 `details/` 一起回传即可。

---

## 八、出问题怎么办

### 情况 1：脚本启动时报 "registry NOT FOUND"

数据没拷全。让师兄重发以下两个目录：

```
external_benchmarks/ai_cuda_engineer/registry.json
external_benchmarks/ai_cuda_engineer/problems/
external_benchmarks/tritonbench_g/registry.json
external_benchmarks/tritonbench_g/problems/
```

### 情况 2：冒烟测试过了，但全量跑到一半卡死

最常见原因是某个 kernel 编译卡住或非确定性死循环。直接：

```bash
# 在另一个终端
ps aux | grep run_fullscale_diff_test
kill <进程号>

# 然后续跑
bash 第三次实验汇总/run_scripts/run_补跑_sakana_tritonbench.sh --skip-smoke
```

卡住的那个 kernel 的 `kid` 不会进 checkpoint，下次会被自动重试。如果连续重试都卡，看日志 `第三次实验汇总/logs/full_补跑_{sakana|tritonbench}.log` 里最后一行的 kernel id，告诉师兄即可。

### 情况 3：triton 没装或版本不对

```bash
pip install -U triton
```

triton 的版本一般跟 PyTorch 配套的就行（PyTorch 2.1+ 自带 triton 2.x）。

---

## 九、文件清单（确认师兄发过来的内容）

师弟收到的项目根目录里至少需要有：

```
MutaKernel/
├── scripts/
│   ├── run_fullscale_diff_test.py       ← 主调度脚本
│   ├── _stress_worker.py                ← 子进程 worker
│   ├── _runtime_smoke_sakana.py         ← AST sanity check (sakana)
│   └── _runtime_smoke_tritonbench.py    ← AST sanity check (tritonbench)
├── src/                                  ← stress / mutengine / bridge 等核心
├── external_benchmarks/
│   ├── ai_cuda_engineer/
│   │   ├── registry.json                ← 必需
│   │   ├── problems/*.py                ← 必需 (229 个)
│   │   └── parquet/*.parquet            ← 备份，可选
│   ├── tritonbench_g/
│   │   ├── registry.json                ← 必需
│   │   ├── problems/*.py                ← 必需 (180 个)
│   │   ├── kernels/*.py                 ← 备份，可选
│   │   └── TritonBench_G_v1.json        ← 备份，可选
│   └── cuda_l1/                          ← 不动，已跑完
└── 第三次实验汇总/
    ├── docs/补跑指南_AI_CUDA_Engineer_TritonBench-G.md  ← 本文件
    ├── run_scripts/run_补跑_sakana_tritonbench.sh        ← 一键启动脚本
    └── results/cuda_l1/                                   ← 师弟已跑完，绝对不动
```

师弟可以先用以下命令快速核对（**不需要 GPU**，纯 Python AST）：

```bash
python scripts/_runtime_smoke_sakana.py
python scripts/_runtime_smoke_tritonbench.py
```

两条都打出 `Bugs: 0` 就说明数据完整、可以开跑。
