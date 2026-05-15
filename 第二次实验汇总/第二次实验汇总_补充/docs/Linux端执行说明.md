# Linux 端执行说明（kbuser 服务器）

> 三个补强任务在 Linux GPU 机器上跑（开发在 Windows，git pull 后此处运行）。

## 0. 前置依赖

```bash
cd /home/kbuser/projects/KernelBench-0   # 即 MutaKernel 仓库的 Linux 副本

# 拉最新代码
git pull

# 安装依赖（如尚未装）
pip install boto3 openai
```

## 1. 凭证 & 网络

```bash
# 把 Windows 端 .env 的内容粘过来；KEY 不要落进版本控制。
# 实际 KEY 见 Windows 项目根目录 .env (gitignored)，或从 AWS Bedrock 控制台重新生成。
export BEDROCK_API_KEY='<PASTE_YOUR_BEDROCK_API_KEY_HERE>'
export AWS_BEARER_TOKEN_BEDROCK="$BEDROCK_API_KEY"
export AWS_REGION=us-west-2
export BEDROCK_MODEL_ID='us.anthropic.claude-opus-4-5-20251101-v1:0'

# 如服务器本身在境外（如 AWS / GCP），无需代理。如在中国大陆，请配境外代理：
# export HTTP_PROXY=http://<your-proxy>:<port>
# export HTTPS_PROXY=http://<your-proxy>:<port>

# 验证 Bedrock 可用：
python scripts/_smoke_bedrock_v2.py
```

## 2. Task A：Opus 4.5 重跑 Phase II 后 365 个未杀死变异体

```bash
# 烟测（1 个 mutant，全 5 轮）— 必跑
python scripts/run_taskA_phase2_rerun.py \
    --mutant-id L1_P100__arith_replace__7 \
    --rounds 5 \
    --no-resume \
    --out-dir "第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/_linuxsmoke"

# 检查 _linuxsmoke/details/<id>.json 看 killed/rounds/execution_result 正常后继续
rm -rf "第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/_linuxsmoke"

# 全量 365 个（5 轮，支持断点续跑）
nohup python scripts/run_taskA_phase2_rerun.py --rounds 5 \
    > 第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/run.log 2>&1 &

# 监控：
tail -f 第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/run.log

# 中断后续跑（completed.json 自动跳过已完成的）：
python scripts/run_taskA_phase2_rerun.py --rounds 5
```

## 3. Task C：Opus 4.5 直接挑战 534 个 Phase I 后未杀死变异体

```bash
# 烟测（1 个 mutant，全 5 轮）— 必跑
python scripts/run_taskC_phase1_direct.py \
    --mutant-id L1_P1__relop_replace__2 \
    --rounds 5 \
    --no-resume \
    --out-dir "第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/_linuxsmoke"

# 确认 _linuxsmoke/details/<id>.json 里 killed/rounds/execution_result 正常
rm -rf "第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/_linuxsmoke"

# 全量 534 个（5 轮，支持断点续跑）
nohup python scripts/run_taskC_phase1_direct.py --rounds 5 \
    > 第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/run.log 2>&1 &

# 监控：
tail -f 第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/run.log

# 可选：只跑某个 Phase I 状态子集（survived 或 candidate_equivalent）
python scripts/run_taskC_phase1_direct.py --rounds 5 --filter-status survived
```

## 4. Task B（待 Task A/C 完成后再开发，预计另外 1-1.5 天）

`scripts/run_taskB_kernel_strengthen.py`（44 个 kernel，引入 Task A/C 的杀死信息）。

## 4.5 Task A 补强：3 个原 LLM-only 杀死的变异体

> 论文最终方法学移除了 Phase II 的 DeepSeek-R1 兜底，因此原本只被该兜底杀掉的 3 个变异体，
> 在新方法学下应视为 Phase II 存活，需要走 Task A（Opus 4.5, 5 轮，extended thinking）审计。
>
> 目标：`L1_P49__arith_replace__11`、`L1_P49__init_modify__0`、`L1_P23__init_modify__0`
>
> 输出会**直接归并**到 `task_a_phase2_rerun/details/` 现有目录，不需要单独统计。

```bash
cd /home/kbuser/projects/KernelBench-0
git pull

# 烟测（dry-run，不调用 LLM，确认 3 个 mutant 都能解析到 Phase I/II）
python scripts/run_taskA_3_extra.py --dry-run

# 正式跑（约 2-4 分钟，3 个 mutant × 最多 5 轮）
python scripts/run_taskA_3_extra.py --rounds 5 \
    > 第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/run_extra3.log 2>&1

# 检查结果
tail -n 20 第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/run_extra3.log
ls 第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details/L1_P49__arith_replace__11.json \
   第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details/L1_P49__init_modify__0.json \
   第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details/L1_P23__init_modify__0.json

# 单独的 manifest 落在 task_a_phase2_rerun/run_manifest_extra3.json，
# 原 run_manifest.json 不被覆盖。
```

跑完后告诉我每个 mutant 的 `killed` / `rounds` 摘要（或我直接读 details JSON），我会重新统计 Task A 总数（365 → 368）、
`reason_category` 分布、`provably_equivalent` / `operationally_indistinguishable` / `ourtool-missed` 计数，并同步更新
论文 RQ3 + 所有相关数字宏。

## 5. 故障排查

| 现象 | 处理 |
|------|------|
| `ValidationException: max_tokens` | 检查 thinking_budget < max_tokens |
| `AccessDeniedException` | 在 Bedrock console 申请该 region 的 Anthropic 模型访问 |
| `not allowed from unsupported countries` | 服务器在中国大陆，需要境外代理 |
| `timeout/crash` 频发 | 把 STRESS_TIMEOUT 调大；或检查 nvidia-smi |
| Opus 响应慢 (>2 min/round) | 关闭 extended thinking 或调小 thinking_budget |
