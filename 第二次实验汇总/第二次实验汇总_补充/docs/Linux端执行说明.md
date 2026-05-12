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

## 5. 故障排查

| 现象 | 处理 |
|------|------|
| `ValidationException: max_tokens` | 检查 thinking_budget < max_tokens |
| `AccessDeniedException` | 在 Bedrock console 申请该 region 的 Anthropic 模型访问 |
| `not allowed from unsupported countries` | 服务器在中国大陆，需要境外代理 |
| `timeout/crash` 频发 | 把 STRESS_TIMEOUT 调大；或检查 nvidia-smi |
| Opus 响应慢 (>2 min/round) | 关闭 extended thinking 或调小 thinking_budget |
