#!/usr/bin/env python3
"""新 key 烟雾测试"""
import os
import sys
import time
from pathlib import Path

# 加载 .env
env_path = Path('/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/.env')
for line in env_path.read_text(encoding='utf-8').splitlines():
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1)
    os.environ[k] = v

print(f"AWS_REGION = {os.environ.get('AWS_REGION')}")
print(f"BEDROCK_MODEL_ID = {os.environ.get('BEDROCK_MODEL_ID')}")
print(f"API key suffix = ...{os.environ.get('BEDROCK_API_KEY','')[-12:]}")
print()

sys.path.insert(0, '/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel')
from src.stress.llm_clients import make_bedrock_caller

caller = make_bedrock_caller(
    model_id=os.environ['BEDROCK_MODEL_ID'],
    region=os.environ['AWS_REGION'],
    max_tokens=4096,
    thinking_budget=2000,
)

t0 = time.time()
print("=== 调用 Opus 4.5（新 key）===")
result = caller(
    "Return exactly the word 'PONG' (no quotes, no other text).",
)
elapsed = time.time() - t0
print(f"耗时: {elapsed:.1f}s")
print(f"返回: {result.get('text', '')[:200]!r}")
print(f"usage: {result.get('usage')}")
print(f"error: {result.get('error')!r}")

if result.get('error'):
    print("❌ 新 key 不可用")
    sys.exit(1)
else:
    print("✓ 新 key OK")
