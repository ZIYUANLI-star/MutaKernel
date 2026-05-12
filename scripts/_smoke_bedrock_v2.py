"""Bedrock smoke test v2 - 用 boto3 + bedrock-runtime + 正确的 inference profile ID。"""
from __future__ import annotations
import json
import os
import sys
import time

key = os.environ.get("BEDROCK_API_KEY") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
if not key:
    print("ERR: BEDROCK_API_KEY env var empty", file=sys.stderr)
    sys.exit(2)
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = key

import boto3

REGION = os.environ.get("AWS_REGION", "us-west-2")
runtime = boto3.client("bedrock-runtime", region_name=REGION)

CANDIDATES = [
    ("Opus 4.7", "us.anthropic.claude-opus-4-7"),
    ("Opus 4.6", "us.anthropic.claude-opus-4-6-v1"),
    ("Opus 4.5", "us.anthropic.claude-opus-4-5-20251101-v1:0"),
    ("Opus 4.1", "us.anthropic.claude-opus-4-1-20250805-v1:0"),
    ("Sonnet 4.6", "us.anthropic.claude-sonnet-4-6"),
    ("Haiku 4.5", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
]


def try_model(label: str, model_id: str):
    print(f"\n[{label}]  {model_id}")
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
    }
    t0 = time.time()
    try:
        resp = runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
    except Exception as e:
        ms = (time.time() - t0) * 1000
        msg = str(e)[:300]
        print(f"  FAIL  {ms:.0f} ms  {msg}")
        return False

    ms = (time.time() - t0) * 1000
    payload = json.loads(resp["body"].read())
    text = ""
    for blk in payload.get("content", []):
        if blk.get("type") == "text":
            text = blk.get("text", "")
            break
    usage = payload.get("usage", {})
    print(f"  OK   {ms:.0f} ms")
    print(f"  Reply: {text!r}")
    print(f"  Usage: in={usage.get('input_tokens')}  out={usage.get('output_tokens')}")
    return True


if __name__ == "__main__":
    any_ok = False
    for label, mid in CANDIDATES:
        if try_model(label, mid):
            any_ok = True
    print("\n" + "=" * 60)
    print("RESULT:", "[OK] AT LEAST ONE MODEL WORKS" if any_ok else "[FAIL] NO MODEL WORKS")
    sys.exit(0 if any_ok else 1)
