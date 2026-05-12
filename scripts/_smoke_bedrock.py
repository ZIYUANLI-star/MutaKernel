"""Bedrock API key 连通性 smoke test。

走 Bedrock OpenAI 兼容端点 (https://bedrock-runtime.{region}.amazonaws.com/openai/v1/chat/completions)
依次尝试多个候选模型，告诉用户哪些可用、哪些需要 Model Access 审批。
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict

import requests

API_KEY = os.environ.get("BEDROCK_API_KEY", "").strip()
REGION = os.environ.get("AWS_REGION", "us-west-2").strip()

if not API_KEY:
    print("ERROR: BEDROCK_API_KEY env var is empty", file=sys.stderr)
    sys.exit(2)

ENDPOINT = f"https://bedrock-runtime.{REGION}.amazonaws.com/openai/v1/chat/completions"

CANDIDATES = [
    ("claude-opus-4-5", "us.anthropic.claude-opus-4-5-20250929-v1:0"),
    ("claude-opus-4-5 (no-prefix)", "anthropic.claude-opus-4-5-20250929-v1:0"),
    ("claude-opus-4", "us.anthropic.claude-opus-4-20250514-v1:0"),
    ("claude-sonnet-4-5", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
    ("claude-sonnet-4", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
    ("claude-3-5-sonnet-v2", "us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
    ("claude-3-5-haiku", "us.anthropic.claude-3-5-haiku-20241022-v1:0"),
]


USE_PROXY = os.environ.get("USE_SYSTEM_PROXY", "").lower() in ("1", "true", "yes")


def try_model(model_id: str) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
        "max_tokens": 16,
    }
    proxies = None if USE_PROXY else {"http": "", "https": ""}
    t0 = time.time()
    try:
        resp = requests.post(
            ENDPOINT, headers=headers, json=body, timeout=30, proxies=proxies
        )
    except Exception as e:
        return {"ok": False, "stage": "network", "error": str(e), "ms": (time.time() - t0) * 1000}

    elapsed_ms = (time.time() - t0) * 1000
    info = {
        "status": resp.status_code,
        "ms": round(elapsed_ms, 1),
    }
    try:
        data = resp.json()
    except Exception:
        data = {"_raw": resp.text[:500]}

    if resp.status_code == 200:
        try:
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {"ok": True, **info, "text": text, "usage": usage}
        except Exception:
            return {"ok": True, **info, "text": "(parse error)", "raw": data}

    err_msg = ""
    if isinstance(data, dict):
        err_msg = (
            data.get("message")
            or data.get("error", {}).get("message", "")
            or data.get("Message", "")
            or json.dumps(data)[:300]
        )
    return {"ok": False, **info, "error": err_msg, "raw": data}


def main():
    print(f"Endpoint  : {ENDPOINT}")
    print(f"Region    : {REGION}")
    print(f"API key   : {API_KEY[:20]}...{API_KEY[-8:]} (len={len(API_KEY)})")
    print("-" * 80)

    any_ok = False
    for label, mid in CANDIDATES:
        print(f"\n[{label}]  model_id={mid}")
        r = try_model(mid)
        if r["ok"]:
            any_ok = True
            print(f"  OK  HTTP {r['status']}  {r['ms']} ms")
            print(f"  Reply : {r['text']!r}")
            print(f"  Usage : {r.get('usage', {})}")
        else:
            stage = r.get("stage", "http")
            status = r.get("status", "?")
            err = r.get("error", "")[:200]
            print(f"  FAIL [{stage}/{status}]  {r.get('ms', 0)} ms")
            print(f"  Reason: {err}")

    print("\n" + "=" * 80)
    print("RESULT:", "AT LEAST ONE MODEL WORKS [OK]" if any_ok else "NO MODEL WORKS [FAIL]")
    if not any_ok:
        print(
            "\nLikely causes:\n"
            "  1. Model access not granted yet — go to Bedrock console -> Model access\n"
            "  2. Wrong region — confirm where you enabled the model\n"
            "  3. API key revoked / typo"
        )
    sys.exit(0 if any_ok else 1)


if __name__ == "__main__":
    main()
