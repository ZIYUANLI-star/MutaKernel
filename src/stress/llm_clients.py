"""Unified LLM client interface for MutaKernel.

Supports:
- Bedrock (Anthropic Claude via boto3 invoke_model, with extended thinking)
- DeepSeek / OpenAI-compatible (via openai SDK, with reasoner support)

All callers return a unified dict:
    {
      "content": str,             # final answer (JSON expected)
      "reasoning_content": str,   # CoT / thinking
      "model": str,               # actual model id returned
      "usage": {                  # token usage
          "prompt_tokens": int,
          "completion_tokens": int,
          "total_tokens": int,
          "reasoning_tokens": int,
      },
      "latency_ms": float,
      "http_status": int,
    }
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bedrock (Anthropic Claude)
# ---------------------------------------------------------------------------
def make_bedrock_caller(
    model_id: str,
    region: str = "us-west-2",
    max_tokens: int = 16384,
    enable_thinking: bool = True,
    thinking_budget: int = 8000,
    api_key: Optional[str] = None,
) -> Callable[[str], Dict[str, Any]]:
    """Create a Bedrock caller using the Anthropic message API.

    Args:
        model_id: Bedrock inference profile ID, e.g.
                  ``us.anthropic.claude-opus-4-5-20251101-v1:0``
        region: AWS region.
        max_tokens: Max tokens for response.
        enable_thinking: Whether to enable extended thinking.
        thinking_budget: Budget for thinking tokens (when enabled).
        api_key: Bedrock API key (will be set as AWS_BEARER_TOKEN_BEDROCK env
                 var; can also be left None to use existing env var).

    Returns:
        A callable that takes a single prompt string and returns the unified
        response dict.
    """
    if api_key:
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
    elif "AWS_BEARER_TOKEN_BEDROCK" not in os.environ:
        # Try BEDROCK_API_KEY as alias.
        alt = os.environ.get("BEDROCK_API_KEY")
        if alt:
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = alt

    import boto3
    import random
    from botocore.config import Config
    from botocore.exceptions import ClientError

    # Extended thinking can take a long time (especially for hard mutants).
    # boto3 built-in adaptive retries was empirically insufficient when
    # Bedrock per-region capacity was exhausted; ~52% of Task C rounds were
    # throttled. We now layer an explicit retry loop with exponential backoff
    # + jitter on top of the boto3 retry, specifically to absorb
    # ThrottlingException bursts that span > 1 minute.
    boto_config = Config(
        connect_timeout=30,
        read_timeout=600,
        retries={"max_attempts": 6, "mode": "adaptive"},
    )
    client = boto3.client(
        "bedrock-runtime", region_name=region, config=boto_config,
    )

    # Sanity-check: Bedrock requires max_tokens > thinking.budget_tokens.
    effective_max_tokens = max_tokens
    if enable_thinking and effective_max_tokens <= thinking_budget:
        effective_max_tokens = thinking_budget + 4096
        logger.warning(
            "max_tokens (%d) <= thinking_budget (%d); bumping max_tokens to %d",
            max_tokens, thinking_budget, effective_max_tokens,
        )

    # Outer retry policy for transient errors that survive boto3 retries.
    # Sequence (s): 5, 15, 30, 60, 120, 240 with ±20% jitter.
    OUTER_MAX_RETRIES = 6
    BACKOFFS = [5.0, 15.0, 30.0, 60.0, 120.0, 240.0]

    def _is_retryable(exc: Exception) -> bool:
        msg = str(exc)
        if isinstance(exc, ClientError):
            err_code = exc.response.get("Error", {}).get("Code", "")
            if err_code in (
                "ThrottlingException",
                "TooManyRequestsException",
                "ServiceUnavailableException",
                "ModelTimeoutException",
                "InternalServerException",
                "ModelErrorException",
            ):
                return True
        # boto3 wraps some retryable conditions in generic Exception.
        return any(s in msg for s in (
            "ThrottlingException",
            "Throttled",
            "TooManyRequests",
            "ServiceUnavailable",
            "ReadTimeoutError",
            "Connection",
        ))

    def call(prompt: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": effective_max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if enable_thinking:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            # Extended thinking requires temperature=1 if specified.
            # Bedrock default is fine; we just don't pass temperature.

        t0 = time.time()
        last_exc: Optional[Exception] = None
        resp = None
        for attempt in range(OUTER_MAX_RETRIES):
            try:
                resp = client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                )
                break  # success
            except Exception as e:
                last_exc = e
                if not _is_retryable(e) or attempt == OUTER_MAX_RETRIES - 1:
                    elapsed_ms = (time.time() - t0) * 1000
                    raise RuntimeError(
                        f"Bedrock invoke_model failed in {elapsed_ms:.0f} ms "
                        f"after {attempt+1} attempts: {e!s}"
                    ) from e
                base = BACKOFFS[min(attempt, len(BACKOFFS) - 1)]
                sleep_s = base * (0.8 + 0.4 * random.random())
                logger.warning(
                    "Bedrock retryable error on attempt %d/%d (sleep %.1fs): %s",
                    attempt + 1, OUTER_MAX_RETRIES, sleep_s, str(e)[:160],
                )
                time.sleep(sleep_s)
        elapsed_ms = (time.time() - t0) * 1000

        payload = json.loads(resp["body"].read())
        content_text = ""
        thinking_text = ""
        for blk in payload.get("content", []):
            btype = blk.get("type", "")
            if btype == "text":
                content_text += blk.get("text", "")
            elif btype == "thinking":
                thinking_text += blk.get("thinking", "")
            # ignore other types (e.g. tool_use)

        usage = payload.get("usage", {}) or {}
        prompt_tokens = usage.get("input_tokens", 0) or 0
        completion_tokens = usage.get("output_tokens", 0) or 0
        # Bedrock does not break out reasoning tokens separately on Anthropic.
        # We approximate via thinking_text length only (informational).
        reasoning_tokens = 0
        if thinking_text:
            # Rough heuristic: 1 token ≈ 4 chars (English). Used only as proxy.
            reasoning_tokens = max(1, len(thinking_text) // 4)

        return {
            "content": content_text,
            "reasoning_content": thinking_text,
            "model": payload.get("model", model_id),
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "reasoning_tokens": reasoning_tokens,
            },
            "latency_ms": elapsed_ms,
            "http_status": 200,
            "raw": payload,  # keep raw for debugging; callers may drop it
        }

    return call


# ---------------------------------------------------------------------------
# OpenAI-compatible (DeepSeek / OneAPI / OpenRouter / Anthropic gateway etc.)
# ---------------------------------------------------------------------------
def make_openai_compat_caller(
    model: str,
    api_key: str,
    api_base: str = "https://api.deepseek.com",
    max_tokens: int = 16384,
    temperature: Optional[float] = 0.3,
    is_reasoner_hint: Optional[bool] = None,
) -> Callable[[str], Dict[str, Any]]:
    """Create an OpenAI-compatible caller (DeepSeek-R1, gateway proxies, etc.).

    Returns a dict matching the unified schema.
    """
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=api_base)
    is_reasoner = (
        is_reasoner_hint
        if is_reasoner_hint is not None
        else "reasoner" in model.lower()
    )

    def call(prompt: str) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        if not is_reasoner and temperature is not None:
            kwargs["temperature"] = temperature

        t0 = time.time()
        resp = client.chat.completions.create(**kwargs)
        elapsed_ms = (time.time() - t0) * 1000

        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", "") or ""

        u = resp.usage
        usage_dict: Dict[str, int] = {}
        if u:
            usage_dict = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }
            ctd = getattr(u, "completion_tokens_details", None)
            if ctd:
                usage_dict["reasoning_tokens"] = (
                    getattr(ctd, "reasoning_tokens", 0) or 0
                )
        usage_dict.setdefault("reasoning_tokens", 0)

        return {
            "content": content,
            "reasoning_content": reasoning,
            "model": resp.model or model,
            "usage": usage_dict,
            "latency_ms": elapsed_ms,
            "http_status": 200,
        }

    return call


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def make_caller(
    provider: str,
    model_id: str,
    **kwargs: Any,
) -> Callable[[str], Dict[str, Any]]:
    """Factory: provider in {bedrock, deepseek, openai_compat}."""
    p = provider.lower()
    if p == "bedrock":
        return make_bedrock_caller(model_id=model_id, **kwargs)
    if p in ("deepseek", "openai_compat"):
        return make_openai_compat_caller(model=model_id, **kwargs)
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Helper: load config from .env file
# ---------------------------------------------------------------------------
def load_env_file(path: Optional[str] = None) -> Dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ and return them.

    If path is None, looks for ``.env`` in the current dir and 2 parents up.
    Lines starting with ``#`` are ignored. Values are not stripped of quotes.
    """
    if path is None:
        cur = os.getcwd()
        for _ in range(3):
            candidate = os.path.join(cur, ".env")
            if os.path.isfile(candidate):
                path = candidate
                break
            cur = os.path.dirname(cur)
    if not path or not os.path.isfile(path):
        return {}

    loaded: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            # strip surrounding quotes if any
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            loaded[k] = v
            os.environ.setdefault(k, v)
    return loaded


__all__ = [
    "make_bedrock_caller",
    "make_openai_compat_caller",
    "make_caller",
    "load_env_file",
]
