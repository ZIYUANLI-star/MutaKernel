"""KGB EMD Layer-3: LLM (Opus 4.5) semantic-equivalence verification.

Reads the 1186 KGB survivors (survived + strict/candidate equivalent),
asks Opus 4.5 whether each mutant is functionally equivalent to the
original kernel for ALL valid inputs, and writes a JSONL checkpoint.

Resumable: already-judged mutants (keyed by uid) are skipped.

Usage:
    python scripts/run_kgb_llm_emd.py --limit 8 --workers 4    # validation
    python scripts/run_kgb_llm_emd.py --workers 6              # full run
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- credentials / network (user's own Bedrock key + local Clash proxy) ---
os.environ["AWS_BEARER_TOKEN_BEDROCK"] = (
    "ABSKQmVkcm9ja0FQSUtleS02YzVyLWF0LTQyMDQ3Nzk2MDk0MzpjZlBHbGdLWitVbTRh"
    "MHUxSDdHVWNMa05kd0pxOGJZbG9ER0U0SVZLa1RjS3BYS09yT2ZoRGFSd2s1VT0="
)
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from stress.llm_clients import make_bedrock_caller  # noqa: E402

BASE = os.path.join(
    ROOT, "外部Benchmark差分测试_RQ4", "MutaKernel-KGB", "MutaKernel-KGB",
    "MutaKernel", "runs", "kgb_ext",
)
IN_FILE = os.path.join(BASE, "llm_emd", "survivors_input.json")
OUT_JSONL = os.path.join(BASE, "llm_emd", "llm_verdicts.jsonl")

MODEL = "us.anthropic.claude-opus-4-5-20251101-v1:0"
REGION = "us-west-2"
MAX_CODE_CHARS = 9000

_print_lock = threading.Lock()
_write_lock = threading.Lock()


def uid(s: dict) -> str:
    return f"{s['kernel_file']}::{s['id']}"


def build_prompt(s: dict) -> str:
    oc = s["original_code"]
    mc = s["mutated_code"]
    diff = "\n".join(difflib.unified_diff(
        oc.splitlines(), mc.splitlines(),
        fromfile="original", tofile="mutant", lineterm="", n=4,
    ))
    if len(diff) > 6000:
        diff = diff[:6000] + "\n... [diff truncated] ..."
    oc_t = oc if len(oc) <= MAX_CODE_CHARS else oc[:MAX_CODE_CHARS] + "\n... [truncated] ..."
    mc_t = mc if len(mc) <= MAX_CODE_CHARS else mc[:MAX_CODE_CHARS] + "\n... [truncated] ..."
    return f"""You are an expert in GPU kernel programming (Triton / CUDA) and \
compiler semantics. Your job is mutation-testing equivalence analysis.

A "mutant" is the original kernel with one small fault injected by operator \
`{s['operator_name']}` (category: {s['operator_category']}).
Mutation description: {s['description']}

DEFINITION — the mutant is EQUIVALENT to the original if and only if, for EVERY \
valid input in the operator's normal domain (any shape, dtype, and value range \
the kernel is meant to support), the mutant produces numerically the SAME output \
as the original (differences only at the level of unavoidable floating-point \
re-association that cannot change any reasonable correctness check).

If there exists ANY valid input for which the outputs differ observably, the \
mutant is NOT equivalent (it is a real fault that a test suite SHOULD catch).

Be rigorous and conservative: only answer "equivalent": true when you are \
confident the change cannot affect output for any valid input (e.g. a change in \
a comment, dead code, an unreachable branch, a redundant boundary guard, a \
no-op re-association). When the change touches arithmetic, indexing, masking, \
reduction, dtype/precision, or control flow that can affect results for some \
input, answer "equivalent": false.

UNIFIED DIFF (original -> mutant):
```diff
{diff}
```

ORIGINAL KERNEL:
```python
{oc_t}
```

MUTATED KERNEL:
```python
{mc_t}
```

Respond with ONLY a single JSON object, no prose, in exactly this schema:
{{"equivalent": <true|false>, "confidence": "<high|medium|low>", \
"change_summary": "<one sentence: what the mutation changed>", \
"reasoning": "<2-4 sentences: why it is or isn't equivalent, citing the \
specific input condition that would expose a difference if not equivalent>"}}"""


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    m = _JSON_RE.search(t)
    if not m:
        raise ValueError(f"no JSON in response: {text[:200]!r}")
    obj = json.loads(m.group(0))
    if "equivalent" not in obj:
        raise ValueError(f"missing 'equivalent': {obj}")
    obj["equivalent"] = bool(obj["equivalent"])
    obj.setdefault("confidence", "medium")
    obj.setdefault("change_summary", "")
    obj.setdefault("reasoning", "")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only-status", default="", help="filter deterministic status")
    args = ap.parse_args()

    with open(IN_FILE, encoding="utf-8") as f:
        survivors = json.load(f)
    if args.only_status:
        survivors = [s for s in survivors if s["deterministic_emd_status"] == args.only_status]

    done = set()
    if os.path.exists(OUT_JSONL):
        with open(OUT_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["uid"])
                except Exception:
                    pass

    todo = [s for s in survivors if uid(s) not in done]
    if args.limit > 0:
        todo = todo[:args.limit]

    with _print_lock:
        print(f"survivors={len(survivors)} already_done={len(done)} todo={len(todo)} "
              f"workers={args.workers}", flush=True)
    if not todo:
        print("nothing to do.", flush=True)
        return

    caller = make_bedrock_caller(
        model_id=MODEL, region=REGION, max_tokens=2048,
        enable_thinking=False,
    )

    counter = {"n": 0, "eq": 0, "neq": 0, "err": 0}
    t_start = time.time()

    def work(s: dict) -> dict:
        prompt = build_prompt(s)
        t0 = time.time()
        try:
            resp = caller(prompt)
            verdict = parse_verdict(resp["content"])
            rec = {
                "uid": uid(s),
                "kernel_file": s["kernel_file"],
                "kernel_name": s["kernel_name"],
                "id": s["id"],
                "operator_name": s["operator_name"],
                "operator_category": s["operator_category"],
                "deterministic_emd_status": s["deterministic_emd_status"],
                "llm_equivalent": verdict["equivalent"],
                "llm_confidence": verdict.get("confidence", ""),
                "llm_change_summary": verdict.get("change_summary", ""),
                "llm_reasoning": verdict.get("reasoning", ""),
                "usage": resp.get("usage", {}),
                "latency_ms": round((time.time() - t0) * 1000),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            rec = {
                "uid": uid(s),
                "kernel_file": s["kernel_file"],
                "id": s["id"],
                "operator_name": s["operator_name"],
                "operator_category": s["operator_category"],
                "deterministic_emd_status": s["deterministic_emd_status"],
                "llm_equivalent": None,
                "error": str(e)[:300],
                "latency_ms": round((time.time() - t0) * 1000),
            }
        return rec

    with open(OUT_JSONL, "a", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, s): s for s in todo}
        for fut in as_completed(futs):
            rec = fut.result()
            with _write_lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
            counter["n"] += 1
            if rec.get("error"):
                counter["err"] += 1
            elif rec["llm_equivalent"]:
                counter["eq"] += 1
            else:
                counter["neq"] += 1
            if counter["n"] % 5 == 0 or counter["n"] == len(todo):
                rate = counter["n"] / max(1e-9, time.time() - t_start)
                with _print_lock:
                    print(f"[{counter['n']}/{len(todo)}] eq={counter['eq']} "
                          f"neq={counter['neq']} err={counter['err']} "
                          f"{rate:.2f}/s eta={ (len(todo)-counter['n'])/max(1e-9,rate):.0f}s",
                          flush=True)

    with _print_lock:
        print(f"DONE n={counter['n']} eq={counter['eq']} neq={counter['neq']} "
              f"err={counter['err']} elapsed={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
