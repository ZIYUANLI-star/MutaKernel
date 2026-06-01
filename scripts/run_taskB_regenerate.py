#!/usr/bin/env python3
"""Task B: Opus 4.5 regenerates kernels that pass KernelBench-default but
fail Phase II stress testing on specific inputs.

Per kernel: 3-round iterative repair.
- Round 1: prompt = reference PyTorch Model + buggy kernel + all failing inputs
           (with diff_summary collected in Round 0 verify pass).
- Round 2-3: same context + previous round's kernel + still-failing cases.

Each round runs dual verification:
- V_stress: replay every failing (policy, seed, mode), require allclose
            with atol=rtol=1e-2 (matches Phase II tolerance).
- V_kb:     KernelBench default get_inputs() across 5 seeds, atol=rtol=1e-3.

A kernel is "fixed" iff both V_stress and V_kb pass entirely in some round.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORKER = SCRIPT_DIR / "_stress_worker.py"
PY_INTERP = sys.executable

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

BUGGY_JSON = (PROJECT_ROOT / "第二次实验汇总" / "第二次实验汇总_补充" /
              "task_b_buggy_kernels_from_existing_data.json")
TRAIN_SUPP_JSON = (PROJECT_ROOT / "第二次实验汇总" / "第二次实验汇总_补充" /
                   "task_b_regenerate" / "_train_refok_supplement.json")
BEST_KERNELS_JSON = PROJECT_ROOT / "best_kernels.json"
OUT_DIR = (PROJECT_ROOT / "第二次实验汇总" / "第二次实验汇总_补充" /
           "task_b_regenerate")

STRESS_TIMEOUT = 240
ATOL_STRESS, RTOL_STRESS = 1e-2, 1e-2
ATOL_KB,     RTOL_KB     = 1e-3, 1e-3
KB_SEEDS = [42, 1337, 7, 100, 2024]


# ============================================================
# 0.  Utility
# ============================================================
def find_problem_file(level: str, problem_id) -> Optional[Path]:
    pid = str(problem_id)
    pdir = PROBLEM_DIRS.get(level)
    if pdir is None or not pdir.exists():
        return None
    for f in pdir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def extract_python_code(content: str) -> str:
    """Extract the largest python fenced code block.

    Falls back to the longest 'class ModelNew' code if no fence is found.
    """
    if not content:
        return ""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content,
                        flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return max(blocks, key=len).strip()
    # Heuristic fallback
    if "class ModelNew" in content:
        idx = content.find("class ModelNew")
        # Walk back to last "import" or top
        start = content.rfind("\nimport ", 0, idx)
        if start == -1:
            start = content.rfind("\nfrom ", 0, idx)
        if start == -1:
            start = max(0, idx - 200)
        return content[start:].strip()
    return ""


# ============================================================
# 1.  Verification (subprocess wrapping _stress_worker.py)
# ============================================================
def _run_stress_worker(cfg: Dict[str, Any],
                       timeout: int = STRESS_TIMEOUT) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="taskb_worker_") as td:
        cp = os.path.join(td, "cfg.json")
        rp = os.path.join(td, "res.json")
        with open(cp, "w") as f:
            json.dump(cfg, f)
        try:
            proc = subprocess.run(
                [PY_INTERP, "-u", str(WORKER), cp, rp],
                cwd=str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                    "error": f"TIMEOUT({timeout}s)"}
        try:
            with open(rp) as f:
                return json.load(f)
        except FileNotFoundError:
            return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                    "error": (f"worker no output rc={proc.returncode}: "
                              f"{proc.stderr[:200]}")}


def verify_one_input(new_kernel_code: str, problem_file: Path,
                     policy: str, seed: int, mode: str,
                     atol: float, rtol: float) -> Dict[str, Any]:
    """Re-run the failing (policy, seed, mode) with the *new* kernel as 'original'.

    Returns: {passed, ref_ok, new_ok, diff_summary, error}
    """
    cfg_mode = "training_stress" if "train" in mode else "value_stress"
    cfg = {
        "mode": cfg_mode,
        "device": "cuda",
        "atol": atol, "rtol": rtol,
        "policy_name": policy, "seed": seed,
        "kernel_code": new_kernel_code,
        "mutated_code": new_kernel_code,
        "problem_file": str(problem_file),
        "sync_weights": True,
    }
    r = _run_stress_worker(cfg)
    ref_ok = bool(r.get("ref_ok"))
    new_ok = bool(r.get("original_ok"))
    passed = ref_ok and new_ok
    return {
        "policy": policy, "seed": seed, "mode": mode,
        "passed": passed, "ref_ok": ref_ok, "new_ok": new_ok,
        "diff_summary": r.get("diff_summary", ""),
        "error": r.get("error", ""),
        "time_ms": r.get("time_ms"),
    }


def verify_kb_default(new_kernel_code: str, problem_file: Path,
                      seeds: List[int] = KB_SEEDS) -> Dict[str, Any]:
    """KernelBench default inputs (no policy) across multiple seeds."""
    per_seed = []
    for s in seeds:
        cfg = {
            "mode": "value_stress",
            "device": "cuda",
            "atol": ATOL_KB, "rtol": RTOL_KB,
            "policy_name": "__identity__", "seed": s,
            "kernel_code": new_kernel_code,
            "mutated_code": new_kernel_code,
            "problem_file": str(problem_file),
            "sync_weights": True,
        }
        r = _run_stress_worker(cfg)
        passed = bool(r.get("ref_ok")) and bool(r.get("original_ok"))
        per_seed.append({
            "seed": s, "passed": passed,
            "ref_ok": bool(r.get("ref_ok")),
            "new_ok": bool(r.get("original_ok")),
            "diff_summary": r.get("diff_summary", ""),
            "error": r.get("error", ""),
        })
    n_pass = sum(1 for x in per_seed if x["passed"])
    return {
        "n_total": len(seeds),
        "n_pass": n_pass,
        "all_pass": n_pass == len(seeds),
        "per_seed": per_seed,
        "regressed_seeds": [x["seed"] for x in per_seed if not x["passed"]],
    }


def verify_stress_set(new_kernel_code: str, problem_file: Path,
                      failing_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_input = []
    n_pass = n_skip = 0
    for fi in failing_inputs:
        r = verify_one_input(
            new_kernel_code, problem_file,
            fi["policy"], fi["seed"], fi["mode"],
            ATOL_STRESS, RTOL_STRESS,
        )
        if not r["ref_ok"]:
            n_skip += 1
            r["status"] = "ref_failed"
        elif r["passed"]:
            n_pass += 1
            r["status"] = "pass"
        else:
            r["status"] = "fail"
        per_input.append(r)
    n_total = len(failing_inputs)
    n_fail = n_total - n_pass - n_skip
    return {
        "n_total": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_ref_failed_skipped": n_skip,
        "all_pass": (n_fail == 0 and n_pass > 0),
        "per_input": per_input,
        "still_failing": [x for x in per_input if x["status"] == "fail"],
    }


# ============================================================
# 2.  Prompt construction
# ============================================================
SYSTEM_HINT = (
    "You are a GPU kernel engineering expert (CUDA / Triton) specializing in "
    "writing correctness-robust optimized kernels."
)

INITIAL_TEMPLATE = """\
{system_hint}

A `ModelNew` kernel passes KernelBench's standard correctness test (random
inputs from `get_inputs()`) but fails Phase II stress testing on N specific
adversarial inputs. Your task is to **regenerate a strengthened version of
this kernel** that fixes those failures *without* breaking the standard
KernelBench test.

────────────────────────────────────────────────────────────────────────────
## 1. Reference PyTorch implementation (ground truth — DO NOT modify)

```python
{reference_code}
```

────────────────────────────────────────────────────────────────────────────
## 2. Current optimized kernel (buggy under stress inputs)

```python
{buggy_kernel_code}
```

────────────────────────────────────────────────────────────────────────────
## 3. Failing inputs that broke this kernel (N = {n_failing})

Each failing case is described by:
- `policy`     — stress policy name (which value-distribution generator was used)
- `seed`       — random seed (so inputs are reproducible)
- `mode`       — eval or train (governs BN/Dropout/LayerNorm running stats)
- `ref_range`  — output range of the reference implementation [min, max]
- `orig_range` — output range of the current kernel (often diverging)
- `max_diff`   — max element-wise |ref - orig|
- `mean_diff`  — mean element-wise |ref - orig|
- `error`      — exception message (if the kernel crashed instead of diverging)

{failing_inputs_block}

────────────────────────────────────────────────────────────────────────────
## 4. Validation criteria you must satisfy

- **V_stress** (on the N failing inputs above):
      torch.allclose(ref_out, new_out, atol=1e-2, rtol=1e-2)
      AND new_out is NaN/Inf-free
      → must pass on EVERY one of the N inputs

- **V_kb** (KernelBench default `get_inputs()` across 5 random seeds):
      torch.allclose(ref_out, new_out, atol=1e-3, rtol=1e-3)
      → must pass on EVERY one of the 5 seeds
      (prevents regression on the standard test)

────────────────────────────────────────────────────────────────────────────
## 5. Your task

Regenerate a strengthened `ModelNew` so that:
  1. The class name `ModelNew` and its `__init__` / `forward` signatures
     remain identical to the buggy kernel above.
  2. All N failing inputs above produce outputs matching the reference under
     `atol=1e-2, rtol=1e-2` (V_stress).
  3. The 5 default KernelBench seeds still pass (V_kb).
  4. Performance characteristics are preserved where possible (do not collapse
     to a trivial pure-PyTorch implementation unless absolutely necessary).

Output ONLY a single fenced ```python``` code block containing the complete,
self-contained, executable kernel source (imports + load_inline + ModelNew).
Do NOT include any prose, explanation, or commentary outside the code block.
"""

ITERATE_TEMPLATE = """\
{system_hint}

A previous attempt to regenerate this kernel was made, but it did not pass
all validation criteria. Use the failure information from the previous round
to produce a better version.

────────────────────────────────────────────────────────────────────────────
## 1. Reference PyTorch implementation (ground truth — DO NOT modify)

```python
{reference_code}
```

────────────────────────────────────────────────────────────────────────────
## 2. Original buggy kernel (for context — DO NOT regenerate this verbatim)

```python
{buggy_kernel_code}
```

────────────────────────────────────────────────────────────────────────────
## 3. Your previous attempt (Round {prev_round})

```python
{prev_kernel_code}
```

────────────────────────────────────────────────────────────────────────────
## 4. Verification result of your previous attempt

### V_stress: {prev_stress_pass}/{prev_stress_total} failing inputs fixed
Still-failing cases ({n_still_failing}):
{still_failing_block}

### V_kb: {prev_kb_pass}/{prev_kb_total} default seeds pass
{kb_regression_block}

────────────────────────────────────────────────────────────────────────────
## 5. Full failing-input set for reference (V_stress target)

{failing_inputs_block}

────────────────────────────────────────────────────────────────────────────
## 6. Your task

Diagnose why your previous attempt failed on the still-failing cases (and on
any regressed KernelBench seeds), then produce a corrected `ModelNew`.

Requirements (unchanged from Round 1):
- Keep class name `ModelNew` and method signatures identical to the buggy kernel.
- V_stress: every failing input must allclose(atol=1e-2, rtol=1e-2) with the
  reference, NaN/Inf-free.
- V_kb: 5 default KernelBench seeds must allclose(atol=1e-3, rtol=1e-3).

Output ONLY a single fenced ```python``` code block containing the complete,
self-contained, executable kernel source. No prose outside the code block.
"""


def _format_failing_input(idx: int, fi: Dict[str, Any]) -> str:
    diff = (fi.get("diff_summary") or "").strip()
    err = (fi.get("error") or "").strip()
    lines = [f"[{idx}] policy={fi['policy']}, seed={fi['seed']}, mode={fi['mode']}:"]
    if diff:
        lines.append(f"      {diff}")
    if err:
        lines.append(f"      error: {err[:200]}")
    if not diff and not err:
        lines.append("      (no diff_summary captured; the kernel deviated from "
                     "the reference by more than atol/rtol)")
    return "\n".join(lines)


def _format_failing_inputs_block(failing_inputs: List[Dict[str, Any]],
                                 max_show: int = 200) -> str:
    if not failing_inputs:
        return "  (none)"
    shown = failing_inputs[:max_show]
    out = "\n".join(_format_failing_input(i + 1, fi)
                    for i, fi in enumerate(shown))
    if len(failing_inputs) > max_show:
        out += (f"\n  ... ({len(failing_inputs) - max_show} more cases "
                f"omitted; same policies/distributions as above)")
    return out


def _format_still_failing(still_failing: List[Dict[str, Any]],
                          max_show: int = 100) -> str:
    if not still_failing:
        return "  (none — V_stress fully passed)"
    shown = still_failing[:max_show]
    out = "\n".join(_format_failing_input(i + 1, fi)
                    for i, fi in enumerate(shown))
    if len(still_failing) > max_show:
        out += (f"\n  ... ({len(still_failing) - max_show} more cases omitted)")
    return out


def _format_kb_regression(v_kb: Dict[str, Any]) -> str:
    if v_kb.get("all_pass"):
        return "  (no regression — all 5 default seeds pass)"
    lines = []
    for ps in v_kb.get("per_seed", []):
        if not ps["passed"]:
            tag = ("ref_failed" if not ps["ref_ok"]
                   else "diverged" if ps["ref_ok"] and not ps["new_ok"]
                   else "?")
            extra = ps["diff_summary"] or ps["error"]
            lines.append(f"  seed={ps['seed']}: {tag}  {extra[:160]}")
    return "\n".join(lines) if lines else "  (no detail)"


def build_initial_prompt(reference_code: str, buggy_kernel_code: str,
                         failing_inputs: List[Dict[str, Any]]) -> str:
    return INITIAL_TEMPLATE.format(
        system_hint=SYSTEM_HINT,
        reference_code=reference_code.strip(),
        buggy_kernel_code=buggy_kernel_code.strip(),
        n_failing=len(failing_inputs),
        failing_inputs_block=_format_failing_inputs_block(failing_inputs),
    )


def build_iterate_prompt(reference_code: str, buggy_kernel_code: str,
                         prev_round: int, prev_kernel_code: str,
                         failing_inputs: List[Dict[str, Any]],
                         prev_v_stress: Dict[str, Any],
                         prev_v_kb: Dict[str, Any]) -> str:
    return ITERATE_TEMPLATE.format(
        system_hint=SYSTEM_HINT,
        reference_code=reference_code.strip(),
        buggy_kernel_code=buggy_kernel_code.strip(),
        prev_round=prev_round,
        prev_kernel_code=prev_kernel_code.strip(),
        prev_stress_pass=prev_v_stress.get("n_pass", 0),
        prev_stress_total=prev_v_stress.get("n_total", 0),
        n_still_failing=len(prev_v_stress.get("still_failing", [])),
        still_failing_block=_format_still_failing(
            prev_v_stress.get("still_failing", [])),
        prev_kb_pass=prev_v_kb.get("n_pass", 0),
        prev_kb_total=prev_v_kb.get("n_total", 0),
        kb_regression_block=_format_kb_regression(prev_v_kb),
        failing_inputs_block=_format_failing_inputs_block(failing_inputs),
    )


# ============================================================
# 3.  Failing-input dataset assembly
# ============================================================
def assemble_failing_inputs(kernel_name: str,
                            buggy_root: Dict[str, Any],
                            train_supp: Dict[str, Any]
                            ) -> List[Dict[str, Any]]:
    """Collect (policy, seed, mode) tuples for the kernel.

    eval events: take all events with mode='eval_value'.
    train events: take only those in train_supp where ref_ok=True and
                  original_ok=False (genuine buggy under train mode).
    """
    out: List[Dict[str, Any]] = []
    for ev in buggy_root.get("buggy_kernels", {}).get(kernel_name, []):
        mode = ev.get("mode", "")
        if mode == "eval_value":
            out.append({
                "policy": ev["policy"], "seed": ev["seed"],
                "mode": "eval_value",
                "diff_summary": "", "error": "",
                "source": "buggy_json_eval",
            })

    if train_supp:
        ks = (train_supp.get("supplemented", {}).get(kernel_name) or [])
        for ev in ks:
            if ev.get("ref_ok") and not ev.get("original_ok"):
                out.append({
                    "policy": ev["policy"], "seed": ev["seed"],
                    "mode": "train_value",
                    "diff_summary": ev.get("diff_summary", ""),
                    "error": ev.get("error", ""),
                    "source": "train_supp_verified",
                })
    return out


def collect_diff_summaries_round0(failing_inputs: List[Dict[str, Any]],
                                  buggy_kernel_code: str,
                                  problem_file: Path,
                                  log_prefix: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Round 0: confirm each failing input still breaks the buggy kernel and
    capture diff_summary. Returns (updated_failing_inputs, stats).
    """
    updated: List[Dict[str, Any]] = []
    n_confirmed_buggy = 0
    n_ref_failed = 0
    n_unexpected_pass = 0
    for i, fi in enumerate(failing_inputs):
        if fi.get("diff_summary"):
            updated.append(fi)
            n_confirmed_buggy += 1
            continue
        r = verify_one_input(buggy_kernel_code, problem_file,
                             fi["policy"], fi["seed"], fi["mode"],
                             ATOL_STRESS, RTOL_STRESS)
        if not r["ref_ok"]:
            n_ref_failed += 1
            print(f"{log_prefix}  [round0 {i+1}/{len(failing_inputs)}] "
                  f"({fi['policy']}, seed={fi['seed']}, {fi['mode']}): "
                  f"ref_failed  err={r['error'][:80]}", flush=True)
            updated.append({**fi, "diff_summary": "", "error": r["error"],
                            "ref_failed_at_round0": True})
        elif r["new_ok"]:
            n_unexpected_pass += 1
            print(f"{log_prefix}  [round0 {i+1}/{len(failing_inputs)}] "
                  f"({fi['policy']}, seed={fi['seed']}, {fi['mode']}): "
                  f"unexpectedly passes!", flush=True)
            updated.append({**fi, "diff_summary": "",
                            "unexpected_pass_at_round0": True})
        else:
            n_confirmed_buggy += 1
            updated.append({**fi, "diff_summary": r["diff_summary"],
                            "error": r["error"]})
            if (i + 1) % 10 == 0 or i + 1 == len(failing_inputs):
                print(f"{log_prefix}  [round0 {i+1}/{len(failing_inputs)}] "
                      f"({fi['policy']}, seed={fi['seed']}): confirmed buggy "
                      f"({r['diff_summary'][:80]})", flush=True)
    return updated, {
        "n_total": len(failing_inputs),
        "n_confirmed_buggy": n_confirmed_buggy,
        "n_ref_failed": n_ref_failed,
        "n_unexpected_pass": n_unexpected_pass,
    }


def filter_active_failing_inputs(failing_inputs: List[Dict[str, Any]]
                                 ) -> List[Dict[str, Any]]:
    """Drop inputs where ref crashed (we can't verify a fix without ref)."""
    return [fi for fi in failing_inputs
            if not fi.get("ref_failed_at_round0")]


# ============================================================
# 4.  Per-kernel main loop
# ============================================================
def run_one_kernel(*, kernel_name: str, kernel_meta: Dict[str, Any],
                   buggy_root: Dict[str, Any],
                   train_supp: Dict[str, Any],
                   call_llm, max_rounds: int,
                   out_dir: Path) -> Dict[str, Any]:
    log_p = f"[{kernel_name}]"
    problem_file = find_problem_file(kernel_meta["level"], kernel_meta["problem_id"])
    if problem_file is None:
        return {"kernel_name": kernel_name, "executed": False,
                "trigger": "no_problem_file"}

    kernel_path = Path(kernel_meta["kernel_path"])
    if not kernel_path.exists():
        return {"kernel_name": kernel_name, "executed": False,
                "trigger": "kernel_path_missing"}
    buggy_kernel_code = kernel_path.read_text(encoding="utf-8")
    reference_code = problem_file.read_text(encoding="utf-8")

    raw_failing = assemble_failing_inputs(kernel_name, buggy_root, train_supp)
    print(f"{log_p} Assembled {len(raw_failing)} raw failing input(s) "
          f"(eval + train_supp).", flush=True)

    # Round 0: capture diff_summary + sanity check
    print(f"{log_p} Round 0: capturing diff_summary on buggy kernel...",
          flush=True)
    failing_inputs, r0_stats = collect_diff_summaries_round0(
        raw_failing, buggy_kernel_code, problem_file, log_prefix=log_p)
    failing_inputs = filter_active_failing_inputs(failing_inputs)
    print(f"{log_p} Round 0 done: "
          f"{r0_stats['n_confirmed_buggy']} confirmed buggy, "
          f"{r0_stats['n_ref_failed']} ref_failed, "
          f"{r0_stats['n_unexpected_pass']} unexpected_pass. "
          f"Active V_stress set size = {len(failing_inputs)}.", flush=True)

    if not failing_inputs:
        return {
            "kernel_name": kernel_name, "executed": True,
            "trigger": "no_active_failing_inputs",
            "round0_stats": r0_stats,
            "rounds": [], "final_status": "skipped_no_failing_inputs",
        }

    prompt_dir = out_dir / "prompts"
    resp_dir = out_dir / "llm_responses"
    kern_dir = out_dir / "kernels"
    for d in (prompt_dir, resp_dir, kern_dir):
        d.mkdir(parents=True, exist_ok=True)

    rounds: List[Dict[str, Any]] = []
    iut = buggy_kernel_code
    final_status = "failed_after_3_rounds"
    final_round_idx = 0

    for round_num in range(1, max_rounds + 1):
        print(f"\n{log_p} === Round {round_num}/{max_rounds} ===", flush=True)

        if round_num == 1:
            prompt = build_initial_prompt(reference_code, buggy_kernel_code,
                                          failing_inputs)
        else:
            prev = rounds[-1]
            prompt = build_iterate_prompt(
                reference_code, buggy_kernel_code,
                prev_round=round_num - 1,
                prev_kernel_code=prev["new_kernel_code"],
                failing_inputs=failing_inputs,
                prev_v_stress=prev["v_stress"],
                prev_v_kb=prev["v_kb"],
            )

        try:
            (prompt_dir / f"{kernel_name}_r{round_num}.txt").write_text(
                prompt, encoding="utf-8")
        except Exception:
            pass

        print(f"{log_p}   Calling Opus 4.5 (prompt len = {len(prompt)} chars)...",
              flush=True)
        t_llm0 = time.time()
        try:
            llm_resp = call_llm(prompt)
        except Exception as e:
            print(f"{log_p}   LLM API error: {e}", flush=True)
            rounds.append({
                "round": round_num,
                "error": f"llm_call_failed: {str(e)[:300]}",
                "elapsed_sec": round(time.time() - t_llm0, 2),
            })
            break
        llm_elapsed = time.time() - t_llm0

        content = llm_resp.get("content", "")
        reasoning = llm_resp.get("reasoning_content", "")
        usage = llm_resp.get("usage", {})
        try:
            (resp_dir / f"{kernel_name}_r{round_num}_response.json").write_text(
                json.dumps({
                    "kernel_name": kernel_name, "round": round_num,
                    "model": llm_resp.get("model", ""),
                    "content": content, "reasoning_content": reasoning,
                    "usage": usage,
                    "latency_ms": llm_resp.get("latency_ms"),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        print(f"{log_p}   LLM done in {llm_elapsed:.1f}s; tokens={usage}; "
              f"thinking={len(reasoning)} chars", flush=True)

        new_code = extract_python_code(content)
        if not new_code or "class ModelNew" not in new_code:
            print(f"{log_p}   ! Code extraction failed (no ModelNew block).",
                  flush=True)
            rounds.append({
                "round": round_num,
                "model": llm_resp.get("model", ""),
                "usage": usage, "reasoning_chars": len(reasoning),
                "llm_elapsed_sec": round(llm_elapsed, 2),
                "new_kernel_code": new_code,
                "extraction_ok": False,
                "v_stress": None, "v_kb": None,
                "round_pass": False,
            })
            iut = iut
            continue

        try:
            (kern_dir / f"{kernel_name}_round{round_num}.py").write_text(
                new_code, encoding="utf-8")
        except Exception:
            pass

        print(f"{log_p}   Running V_stress on {len(failing_inputs)} inputs...",
              flush=True)
        t_v0 = time.time()
        v_stress = verify_stress_set(new_code, problem_file, failing_inputs)
        t_vs = time.time() - t_v0
        print(f"{log_p}     V_stress: pass={v_stress['n_pass']}/"
              f"{v_stress['n_total']}, fail={v_stress['n_fail']}, "
              f"skip_ref_failed={v_stress['n_ref_failed_skipped']}  "
              f"({t_vs:.1f}s)", flush=True)

        print(f"{log_p}   Running V_kb on 5 default seeds...", flush=True)
        t_kb0 = time.time()
        v_kb = verify_kb_default(new_code, problem_file)
        t_kb = time.time() - t_kb0
        print(f"{log_p}     V_kb: pass={v_kb['n_pass']}/{v_kb['n_total']}, "
              f"regressed_seeds={v_kb['regressed_seeds']}  ({t_kb:.1f}s)",
              flush=True)

        round_pass = bool(v_stress.get("all_pass") and v_kb.get("all_pass"))

        rounds.append({
            "round": round_num,
            "model": llm_resp.get("model", ""),
            "usage": usage,
            "reasoning_chars": len(reasoning),
            "llm_elapsed_sec": round(llm_elapsed, 2),
            "v_stress_elapsed_sec": round(t_vs, 2),
            "v_kb_elapsed_sec": round(t_kb, 2),
            "new_kernel_code": new_code,
            "extraction_ok": True,
            "v_stress": v_stress,
            "v_kb": v_kb,
            "round_pass": round_pass,
        })

        if round_pass:
            final_status = f"fixed_at_round_{round_num}"
            final_round_idx = round_num
            try:
                (kern_dir / f"{kernel_name}_final.py").write_text(
                    new_code, encoding="utf-8")
            except Exception:
                pass
            print(f"{log_p}   *** FIXED at round {round_num} ***", flush=True)
            break

        iut = new_code

    return {
        "kernel_name": kernel_name,
        "level": kernel_meta.get("level"),
        "problem_id": kernel_meta.get("problem_id"),
        "executed": True,
        "trigger": "task_b_regenerate_opus45",
        "max_rounds": max_rounds,
        "n_failing_inputs": len(failing_inputs),
        "round0_stats": r0_stats,
        "rounds": rounds,
        "final_status": final_status,
        "final_round": final_round_idx,
    }


# ============================================================
# 5.  Driver
# ============================================================
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-kernel", default="", help="restrict to one kernel")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "details").mkdir(parents=True, exist_ok=True)

    buggy = json.loads(BUGGY_JSON.read_text(encoding="utf-8"))
    best = json.loads(BEST_KERNELS_JSON.read_text(encoding="utf-8"))
    try:
        train_supp = json.loads(TRAIN_SUPP_JSON.read_text(encoding="utf-8"))
    except FileNotFoundError:
        train_supp = {"supplemented": {}}
        print(f"[WARN] No train supplement found at {TRAIN_SUPP_JSON} — "
              f"running with eval-only failing inputs.", flush=True)

    targets = [k for k in buggy.get("buggy_kernels", {}).keys()
               if k in best]
    if args.only_kernel:
        targets = [k for k in targets if k == args.only_kernel]
    targets = targets[args.start:]
    if args.limit > 0:
        targets = targets[:args.limit]

    completed_file = out_dir / "completed.json"
    completed: set = set()
    if args.resume and not args.no_resume and completed_file.exists():
        try:
            completed = set(json.loads(completed_file.read_text(encoding="utf-8")))
        except Exception:
            completed = set()
    print(f"[INIT] {len(targets)} target kernel(s); already completed: "
          f"{len(completed)}", flush=True)

    from src.stress.llm_clients import load_env_file, make_caller
    env = load_env_file()
    model_id = env.get("BEDROCK_MODEL_ID") or os.environ.get("BEDROCK_MODEL_ID")
    region = env.get("AWS_REGION") or os.environ.get("AWS_REGION", "us-west-2")
    print(f"[INIT] Bedrock model: {model_id} | region: {region}", flush=True)
    call_llm = make_caller("bedrock",
                           model_id=model_id, region=region,
                           enable_thinking=True, thinking_budget=8000,
                           max_tokens=16384)

    manifest = {
        "task": "task_b_regenerate",
        "git_commit": _git_commit(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_at": None,
        "model_id": model_id, "region": region,
        "max_rounds": args.rounds,
        "atol_stress": ATOL_STRESS, "rtol_stress": RTOL_STRESS,
        "atol_kb": ATOL_KB, "rtol_kb": RTOL_KB, "kb_seeds": KB_SEEDS,
        "kernels_total": len(targets),
        "kernels_completed": len(completed),
        "kernels_fixed": 0,
        "total_tokens": {"input": 0, "output": 0, "reasoning": 0},
        "hostname": socket.gethostname(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    in_tok = out_tok = reason_tok = 0
    fixed = 0

    for idx, kname in enumerate(targets, 1):
        if kname in completed and args.resume and not args.no_resume:
            print(f"[{idx}/{len(targets)}] {kname} -- skip (resume)", flush=True)
            continue
        meta = best[kname]
        print(f"\n{'='*72}\n[{idx}/{len(targets)}] {kname} "
              f"({meta['level']}/P{meta['problem_id']})\n{'='*72}", flush=True)
        t0 = time.time()

        try:
            res = run_one_kernel(
                kernel_name=kname, kernel_meta=meta,
                buggy_root=buggy, train_supp=train_supp,
                call_llm=call_llm, max_rounds=args.rounds, out_dir=out_dir,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            res = {"kernel_name": kname, "executed": False,
                   "trigger": "exception", "error": str(e)}

        res["elapsed_sec"] = round(time.time() - t0, 2)
        try:
            (out_dir / "details" / f"{kname}.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"  [WARN] save detail failed: {e}", flush=True)

        completed.add(kname)
        try:
            completed_file.write_text(
                json.dumps(sorted(list(completed))), encoding="utf-8")
        except Exception:
            pass

        for r in res.get("rounds", []):
            u = r.get("usage", {}) or {}
            in_tok += u.get("prompt_tokens", 0) or 0
            out_tok += u.get("completion_tokens", 0) or 0
            reason_tok += u.get("reasoning_tokens", 0) or 0
        if str(res.get("final_status", "")).startswith("fixed_at"):
            fixed += 1
        print(f"  -> status={res.get('final_status')} | "
              f"elapsed={res['elapsed_sec']}s | "
              f"running fixed={fixed}/{idx}", flush=True)

        manifest.update({
            "kernels_completed": len(completed),
            "kernels_fixed": fixed,
            "total_tokens": {"input": in_tok, "output": out_tok,
                             "reasoning": reason_tok},
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        (out_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[DONE] Task B finished.", flush=True)
    print(f"  Kernels attempted: {len(targets)}", flush=True)
    print(f"  Kernels fixed:     {fixed}", flush=True)
    print(f"  Tokens: in={in_tok} out={out_tok} reasoning={reason_tok}",
          flush=True)


if __name__ == "__main__":
    main()
