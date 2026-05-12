#!/usr/bin/env python3
"""Task C: Opus 4.5 directly challenges Phase I survivors.

Supervisor's idea: skip Phase II's 5-dimension deterministic enhancement
entirely. Feed only Phase I EMD evidence (Layer 0-3) to Claude Opus 4.5
and let it generate value-level test inputs.

Targets: 534 mutants where Phase I status in {survived, candidate_equivalent}.
Compared to Task A (which uses Phase II results), this is the **ablation
arm**: how much does Phase II's deterministic enhancement actually help?

Outputs: ``第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/``
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

BLOCK12_RESULT_DIR = PROJECT_ROOT / "第二次实验汇总" / "full_block12_results"
TASK_C_OUT_DIR = (
    PROJECT_ROOT / "第二次实验汇总" / "第二次实验汇总_补充" / "task_c_phase1_direct"
)

ENHANCEABLE_STATUSES = {"survived", "candidate_equivalent"}


def _lazy_imports():
    """Import torch-dependent helpers from run_stress_enhance (Linux GPU)."""
    from scripts import run_stress_enhance as rse
    return {
        "verify_llm_suggestion": rse._verify_llm_suggestion,
        "llm_suggestion_violates_fixed_shape":
            rse._llm_suggestion_violates_fixed_shape,
        "find_problem_file": rse.find_problem_file,
        "PROBLEM_DIRS": rse.PROBLEM_DIRS,
    }


def _noop_helpers():
    """LLM-only stand-ins (Windows dev)."""
    def fake_violates_fixed_shape(kill_strategy: str) -> bool:
        if not kill_strategy:
            return False
        lower = kill_strategy.lower()
        kws = ["change shape", "different shape", "different dimension",
               "vary m/n/k", "vary m,n,k", "non-divisible size",
               "non-divisible", "change the input size",
               "change input dimensions", "different batch", "vary batch",
               "change batch_size", "modify the shape", "alter the dimensions"]
        return any(k in lower for k in kws)

    def fake_verify(*args, **kwargs):
        return {"killed": False, "ref_ok": None, "original_ok": None,
                "mutant_ok": None, "diff_summary": "",
                "error": "no_execute_mode"}

    return {
        "verify_llm_suggestion": fake_verify,
        "llm_suggestion_violates_fixed_shape": fake_violates_fixed_shape,
        "find_problem_file": lambda *a, **k: None,
        "PROBLEM_DIRS": {},
    }


def load_phase1_survivors() -> List[Dict[str, Any]]:
    """Return all Phase I mutants with status in ENHANCEABLE_STATUSES.

    Each item is ``{phase1_record, kernel_meta, kernel_name}``.
    """
    details_dir = BLOCK12_RESULT_DIR / "details"
    survivors: List[Dict[str, Any]] = []
    for f in sorted(details_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        kernel_meta = d.get("kernel", {})
        kernel_name = kernel_meta.get("problem_name", f.stem)
        for m in d.get("mutants", []):
            if m.get("status", "") in ENHANCEABLE_STATUSES:
                survivors.append({
                    "phase1_record": m,
                    "kernel_meta": kernel_meta,
                    "kernel_name": kernel_name,
                })
    return survivors


def run_one_mutant_taskC(
    *,
    phase1_record: Dict[str, Any],
    kernel_meta: Dict[str, Any],
    kernel_name: str,
    call_llm,
    rse_helpers: Dict[str, Any],
    max_rounds: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run up to ``max_rounds`` LLM rounds for one mutant (Phase I evidence only)."""
    from src.stress.llm_analyzer import (
        parse_llm_response,
        build_phase1_direct_prompt,
        build_phase1_direct_reanalysis_prompt,
    )

    mutant_id = phase1_record["id"]
    operator_name = phase1_record["operator_name"]
    site = phase1_record.get("site", {})
    equiv_detail = phase1_record.get("equiv_detail", {})
    input_spec_str = equiv_detail.get("input_spec", "unknown")
    if not isinstance(input_spec_str, str):
        input_spec_str = json.dumps(input_spec_str, ensure_ascii=False)

    kernel_code = phase1_record.get("original_code", "")
    mutated_code = phase1_record.get("mutated_code", "")
    if not kernel_code or not mutated_code:
        return {
            "mutant_id": mutant_id, "executed": False,
            "trigger": "missing_code", "rounds": [], "killed": False,
        }

    level_key = f"L{kernel_meta.get('level')}"
    problem_dir = rse_helpers["PROBLEM_DIRS"].get(level_key)
    problem_file = None
    if problem_dir is not None:
        problem_file = rse_helpers["find_problem_file"](
            problem_dir, kernel_meta.get("problem_id"))

    rounds_history: List[Dict[str, Any]] = []
    killed = False
    killing_round = 0

    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    resp_dir = out_dir / "llm_responses"
    resp_dir.mkdir(parents=True, exist_ok=True)

    for round_num in range(1, max_rounds + 1):
        print(f"    [Round {round_num}/{max_rounds}] {mutant_id}", flush=True)

        try:
            if round_num == 1:
                prompt = build_phase1_direct_prompt(
                    original_code=kernel_code,
                    mutated_code=mutated_code,
                    operator_name=operator_name,
                    site=site,
                    input_spec=input_spec_str,
                    equiv_detail=equiv_detail,
                )
            else:
                prompt = build_phase1_direct_reanalysis_prompt(
                    original_code=kernel_code,
                    mutated_code=mutated_code,
                    operator_name=operator_name,
                    site=site,
                    input_spec=input_spec_str,
                    previous_rounds=rounds_history,
                    equiv_detail=equiv_detail,
                )
        except Exception as e:
            print(f"      -> Prompt build error: {e}", flush=True)
            rounds_history.append({
                "round": round_num, "prompt_type": "error",
                "error": str(e), "killed": False,
            })
            break

        try:
            (prompt_dir / f"{mutant_id}_r{round_num}.txt").write_text(
                prompt, encoding="utf-8")
        except Exception:
            pass

        try:
            llm_resp = call_llm(prompt)
        except Exception as e:
            print(f"      -> LLM API error: {e}", flush=True)
            rounds_history.append({
                "round": round_num,
                "prompt_type": ("ANALYSIS_PROMPT_C" if round_num == 1
                                else "REANALYSIS_PROMPT_C"),
                "error": str(e), "killed": False,
            })
            break

        raw_response = llm_resp.get("content", "")
        reasoning_content = llm_resp.get("reasoning_content", "")
        llm_usage = llm_resp.get("usage", {})
        llm_model_used = llm_resp.get("model", "")

        if reasoning_content:
            print(f"      -> thinking: {len(reasoning_content)} chars",
                  flush=True)
        print(f"      -> tokens: {llm_usage}", flush=True)

        parsed = parse_llm_response(raw_response)

        resp_record = {
            "mutant_id": mutant_id,
            "round": round_num,
            "model": llm_model_used,
            "killable": parsed.get("killable") if parsed else None,
            "reason_category": parsed.get("reason_category") if parsed else None,
            "proof_sketch": parsed.get("proof_sketch") if parsed else None,
            "content": raw_response,
            "reasoning_content": reasoning_content,
            "usage": llm_usage,
            "latency_ms": llm_resp.get("latency_ms"),
        }
        try:
            (resp_dir / f"{mutant_id}_r{round_num}_response.json"
             ).write_text(json.dumps(resp_record, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        except Exception:
            pass

        if parsed is None:
            rounds_history.append({
                "round": round_num,
                "prompt_type": ("ANALYSIS_PROMPT_C" if round_num == 1
                                else "REANALYSIS_PROMPT_C"),
                "survival_reason": "parse_error",
                "killable": False, "kill_strategy": "",
                "suggested_code": "", "execution_result": None,
                "killed": False,
                "raw_response": raw_response[:500],
                "reasoning_content":
                    reasoning_content[:1000] if reasoning_content else "",
                "model": llm_model_used,
                "usage": llm_usage,
            })
            continue

        survival_reason = parsed.get("survival_reason", "")
        is_killable = parsed.get("killable", False)
        kill_strategy = parsed.get("kill_strategy", "")
        suggested = parsed.get("suggested_test")
        recommendations = parsed.get("recommendations", "")
        reason_category = parsed.get("reason_category", "unknown")
        proof_sketch = parsed.get("proof_sketch", "")

        round_record = {
            "round": round_num,
            "prompt_type": ("ANALYSIS_PROMPT_C" if round_num == 1
                            else "REANALYSIS_PROMPT_C"),
            "model": llm_model_used,
            "usage": llm_usage,
            "reason_category": reason_category,
            "proof_sketch": proof_sketch,
            "survival_reason": survival_reason,
            "killable": is_killable,
            "kill_strategy": kill_strategy,
            "recommendations": recommendations,
            "suggested_code": "",
            "execution_result": None,
            "killed": False,
        }

        if not is_killable or not suggested or not isinstance(suggested, dict):
            print(f"      -> LLM says unkillable", flush=True)
            rounds_history.append(round_record)
            break

        python_code = (suggested.get("python_code") or "").strip()
        if not python_code:
            print(f"      -> LLM gave no code", flush=True)
            rounds_history.append(round_record)
            continue

        if rse_helpers["llm_suggestion_violates_fixed_shape"](kill_strategy):
            print(f"      -> Violates fixed-shape, skip", flush=True)
            round_record["execution_result"] = {
                "killed": False, "error": "violates_fixed_shape"}
            rounds_history.append(round_record)
            continue

        round_record["suggested_code"] = python_code
        desc = suggested.get("description", "N/A")
        print(f"      -> Executing: {str(desc)[:80]}", flush=True)

        if problem_file is None:
            print(f"      -> No problem_file resolved, skip exec", flush=True)
            round_record["execution_result"] = {
                "killed": False, "error": "no_problem_file"}
            rounds_history.append(round_record)
            continue

        exec_result = rse_helpers["verify_llm_suggestion"](
            problem_file, kernel_code, mutated_code, python_code)
        round_record["execution_result"] = exec_result

        if exec_result.get("killed"):
            print(f"      *** KILLED in round {round_num}! ***", flush=True)
            round_record["killed"] = True
            killed = True
            killing_round = round_num
            rounds_history.append(round_record)
            break
        else:
            err = exec_result.get("error", "")
            if err:
                print(f"      -> not killed (error: {err[:80]})", flush=True)
            else:
                ds = exec_result.get("diff_summary") or ""
                print(f"      -> not killed (within atol/rtol)  {ds[:120]}",
                      flush=True)
            rounds_history.append(round_record)

    result: Dict[str, Any] = {
        "mutant_id": mutant_id,
        "operator_name": operator_name,
        "kernel_name": kernel_name,
        "phase1_status": phase1_record.get("status"),
        "executed": True,
        "trigger": "task_c_phase1_direct_opus45",
        "max_rounds": max_rounds,
        "rounds": rounds_history,
        "killed": killed,
        "killing_round": killing_round,
        "model_id": rounds_history[0].get("model") if rounds_history else None,
    }

    if killed and killing_round > 0:
        kr = rounds_history[killing_round - 1]
        result["test_construction_rule"] = {
            "kill_strategy": kr.get("kill_strategy", ""),
            "suggested_code": kr.get("suggested_code", ""),
        }

    return result


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Task C: Opus 4.5 directly "
                                                 "challenges Phase I survivors")
    parser.add_argument("--max-mutants", type=int, default=0,
                        help="Limit number of mutants (0=all)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Max LLM rounds per mutant (default 5)")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip first N mutants")
    parser.add_argument("--mutant-id", type=str, default="",
                        help="Run only this single mutant_id")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing completed.json")
    parser.add_argument("--out-dir", type=str,
                        default=str(TASK_C_OUT_DIR),
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't call LLM, print plan")
    parser.add_argument("--no-execute", action="store_true",
                        help="Skip torch+CUDA verification (LLM only)")
    parser.add_argument("--filter-status", type=str, default="",
                        choices=["", "survived", "candidate_equivalent"],
                        help="Restrict to one Phase I status")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "details").mkdir(exist_ok=True)

    completed_file = out_dir / "completed.json"
    completed: set = set()
    if completed_file.exists() and not args.no_resume:
        try:
            completed = set(json.loads(completed_file.read_text(encoding="utf-8")))
        except Exception:
            completed = set()
    print(f"[INIT] Resume mode: {not args.no_resume}; "
          f"already completed: {len(completed)}", flush=True)

    from src.stress.llm_clients import load_env_file, make_caller
    env = load_env_file()
    model_id = env.get("BEDROCK_MODEL_ID") or os.environ.get("BEDROCK_MODEL_ID")
    region = env.get("AWS_REGION") or os.environ.get("AWS_REGION", "us-west-2")
    print(f"[INIT] Bedrock model: {model_id} | region: {region}", flush=True)

    print("[LOAD] Loading Phase I survivors...", flush=True)
    targets = load_phase1_survivors()
    print(f"[LOAD] Phase I survivors total: {len(targets)}", flush=True)
    by_status = {}
    for t in targets:
        s = t["phase1_record"].get("status", "?")
        by_status[s] = by_status.get(s, 0) + 1
    print(f"[LOAD] By status: {by_status}", flush=True)

    if args.filter_status:
        targets = [t for t in targets
                   if t["phase1_record"].get("status") == args.filter_status]
        print(f"[FILTER] status={args.filter_status} -> "
              f"{len(targets)} target(s)", flush=True)

    if args.mutant_id:
        targets = [t for t in targets
                   if t["phase1_record"]["id"] == args.mutant_id]
        print(f"[FILTER] mutant_id={args.mutant_id} -> "
              f"{len(targets)} target(s)", flush=True)

    if args.start > 0:
        targets = targets[args.start:]
        print(f"[FILTER] start={args.start} -> {len(targets)} remaining",
              flush=True)

    if args.max_mutants > 0:
        targets = targets[:args.max_mutants]
        print(f"[FILTER] max-mutants={args.max_mutants} -> "
              f"{len(targets)} to run", flush=True)

    if not targets:
        print("[DONE] No targets, exiting.", flush=True)
        return

    if args.dry_run:
        print("[DRY-RUN] Would run:")
        for t in targets[:10]:
            print(f"  {t['phase1_record']['id']} "
                  f"({t['phase1_record'].get('status')})")
        return

    if args.no_execute:
        print("[INIT] --no-execute: LLM-only mode", flush=True)
        rse_helpers = _noop_helpers()
    else:
        print("[INIT] Loading torch + CUDA helpers...", flush=True)
        rse_helpers = _lazy_imports()

    print("[INIT] Building Bedrock LLM caller...", flush=True)
    call_llm = make_caller(
        "bedrock", model_id=model_id, region=region,
        enable_thinking=True, thinking_budget=8000,
        max_tokens=16384,
    )

    manifest = {
        "task": "task_c_phase1_direct",
        "git_commit": _git_commit(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_at": None,
        "model_id": model_id, "region": region,
        "max_rounds": args.rounds,
        "extended_thinking": {"enabled": True, "budget_tokens": 8000},
        "max_tokens": 16384,
        "input_count": len(targets) + len(completed),
        "completed_count": len(completed),
        "killed_count": 0,
        "total_tokens": {"input": 0, "output": 0, "reasoning": 0},
        "hostname": socket.gethostname(),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total_in_tok = 0
    total_out_tok = 0
    total_reason_tok = 0
    killed_count = 0

    for idx, t in enumerate(targets):
        mid = t["phase1_record"]["id"]
        if mid in completed and not args.no_resume:
            print(f"  [{idx+1}/{len(targets)}] {mid} -- skip", flush=True)
            continue

        print(f"\n{'_'*70}\n  [{idx+1}/{len(targets)}] {mid} "
              f"(Phase1={t['phase1_record'].get('status')})\n{'_'*70}",
              flush=True)
        t0 = time.time()

        try:
            result = run_one_mutant_taskC(
                phase1_record=t["phase1_record"],
                kernel_meta=t["kernel_meta"],
                kernel_name=t["kernel_name"],
                call_llm=call_llm,
                rse_helpers=rse_helpers,
                max_rounds=args.rounds,
                out_dir=out_dir,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {
                "mutant_id": mid, "executed": False,
                "trigger": "exception", "error": str(e),
                "rounds": [], "killed": False,
            }

        result["elapsed_sec"] = round(time.time() - t0, 2)

        try:
            (out_dir / "details" / f"{mid}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as e:
            print(f"  [WARN] save detail failed: {e}", flush=True)

        completed.add(mid)
        try:
            completed_file.write_text(
                json.dumps(sorted(list(completed))), encoding="utf-8")
        except Exception:
            pass

        for r in result.get("rounds", []):
            u = r.get("usage", {}) or {}
            total_in_tok += u.get("prompt_tokens", 0) or 0
            total_out_tok += u.get("completion_tokens", 0) or 0
            total_reason_tok += u.get("reasoning_tokens", 0) or 0
        if result.get("killed"):
            killed_count += 1

        print(f"  -> killed={result.get('killed')} | "
              f"elapsed={result['elapsed_sec']}s | "
              f"running kill rate={killed_count}/{idx+1}", flush=True)

        manifest.update({
            "completed_count": len(completed),
            "killed_count": killed_count,
            "total_tokens": {
                "input": total_in_tok,
                "output": total_out_tok,
                "reasoning": total_reason_tok,
            },
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        (out_dir / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8")

    print("\n[DONE] Task C finished.", flush=True)
    print(f"  Mutants attempted: {len(targets)}", flush=True)
    print(f"  Killed: {killed_count}", flush=True)
    print(f"  Total tokens: in={total_in_tok} | out={total_out_tok} | "
          f"reasoning≈{total_reason_tok}", flush=True)


if __name__ == "__main__":
    main()
