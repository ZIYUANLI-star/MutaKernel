#!/usr/bin/env python3
"""Task A: Re-run Phase II LLM iterative analysis with Claude Opus 4.5.

Targets: 365 mutants where Phase II ``any_killed=false``.
Evidence: Phase I EMD (4 layers) + Phase II 5-dimension stress results.
LLM: Claude Opus 4.5 via Bedrock, extended thinking enabled.
Max rounds: 5 (default).
Prompt: reuse ``build_analysis_prompt`` / ``build_reanalysis_prompt`` unchanged.

Outputs: ``第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/``
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
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BLOCK12_RESULT_DIR = PROJECT_ROOT / "第二次实验汇总" / "full_block12_results"
PHASE2_RESULT_DIR = PROJECT_ROOT / "第二次实验汇总" / "stress_enhance_results"
TASK_A_OUT_DIR = (
    PROJECT_ROOT / "第二次实验汇总" / "第二次实验汇总_补充" / "task_a_phase2_rerun"
)


# ---------------------------------------------------------------------------
# Lazy imports (torch + CUDA only on Linux runtime)
# ---------------------------------------------------------------------------
def _lazy_imports():
    """Import torch-dependent helpers from run_stress_enhance.

    Returns a dict of {name: function} for use in the main loop.
    Triggers torch import; only call on the GPU machine.
    """
    from scripts import run_stress_enhance as rse
    return {
        "run_stress_worker": rse._run_stress_worker,
        "verify_llm_suggestion": rse._verify_llm_suggestion,
        "llm_suggestion_violates_fixed_shape":
            rse._llm_suggestion_violates_fixed_shape,
        "STRESS_TIMEOUT": rse.STRESS_TIMEOUT,
        "ATOL": rse.ATOL, "RTOL": rse.RTOL, "DEVICE": rse.DEVICE,
        "find_problem_file": rse.find_problem_file,
        "PROBLEM_DIRS": rse.PROBLEM_DIRS,
        "BEST_KERNELS_FILE": rse.BEST_KERNELS_FILE,
    }


def _noop_helpers():
    """LLM-only stand-ins for dev machine without torch+CUDA.

    Skips verification: all rounds will record execution_result=null and
    the mutant will be marked killed=False unconditionally.
    """
    def fake_violates_fixed_shape(kill_strategy: str) -> bool:
        # Still apply the keyword filter so we don't waste a round.
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

    def fake_find_problem_file(*args, **kwargs):
        return None

    return {
        "run_stress_worker": None,
        "verify_llm_suggestion": fake_verify,
        "llm_suggestion_violates_fixed_shape": fake_violates_fixed_shape,
        "STRESS_TIMEOUT": 0,
        "ATOL": 1e-2, "RTOL": 1e-2, "DEVICE": "cpu",
        "find_problem_file": fake_find_problem_file,
        "PROBLEM_DIRS": {},
        "BEST_KERNELS_FILE": "",
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_phase2_unkilled() -> List[Dict[str, Any]]:
    """Read all Phase II detail JSONs, return those with any_killed=False.

    Each item is the raw Phase II detail dict (also contains tier, equiv_detail,
    main_track, config_track, llm_iterative_analysis).
    """
    details_dir = PHASE2_RESULT_DIR / "details"
    unkilled: List[Dict[str, Any]] = []
    for f in sorted(details_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("any_killed") is False:
            unkilled.append(d)
    return unkilled


def index_phase1_mutants() -> Dict[str, Dict[str, Any]]:
    """Build a {mutant_id: phase1_mutant_record} index across all Phase I files.

    Each record includes ``mutated_code``, ``original_code``, ``site``, etc.
    """
    details_dir = BLOCK12_RESULT_DIR / "details"
    idx: Dict[str, Dict[str, Any]] = {}
    for f in sorted(details_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        kernel_meta = d.get("kernel", {})
        for m in d.get("mutants", []):
            mid = m.get("id")
            if not mid:
                continue
            idx[mid] = {
                "phase1_record": m,
                "kernel_meta": kernel_meta,
            }
    return idx


# ---------------------------------------------------------------------------
# LLM iterative loop (Task A: reuse Phase II prompts)
# ---------------------------------------------------------------------------
def run_one_mutant_taskA(
    *,
    phase2_detail: Dict[str, Any],
    phase1_record: Dict[str, Any],
    kernel_meta: Dict[str, Any],
    call_llm,
    prompt_builders,
    rse_helpers: Dict[str, Any],
    max_rounds: int,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run up to ``max_rounds`` LLM rounds for one mutant, with verification."""
    from src.stress.llm_analyzer import parse_llm_response

    mutant_id = phase2_detail["mutant_id"]
    operator_name = phase2_detail["operator_name"]
    site = phase1_record.get("site", {})
    equiv_detail = phase2_detail.get("equiv_detail", {})
    # Fallback to Phase I equiv_detail if Phase II's is empty.
    if not equiv_detail.get("layer3"):
        ph1_ed = phase1_record.get("equiv_detail", {})
        if ph1_ed.get("layer3"):
            equiv_detail = {**equiv_detail, "layer3": ph1_ed["layer3"]}

    input_spec_str = equiv_detail.get("input_spec", "unknown")
    if not isinstance(input_spec_str, str):
        input_spec_str = json.dumps(input_spec_str, ensure_ascii=False)

    kernel_code = phase1_record.get("original_code", "")
    mutated_code = phase1_record.get("mutated_code", "")
    if not kernel_code or not mutated_code:
        return {
            "mutant_id": mutant_id,
            "executed": False,
            "trigger": "missing_code",
            "rounds": [],
            "killed": False,
        }

    enhanced_data = {
        "main_track": phase2_detail.get("main_track", {}),
        "config_track": phase2_detail.get("config_track", {}),
    }

    # Resolve problem_file for verification (Linux side).
    level_key = f"L{kernel_meta.get('level')}"
    problem_dir = rse_helpers["PROBLEM_DIRS"].get(level_key)
    problem_file = None
    if problem_dir is not None:
        problem_file = rse_helpers["find_problem_file"](
            problem_dir, kernel_meta.get("problem_id")
        )

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
                prompt = prompt_builders["build_analysis_prompt"](
                    original_code=kernel_code,
                    mutated_code=mutated_code,
                    operator_name=operator_name,
                    site=site,
                    input_spec=input_spec_str,
                    equiv_detail=equiv_detail,
                    enhanced_results=enhanced_data,
                )
            else:
                prompt = prompt_builders["build_reanalysis_prompt"](
                    original_code=kernel_code,
                    mutated_code=mutated_code,
                    operator_name=operator_name,
                    site=site,
                    input_spec=input_spec_str,
                    previous_rounds=rounds_history,
                    equiv_detail=equiv_detail,
                    enhanced_results=enhanced_data,
                )
        except Exception as e:
            print(f"      -> Prompt build error: {e}", flush=True)
            rounds_history.append({
                "round": round_num, "prompt_type": "error",
                "error": str(e), "killed": False,
            })
            break

        # Save prompt
        try:
            (prompt_dir / f"{mutant_id}_r{round_num}.txt").write_text(
                prompt, encoding="utf-8")
        except Exception:
            pass

        # Call LLM
        try:
            llm_resp = call_llm(prompt)
        except Exception as e:
            print(f"      -> LLM API error: {e}", flush=True)
            rounds_history.append({
                "round": round_num,
                "prompt_type": ("ANALYSIS_PROMPT_V2" if round_num == 1
                                else "REANALYSIS_PROMPT_V2"),
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

        # Persist raw response (without huge `raw` field)
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
                "prompt_type": ("ANALYSIS_PROMPT_V2" if round_num == 1
                                else "REANALYSIS_PROMPT_V2"),
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
            "prompt_type": ("ANALYSIS_PROMPT_V2" if round_num == 1
                            else "REANALYSIS_PROMPT_V2"),
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
        "kernel_name": phase2_detail.get("kernel_name"),
        "tier": phase2_detail.get("tier"),
        "executed": True,
        "trigger": "task_a_phase2_rerun_opus45",
        "max_rounds": max_rounds,
        "rounds": rounds_history,
        "killed": killed,
        "killing_round": killing_round,
        "model_id": rounds_history[0].get("model") if rounds_history else None,
        # Reference to DeepSeek's original result for comparison
        "deepseek_baseline": {
            "executed": phase2_detail.get("llm_iterative_analysis", {}).get(
                "executed", False),
            "killed": phase2_detail.get("llm_iterative_analysis", {}).get(
                "killed", False),
            "rounds_count": len(phase2_detail.get(
                "llm_iterative_analysis", {}).get("rounds", [])),
        },
    }

    if killed and killing_round > 0:
        kr = rounds_history[killing_round - 1]
        result["test_construction_rule"] = {
            "kill_strategy": kr.get("kill_strategy", ""),
            "suggested_code": kr.get("suggested_code", ""),
        }

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Task A: Opus 4.5 rerun on "
                                                 "Phase II unkilled mutants")
    parser.add_argument("--max-mutants", type=int, default=0,
                        help="Limit number of mutants (0=all)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Max LLM rounds per mutant (default 5)")
    parser.add_argument("--start", type=int, default=0,
                        help="Skip first N mutants (resume helper)")
    parser.add_argument("--mutant-id", type=str, default="",
                        help="Run only this single mutant_id (for smoke test)")
    parser.add_argument("--only-mutants", type=str, default="",
                        help="Path to a text file with one mutant_id per "
                             "line; rerun only those (ignores completed.json "
                             "for these ids)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing completed.json")
    parser.add_argument("--out-dir", type=str,
                        default=str(TASK_A_OUT_DIR),
                        help="Output directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually call LLM; print plan")
    parser.add_argument("--no-execute", action="store_true",
                        help="Skip torch+CUDA verification; LLM-only "
                             "(use on dev machine without GPU)")
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

    # Load env (.env file with Bedrock creds)
    from src.stress.llm_clients import load_env_file, make_caller
    env = load_env_file()
    model_id = env.get("BEDROCK_MODEL_ID") or os.environ.get("BEDROCK_MODEL_ID")
    region = env.get("AWS_REGION") or os.environ.get("AWS_REGION", "us-west-2")
    print(f"[INIT] Bedrock model: {model_id} | region: {region}", flush=True)

    # Load data
    print("[LOAD] Loading Phase II detail JSONs...", flush=True)
    phase2_unkilled = load_phase2_unkilled()
    print(f"[LOAD] Phase II unkilled: {len(phase2_unkilled)}", flush=True)

    print("[LOAD] Indexing Phase I mutants...", flush=True)
    phase1_idx = index_phase1_mutants()
    print(f"[LOAD] Phase I mutants indexed: {len(phase1_idx)}", flush=True)

    targets: List[Dict[str, Any]] = []
    missing_in_phase1 = 0
    for d in phase2_unkilled:
        mid = d["mutant_id"]
        if mid not in phase1_idx:
            missing_in_phase1 += 1
            continue
        targets.append({"phase2": d, "phase1": phase1_idx[mid]})

    print(f"[LOAD] Targets resolved: {len(targets)} "
          f"(missing in Phase I: {missing_in_phase1})", flush=True)

    if args.mutant_id:
        targets = [t for t in targets
                   if t["phase2"]["mutant_id"] == args.mutant_id]
        print(f"[FILTER] mutant_id filter: {args.mutant_id} -> "
              f"{len(targets)} target(s)", flush=True)

    only_ids: Optional[set] = None
    if args.only_mutants:
        with open(args.only_mutants, "r", encoding="utf-8") as f:
            only_ids = {ln.strip() for ln in f if ln.strip()}
        targets = [t for t in targets
                   if t["phase2"]["mutant_id"] in only_ids]
        # Force-remove these ids from `completed` so they get rerun.
        completed -= only_ids
        print(f"[FILTER] only-mutants={args.only_mutants} ({len(only_ids)} "
              f"ids) -> {len(targets)} target(s); "
              f"removed from completed.", flush=True)

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

    # Dry-run path
    if args.dry_run:
        print("[DRY-RUN] Would run:")
        for t in targets[:10]:
            print(f"  {t['phase2']['mutant_id']} (Tier {t['phase2']['tier']})")
        return

    # Bring up runtime helpers (torch + subprocess infra). On Linux only.
    if args.no_execute:
        print("[INIT] --no-execute: skipping torch import; LLM-only mode",
              flush=True)
        rse_helpers = _noop_helpers()
    else:
        print("[INIT] Loading torch + CUDA helpers...", flush=True)
        rse_helpers = _lazy_imports()

    # Build LLM caller
    print("[INIT] Building Bedrock LLM caller...", flush=True)
    call_llm = make_caller(
        "bedrock", model_id=model_id, region=region,
        enable_thinking=True, thinking_budget=8000,
        max_tokens=16384,
    )

    # Prompt builders
    from src.stress.llm_analyzer import (
        build_analysis_prompt, build_reanalysis_prompt,
    )
    prompt_builders = {
        "build_analysis_prompt": build_analysis_prompt,
        "build_reanalysis_prompt": build_reanalysis_prompt,
    }

    # Write run_manifest skeleton (will update on each save)
    manifest = {
        "task": "task_a_phase2_rerun",
        "git_commit": _git_commit(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_at": None,
        "model_id": model_id,
        "region": region,
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

    # Main loop
    total_in_tok = 0
    total_out_tok = 0
    total_reason_tok = 0
    killed_count = 0

    for idx, t in enumerate(targets):
        mid = t["phase2"]["mutant_id"]
        if mid in completed and not args.no_resume:
            print(f"  [{idx+1}/{len(targets)}] {mid} -- "
                  f"already done, skip", flush=True)
            continue

        print(f"\n{'_'*70}\n  [{idx+1}/{len(targets)}] {mid} "
              f"(Tier {t['phase2'].get('tier')})\n{'_'*70}", flush=True)
        t0 = time.time()

        try:
            result = run_one_mutant_taskA(
                phase2_detail=t["phase2"],
                phase1_record=t["phase1"]["phase1_record"],
                kernel_meta=t["phase1"]["kernel_meta"],
                call_llm=call_llm,
                prompt_builders=prompt_builders,
                rse_helpers=rse_helpers,
                max_rounds=args.rounds,
                out_dir=out_dir,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {
                "mutant_id": mid, "executed": False,
                "trigger": "exception",
                "error": str(e), "rounds": [], "killed": False,
            }

        result["elapsed_sec"] = round(time.time() - t0, 2)

        # Save detail
        try:
            (out_dir / "details" / f"{mid}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as e:
            print(f"  [WARN] save detail failed: {e}", flush=True)

        # Update completed.json
        completed.add(mid)
        try:
            completed_file.write_text(
                json.dumps(sorted(list(completed))), encoding="utf-8")
        except Exception:
            pass

        # Update token totals
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

        # Refresh manifest periodically
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

    print("\n[DONE] Task A finished.", flush=True)
    print(f"  Mutants attempted: {len(targets)}", flush=True)
    print(f"  Killed: {killed_count}", flush=True)
    print(f"  Total tokens: in={total_in_tok} | out={total_out_tok} | "
          f"reasoning≈{total_reason_tok}", flush=True)


if __name__ == "__main__":
    main()
