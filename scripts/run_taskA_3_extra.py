#!/usr/bin/env python3
"""Task A supplement: rerun Opus 4.5 audit on 3 mutants previously
killed only by the Phase II DeepSeek-R1 fallback.

Background
----------
The paper drops the Phase II LLM iterative-kill step from MutaKernel's
in-tool kill chain. Under the new methodology those 3 mutants are
treated as Phase II *survivors* (any_killed effectively False) and must
be audited under the same protocol as the other 365 Task A targets.

This script reuses ``run_taskA_phase2_rerun.run_one_mutant_taskA``
verbatim (same prompts, same Bedrock model, same 5-round budget,
same torch+CUDA verification) and writes outputs into the existing
``task_a_phase2_rerun/details/`` directory so that downstream stats
just need to re-scan that folder.

The Phase II detail JSONs of these 3 mutants do contain
``llm_iterative_analysis.killed=True``. ``build_analysis_prompt`` only
reads ``main_track`` / ``config_track`` / ``equiv_detail`` and never
touches ``llm_iterative_analysis`` or ``any_killed``, so the prompt is
identical in shape to the original 365 audits and does not leak the
DeepSeek-R1 fallback verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse Task A core verbatim.
from scripts.run_taskA_phase2_rerun import (  # noqa: E402
    PHASE2_RESULT_DIR,
    TASK_A_OUT_DIR,
    _git_commit,
    _lazy_imports,
    _noop_helpers,
    index_phase1_mutants,
    run_one_mutant_taskA,
)

# The 3 mutants that were killed only by Phase II's DeepSeek-R1 fallback.
EXTRA_MUTANT_IDS: List[str] = [
    "L1_P49__arith_replace__11",
    "L1_P49__init_modify__0",
    "L1_P23__init_modify__0",
]


def load_specific_phase2_details(ids: List[str]) -> List[Dict[str, Any]]:
    """Read the 3 specific Phase II detail JSONs regardless of any_killed."""
    details_dir = PHASE2_RESULT_DIR / "details"
    out: List[Dict[str, Any]] = []
    for mid in ids:
        f = details_dir / f"{mid}.json"
        if not f.exists():
            print(f"[WARN] missing Phase II detail: {f}", flush=True)
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] failed to parse {f}: {e}", flush=True)
            continue
        out.append(d)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Task A supplement: 3 extra mutants for Opus 4.5 audit"
    )
    parser.add_argument("--rounds", type=int, default=5,
                        help="Max LLM rounds per mutant (default 5)")
    parser.add_argument("--out-dir", type=str,
                        default=str(TASK_A_OUT_DIR),
                        help="Output directory (default merges into the "
                             "existing task_a_phase2_rerun/)")
    parser.add_argument("--no-execute", action="store_true",
                        help="Skip torch+CUDA verification; LLM-only "
                             "(use on dev machine without GPU)")
    parser.add_argument("--mutants", type=str, default="",
                        help="Comma-separated mutant_ids to override the "
                             "default 3-mutant list")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without calling LLM")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "details").mkdir(parents=True, exist_ok=True)

    target_ids = (
        [s.strip() for s in args.mutants.split(",") if s.strip()]
        if args.mutants else EXTRA_MUTANT_IDS
    )
    print(f"[INIT] Targets: {target_ids}", flush=True)

    phase2_details = load_specific_phase2_details(target_ids)
    print(f"[LOAD] Phase II details loaded: {len(phase2_details)}", flush=True)

    print("[LOAD] Indexing Phase I mutants...", flush=True)
    phase1_idx = index_phase1_mutants()
    print(f"[LOAD] Phase I mutants indexed: {len(phase1_idx)}", flush=True)

    targets: List[Dict[str, Any]] = []
    for d in phase2_details:
        mid = d["mutant_id"]
        if mid not in phase1_idx:
            print(f"[WARN] {mid} missing in Phase I index, skip", flush=True)
            continue
        targets.append({"phase2": d, "phase1": phase1_idx[mid]})

    if not targets:
        print("[DONE] No targets after resolution.", flush=True)
        return

    if args.dry_run:
        print("[DRY-RUN] Would run:")
        for t in targets:
            p2 = t["phase2"]
            print(f"  {p2['mutant_id']} (Tier {p2.get('tier')}, "
                  f"operator={p2.get('operator_name')})")
        return

    from src.stress.llm_clients import load_env_file, make_caller
    env = load_env_file()
    model_id = env.get("BEDROCK_MODEL_ID") or os.environ.get("BEDROCK_MODEL_ID")
    region = env.get("AWS_REGION") or os.environ.get("AWS_REGION", "us-west-2")
    print(f"[INIT] Bedrock model: {model_id} | region: {region}", flush=True)

    if args.no_execute:
        print("[INIT] --no-execute: LLM-only mode (no GPU verification)",
              flush=True)
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

    from src.stress.llm_analyzer import (
        build_analysis_prompt, build_reanalysis_prompt,
    )
    prompt_builders = {
        "build_analysis_prompt": build_analysis_prompt,
        "build_reanalysis_prompt": build_reanalysis_prompt,
    }

    extra_manifest = {
        "task": "task_a_phase2_rerun_extra3",
        "git_commit": _git_commit(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "finished_at": None,
        "model_id": model_id,
        "region": region,
        "max_rounds": args.rounds,
        "extended_thinking": {"enabled": True, "budget_tokens": 8000},
        "max_tokens": 16384,
        "input_count": len(targets),
        "completed_count": 0,
        "killed_count": 0,
        "total_tokens": {"input": 0, "output": 0, "reasoning": 0},
        "hostname": socket.gethostname(),
        "target_ids": [t["phase2"]["mutant_id"] for t in targets],
        "note": ("Supplement audit: these mutants were previously killed "
                 "only by the Phase II DeepSeek-R1 fallback, which has been "
                 "removed from the in-tool kill chain. Audited here under "
                 "the same Task A protocol as the other 365 mutants."),
    }
    extra_manifest_path = out_dir / "run_manifest_extra3.json"
    extra_manifest_path.write_text(
        json.dumps(extra_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8")

    total_in = total_out = total_reason = 0
    killed_count = 0

    for idx, t in enumerate(targets):
        mid = t["phase2"]["mutant_id"]
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
        result["supplement_run"] = True

        try:
            (out_dir / "details" / f"{mid}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8")
        except Exception as e:
            print(f"  [WARN] save detail failed: {e}", flush=True)

        for r in result.get("rounds", []):
            u = r.get("usage", {}) or {}
            total_in += u.get("prompt_tokens", 0) or 0
            total_out += u.get("completion_tokens", 0) or 0
            total_reason += u.get("reasoning_tokens", 0) or 0
        if result.get("killed"):
            killed_count += 1

        print(f"  -> killed={result.get('killed')} | "
              f"elapsed={result['elapsed_sec']}s | "
              f"running kill rate={killed_count}/{idx+1}", flush=True)

        extra_manifest.update({
            "completed_count": idx + 1,
            "killed_count": killed_count,
            "total_tokens": {
                "input": total_in,
                "output": total_out,
                "reasoning": total_reason,
            },
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        extra_manifest_path.write_text(
            json.dumps(extra_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8")

    print("\n[DONE] Task A supplement finished.", flush=True)
    print(f"  Mutants attempted: {len(targets)}", flush=True)
    print(f"  Killed by Opus 4.5: {killed_count}", flush=True)
    print(f"  Total tokens: in={total_in} | out={total_out} | "
          f"reasoning≈{total_reason}", flush=True)
    print(f"  Manifest: {extra_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
