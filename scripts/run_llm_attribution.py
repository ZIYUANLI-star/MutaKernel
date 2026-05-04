#!/usr/bin/env python3
"""Layer 5: LLM-Assisted Mutation Attribution.

For ALL mutants that survive the 4-layer stress testing pipeline, this script
uses an LLM as the **primary attribution method** (replacing the old rule-based
attribution). For each survived mutant it:
  1. Constructs a structured prompt with code context + stress test summary
  2. Calls an LLM to classify into Type1-6 / Equivalent / Uncertain
  3. Verifies LLM-suggested inputs via _stress_worker.py (if --verify)

Usage:
  python scripts/run_llm_attribution.py \
      --model deepseek-chat \
      --api-base https://api.deepseek.com/v1 \
      --api-key $KEY \
      --verify
"""
import argparse
import gc
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stress.llm_analyzer import (
    LLMAnalysisResult,
    build_analysis_prompt,
    parse_llm_response,
    validate_suggested_code,
)

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
BLOCK12_RESULT_DIR = PROJECT_ROOT / "full_block12_results"
STRESS_RESULT_DIR = PROJECT_ROOT / "stress_enhance_results"
LLM_RESULT_DIR = PROJECT_ROOT / "llm_attribution_results"
WORKER_SCRIPT = SCRIPT_DIR / "_stress_worker.py"

PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

ATOL = 1e-2
RTOL = 1e-2
DEVICE = "cuda"
VERIFY_TIMEOUT = 120

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("llm_attribution")


def P(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# LLM caller (reuse OpenAI-compatible pattern from run_repair.py)
# ---------------------------------------------------------------------------

def create_llm_caller(model: str, api_base: str | None, api_key: str | None,
                      temperature: float, max_tokens: int):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=api_base)

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", "") or ""
        if content:
            return content
        if reasoning:
            return reasoning
        return ""
    return call


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_stress_results(result_dir: Path) -> list[dict]:
    """Load per-mutant stress results from the main experiment."""
    summary_path = result_dir / "stress_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text())
        return data.get("per_mutant", [])

    # Fallback: load individual detail files
    details_dir = result_dir / "details"
    if not details_dir.exists():
        return []
    results = []
    for jf in sorted(details_dir.glob("*.json")):
        try:
            results.append(json.loads(jf.read_text()))
        except Exception:
            continue
    return results


def load_stress30_results() -> list[dict]:
    """Load the pilot 30-mutant stress test results as fallback."""
    path = PROJECT_ROOT / "pilot_stress_results" / "stress30_results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _is_killed(r: dict) -> bool:
    """Check if a mutant was killed by any of the 4 stress layers."""
    if r.get("killed", False):
        return True
    for key in ("l1_killed", "l2_killed", "l3_killed", "l4_killed",
                "value_killed", "dtype_killed", "repeated_killed",
                "training_killed"):
        if r.get(key, False):
            return True
    return False


def filter_targets(results: list[dict], target: str) -> list[dict]:
    """Filter mutants by attribution target.

    Default (all_survived): every mutant NOT killed by Layer 1-4 → send to LLM.
    """
    targets = []
    for r in results:
        if target == "all_survived":
            if not _is_killed(r):
                targets.append(r)
        elif target == "unresolved":
            attr = r.get("attribution", "")
            if "unresolved" in attr.lower():
                targets.append(r)
        elif target == "type1":
            attr = r.get("attribution", "")
            if "type1" in attr.lower().replace(" ", ""):
                targets.append(r)
    return targets


def load_mutant_details(kernel_name: str, mutant_id: str) -> dict | None:
    """Load full mutant details from Block 1-2 results."""
    detail_path = BLOCK12_RESULT_DIR / "details" / f"{kernel_name}.json"
    if not detail_path.exists():
        return None
    data = json.loads(detail_path.read_text())
    for m in data.get("mutants", []):
        if m["id"] == mutant_id:
            return m
    return None


def find_problem_file(level_key: str, problem_id) -> Path | None:
    problem_dir = PROBLEM_DIRS.get(level_key)
    if not problem_dir:
        return None
    pid = str(problem_id)
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def get_input_spec(problem_file: Path) -> str:
    """Extract get_inputs() call to describe input shapes."""
    try:
        import torch
        import importlib.util
        spec = importlib.util.spec_from_file_location("_probe", str(problem_file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        torch.manual_seed(42)
        inputs = mod.get_inputs()
        parts = []
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                parts.append(f"  input[{i}]: shape={list(inp.shape)}, dtype={inp.dtype}")
            else:
                parts.append(f"  input[{i}]: {type(inp).__name__} = {inp}")
        return "\n".join(parts) if parts else "unknown"
    except Exception as e:
        return f"could not probe: {e}"


# ---------------------------------------------------------------------------
# Verification via subprocess
# ---------------------------------------------------------------------------

def run_verify_worker(problem_file: Path, kernel_code: str,
                      mutated_code: str, test_code: str,
                      timeout: int) -> dict | None:
    cfg = {
        "mode": "llm_verify",
        "problem_file": str(problem_file),
        "kernel_code": kernel_code,
        "mutated_code": mutated_code,
        "test_inputs_code": test_code,
        "atol": ATOL,
        "rtol": RTOL,
        "device": DEVICE,
    }
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="llmcfg_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="llmres_")
    os.close(cfg_fd)
    os.close(res_fd)

    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    try:
        proc = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT), cfg_path, res_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT), start_new_session=True,
        )
        try:
            _, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait()
            return None

        if os.path.exists(res_path) and os.path.getsize(res_path) > 2:
            with open(res_path) as f:
                return json.load(f)
        return None
    except Exception:
        return None
    finally:
        for p in [cfg_path, res_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Layer 5: LLM-Assisted Attribution")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--target", default="all_survived",
                        choices=["all_survived", "unresolved", "type1"])
    parser.add_argument("--max-mutants", type=int, default=100)
    parser.add_argument("--verify", action="store_true",
                        help="Execute LLM-suggested inputs to verify")
    parser.add_argument("--verify-timeout", type=int, default=VERIFY_TIMEOUT)
    parser.add_argument("--source", default="auto",
                        choices=["auto", "stress_enhance", "stress30"],
                        help="Which result set to read")
    args = parser.parse_args()

    t_start = time.time()
    LLM_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (LLM_RESULT_DIR / "details").mkdir(exist_ok=True)
    (LLM_RESULT_DIR / "prompts").mkdir(exist_ok=True)

    with open(BEST_KERNELS_FILE) as f:
        best_kernels = json.load(f)

    # --- Load stress results ---
    if args.source == "stress_enhance" or (args.source == "auto" and STRESS_RESULT_DIR.exists()):
        P("Loading results from stress_enhance_results/")
        all_results = load_stress_results(STRESS_RESULT_DIR)
    else:
        P("Loading results from pilot_stress_results/stress30_results.json")
        all_results = load_stress30_results()

    if not all_results:
        P("ERROR: No stress test results found. Run stress testing first.")
        return

    targets = filter_targets(all_results, args.target)
    if args.max_mutants > 0:
        targets = targets[:args.max_mutants]

    P(f"\n{'='*70}")
    P(f"  Layer 5: LLM-Assisted Mutation Attribution")
    P(f"{'='*70}")
    P(f"  Model: {args.model}")
    P(f"  Target filter: {args.target}")
    P(f"  Total stress results: {len(all_results)}")
    P(f"  Target mutants: {len(targets)}")
    P(f"  Verify suggestions: {args.verify}")
    P("")

    if not targets:
        P("  No target mutants found. Done.")
        return

    llm_caller = create_llm_caller(
        args.model, args.api_base, args.api_key,
        args.temperature, args.max_tokens,
    )

    # --- Process each target ---
    all_analysis: list[LLMAnalysisResult] = []
    verified_kills = 0
    llm_errors = 0

    for idx, stress_r in enumerate(targets):
        mutant_id = stress_r.get("mutant_id", stress_r.get("id", ""))
        kernel_name = stress_r.get("kernel_name", stress_r.get("kernel", ""))
        op_name = stress_r.get("operator_name", stress_r.get("operator", ""))

        P(f"\n{'_'*70}")
        P(f"  [{idx+1}/{len(targets)}] {mutant_id}")
        P(f"    kernel={kernel_name}  op={op_name}")

        # --- Gather code context ---
        mutant_detail = load_mutant_details(kernel_name, mutant_id)
        if not mutant_detail:
            P(f"    SKIP: mutant details not found in Block 1-2 results")
            continue

        mutated_code = mutant_detail.get("mutated_code", "")
        if not mutated_code:
            P(f"    SKIP: no mutated_code in details")
            continue

        site = mutant_detail.get("site", {})
        bk_info = best_kernels.get(kernel_name, {})
        kernel_path = Path(bk_info.get("kernel_path", ""))

        if kernel_path.exists():
            kernel_code = kernel_path.read_text()
        else:
            P(f"    WARNING: kernel file not found at {kernel_path}, using mutated_code as reference")
            kernel_code = mutated_code

        # Problem file for get_inputs
        level_key = f"L{bk_info.get('level', kernel_name.split('_')[0])}"
        pid = bk_info.get("problem_id", kernel_name.split("_P")[-1] if "_P" in kernel_name else "0")
        problem_file = find_problem_file(level_key, pid)

        input_spec = "unknown"
        if problem_file:
            input_spec = get_input_spec(problem_file)

        # Build stress summary from result
        stress_summary = {
            "l2_summary": "not killed" if not stress_r.get("dtype_killed", stress_r.get("l2_killed", False)) else "killed",
            "l4_summary": ("skipped" if not stress_r.get("training_killed", stress_r.get("l4_killed", False))
                           else "killed"),
            "original_failures": stress_r.get("original_failures", []),
        }

        # --- Build prompt ---
        prompt = build_analysis_prompt(
            kernel_code=kernel_code,
            mutated_code=mutated_code,
            operator_name=op_name,
            site=site,
            stress_summary=stress_summary,
            input_spec=input_spec,
        )

        prompt_path = LLM_RESULT_DIR / "prompts" / f"{mutant_id}_prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # --- Call LLM ---
        P(f"    Calling LLM ({args.model})...")
        try:
            raw_response = llm_caller(prompt)
        except Exception as e:
            P(f"    LLM ERROR: {e}")
            llm_errors += 1
            continue

        analysis = parse_llm_response(raw_response, mutant_id, op_name, kernel_name)
        P(f"    LLM verdict: equivalent={analysis.equivalent}, "
          f"category={analysis.category}, confidence={analysis.confidence:.2f}")
        P(f"    Reason: {analysis.reason}")

        # --- Verify suggested input ---
        if args.verify and analysis.suggested_test_code and problem_file:
            safety_err = validate_suggested_code(analysis.suggested_test_code)
            if safety_err:
                P(f"    Suggested code rejected: {safety_err}")
                analysis.verification_error = f"safety: {safety_err}"
            else:
                P(f"    Verifying LLM-suggested input...")
                vdata = run_verify_worker(
                    problem_file, kernel_code, mutated_code,
                    analysis.suggested_test_code, args.verify_timeout,
                )
                analysis.verification_ran = True
                if vdata is None:
                    P(f"    Verification: TIMEOUT/CRASH")
                    analysis.verification_error = "timeout or crash"
                elif vdata.get("error"):
                    P(f"    Verification error: {vdata['error']}")
                    analysis.verification_error = vdata["error"]
                elif not vdata.get("ref_ok", False):
                    P(f"    Verification: ref failed on LLM input (invalid input)")
                    analysis.verification_error = "ref_fail"
                elif vdata.get("original_ok") and not vdata.get("mutant_ok"):
                    P(f"    >>> KILLED by LLM-suggested input!")
                    analysis.verification_killed = True
                    verified_kills += 1
                elif not vdata.get("original_ok"):
                    P(f"    Verification: original also fails (potential Type 4)")
                    analysis.verification_error = "original_also_fails"
                else:
                    P(f"    Verification: both OK (LLM input did not kill)")

        all_analysis.append(analysis)

        # Save individual result
        detail_path = LLM_RESULT_DIR / "details" / f"{mutant_id}.json"
        with open(detail_path, "w") as f:
            json.dump(analysis.to_dict(), f, indent=2, ensure_ascii=False)

        gc.collect()

    # --- Summary ---
    elapsed = time.time() - t_start

    equiv_count = sum(1 for a in all_analysis if a.equivalent is True)
    non_equiv_count = sum(1 for a in all_analysis if a.equivalent is False)
    uncertain_count = sum(1 for a in all_analysis if a.equivalent is None)
    suggested_count = sum(1 for a in all_analysis if a.suggested_test_code)
    verified_count = sum(1 for a in all_analysis if a.verification_ran)

    cat_dist = defaultdict(int)
    for a in all_analysis:
        cat_dist[a.category] += 1

    summary = {
        "total_analyzed": len(all_analysis),
        "llm_errors": llm_errors,
        "equivalent_true": equiv_count,
        "equivalent_false": non_equiv_count,
        "equivalent_uncertain": uncertain_count,
        "category_distribution": dict(cat_dist),
        "suggested_inputs": suggested_count,
        "verified_inputs": verified_count,
        "verified_kills": verified_kills,
        "avg_confidence": (sum(a.confidence for a in all_analysis) / len(all_analysis)
                           if all_analysis else 0.0),
        "elapsed_min": round(elapsed / 60, 1),
        "model": args.model,
        "target_filter": args.target,
    }

    P(f"\n{'='*70}")
    P(f"  Layer 5: LLM-Assisted Attribution Complete")
    P(f"{'='*70}")
    P(f"  Analyzed:               {len(all_analysis)}")
    P(f"  LLM errors:             {llm_errors}")
    P(f"  Judged equivalent:      {equiv_count}")
    P(f"  Judged non-equivalent:  {non_equiv_count}")
    P(f"  Uncertain:              {uncertain_count}")
    P(f"  Category distribution:  {dict(cat_dist)}")
    P(f"  Suggested inputs:       {suggested_count}")
    P(f"  Verified:               {verified_count}")
    P(f"  Verified KILLS:         {verified_kills}")
    P(f"  Elapsed: {elapsed/60:.1f} min")

    summary_path = LLM_RESULT_DIR / "llm_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    P(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
