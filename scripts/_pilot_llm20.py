"""Phase 2+3: LLM iterative analysis + verification + clustering.

This script:
  1. Collects mutants that survived Phase 1 (4-layer stress testing)
  2. For each: LLM free-form analysis → verify suggested inputs → iterate (up to 3 rounds)
  3. For killed mutants: extract test construction rules
  4. For unkilled mutants: generate robustness suggestions + fixed code
  5. After all: cluster survival reasons into a taxonomy

Run on the same machine with GPU (needs: openai, torch).
"""
import json
import os
import sys
import time
import glob
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stress.llm_analyzer import (
    LLMAnalysisResult,
    build_analysis_prompt,
    build_reanalysis_prompt,
    build_robustness_prompt,
    build_test_rule_prompt,
    build_cluster_prompt,
    parse_analysis_response,
    parse_llm_response,
    validate_suggested_code,
)
from scripts._verify_llm_suggestions import verify_one_mutant, resolve_problem_file

BLOCK12_DIR = PROJECT_ROOT / "full_block12_results" / "details"
STRESS_RESULT_DIR = PROJECT_ROOT / "stress_enhance_results"
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
OUTPUT_DIR = PROJECT_ROOT / "llm_analysis_results"

MAX_MUTANTS = int(os.environ.get("LLM_MAX_MUTANTS", "0"))
MAX_ROUNDS = int(os.environ.get("LLM_MAX_ROUNDS", "3"))
MODEL = os.environ.get("LLM_MODEL", "deepseek-reasoner")
API_BASE = os.environ.get("LLM_API_BASE", "https://api.deepseek.com/v1")
API_KEY = os.environ.get("LLM_API_KEY", os.environ.get("DEEPSEEK_API_KEY", ""))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "16384"))


def P(msg):
    print(msg, flush=True)


def create_caller():
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=API_BASE)

    def call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
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
def load_mutant_detail(kernel_name: str, mutant_id: str) -> Optional[Dict]:
    detail_path = BLOCK12_DIR / f"{kernel_name}.json"
    if not detail_path.exists():
        return None
    data = json.loads(detail_path.read_text(encoding="utf-8"))
    for m in data.get("mutants", []):
        if m["id"] == mutant_id:
            return m
    return None


def get_input_spec_from_problem(problem_file: Optional[str]) -> str:
    """Run get_inputs() from the problem file to extract real tensor specs."""
    if not problem_file:
        return "unknown (no problem file)"
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_probe_problem", problem_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        get_inputs_fn = getattr(mod, "get_inputs", None)
        if not get_inputs_fn:
            return "unknown (no get_inputs function)"
        import torch
        inputs = get_inputs_fn()
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        parts = []
        for i, t in enumerate(inputs):
            if isinstance(t, torch.Tensor):
                parts.append(f"  arg[{i}]: shape={list(t.shape)}, dtype={t.dtype}")
            else:
                parts.append(f"  arg[{i}]: type={type(t).__name__}, value={t}")
        header = f"forward() takes {len(inputs)} tensor argument(s):"
        return header + "\n" + "\n".join(parts)
    except Exception as e:
        return f"unknown (probe error: {str(e)[:100]})"


def get_input_spec(detail: Dict, problem_file: Optional[str] = None) -> str:
    """Get input specification, preferring dynamic extraction from problem file."""
    dynamic = get_input_spec_from_problem(problem_file)
    if not dynamic.startswith("unknown"):
        return dynamic
    meta = detail.get("test_meta", {})
    inputs_info = meta.get("inputs_info", "")
    if inputs_info:
        return inputs_info
    shapes = detail.get("input_shapes", "")
    if shapes:
        return str(shapes)
    return dynamic


def collect_survived_mutants() -> List[Dict]:
    """Collect mutants that survived ALL 4 layers of stress testing."""
    stress_details = STRESS_RESULT_DIR / "details"

    if stress_details.exists() and any(stress_details.glob("*.json")):
        P("  [Data source: stress_enhance_results (post Layer 1-4)]")
        survived = []
        for jf in sorted(stress_details.glob("*.json")):
            try:
                sd = json.loads(jf.read_text(encoding="utf-8"))
            except Exception:
                continue
            killed = (sd.get("killed") or sd.get("dtype_killed") or
                      sd.get("repeated_killed") or sd.get("training_killed"))
            if killed:
                continue
            mid = sd.get("mutant_id", jf.stem)
            kname = sd.get("kernel_name", "")
            op = sd.get("operator_name", "")
            detail = load_mutant_detail(kname, mid)
            if not detail:
                continue
            stress_info = {
                "l2_summary": "killed" if sd.get("dtype_killed") else "not killed",
                "l4_summary": "killed" if sd.get("training_killed") else "not killed",
                "original_failures": sd.get("original_failures", []),
                "policy_results": sd.get("policy_results", []),
            }
            survived.append({
                "mutant_id": mid, "kernel_name": kname,
                "operator_name": op, "detail": detail,
                "stress_summary": stress_info,
            })
        return survived

    P("  [Data source: full_block12_results (no stress test yet)]")
    survived = []
    for jf in sorted(BLOCK12_DIR.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        kernel_name = jf.stem
        for m in data.get("mutants", []):
            if m.get("status") == "survived":
                survived.append({
                    "mutant_id": m["id"], "kernel_name": kernel_name,
                    "operator_name": m.get("operator_name", ""),
                    "detail": m,
                    "stress_summary": {
                        "l2_summary": "not tested",
                        "l4_summary": "not tested",
                        "original_failures": [],
                    },
                })
    return survived


# ---------------------------------------------------------------------------
# Core: process one mutant (iterative analysis + verification)
# ---------------------------------------------------------------------------
def process_one_mutant(
    entry: Dict,
    caller,
    best_kernels: Dict,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """Iteratively analyze and verify one survived mutant.

    Returns a dict with all analysis results, rounds, and final outputs.
    """
    mutant_id = entry["mutant_id"]
    kernel_name = entry["kernel_name"]
    op_name = entry["operator_name"]
    detail = entry["detail"]

    mutated_code = detail.get("mutated_code", "")
    site = detail.get("site", {})

    bk = best_kernels.get(kernel_name, {})
    kpath = Path(bk.get("kernel_path", ""))
    kernel_code = ""
    if kpath.exists():
        kernel_code = kpath.read_text(encoding="utf-8")
    else:
        kernel_code = detail.get("kernel_code", detail.get("original_code", mutated_code))

    stress_summary = entry.get("stress_summary", {})
    problem_file = resolve_problem_file(kernel_name, best_kernels)
    input_spec = get_input_spec(detail, problem_file)

    result = {
        "mutant_id": mutant_id,
        "kernel_name": kernel_name,
        "operator_name": op_name,
        "status": "survived",
        "survival_reason": "",
        "killable": None,
        "kill_strategy": "",
        "recommendations": "",
        "rounds": [],
        "killed_by_llm": False,
        "killing_round": 0,
        "robustness_suggestion": "",
        "robustness_code": "",
        "test_construction_rule": None,
        "cluster_label": "",
    }

    rounds_history = []
    latest_analysis = None

    for round_num in range(1, max_rounds + 1):
        P(f"      Round {round_num}/{max_rounds}...")

        # Build prompt
        if round_num == 1:
            prompt = build_analysis_prompt(
                kernel_code, mutated_code, op_name, site,
                stress_summary, input_spec,
            )
        else:
            prompt = build_reanalysis_prompt(
                kernel_code, mutated_code, op_name, site,
                input_spec, rounds_history,
            )

        # Save prompt
        prompt_path = OUTPUT_DIR / "prompts" / f"{mutant_id}_r{round_num}.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        # Call LLM
        try:
            raw = caller(prompt)
            P(f"      Response: {len(raw)} chars")
        except Exception as e:
            P(f"      LLM ERROR: {e}")
            rounds_history.append({
                "round": round_num, "error": str(e),
                "killed": False, "detail": {},
            })
            result["rounds"] = rounds_history
            continue

        # Parse
        analysis = parse_analysis_response(raw, mutant_id, op_name, kernel_name)
        latest_analysis = analysis
        result["survival_reason"] = analysis.survival_reason
        result["killable"] = analysis.killable
        result["kill_strategy"] = analysis.kill_strategy
        result["recommendations"] = analysis.recommendations

        P(f"      Killable: {analysis.killable}")
        P(f"      Reason: {analysis.survival_reason[:120]}...")

        # If LLM says not killable or no suggested code → stop iterating
        if not analysis.killable or not analysis.suggested_test_code:
            P(f"      LLM says NOT killable → ending iteration")
            rounds_history.append({
                "round": round_num,
                "survival_reason": analysis.survival_reason,
                "kill_strategy": analysis.kill_strategy,
                "killable": False,
                "suggested_code": None,
                "killed": False,
                "detail": {},
            })
            result["rounds"] = rounds_history
            break

        # Verify suggested inputs
        if not problem_file:
            P(f"      SKIP verification: no problem file found")
            rounds_history.append({
                "round": round_num,
                "survival_reason": analysis.survival_reason,
                "kill_strategy": analysis.kill_strategy,
                "killable": True,
                "suggested_code": analysis.suggested_test_code,
                "killed": False,
                "detail": {"error": "no_problem_file"},
            })
            result["rounds"] = rounds_history
            continue

        P(f"      Verifying on GPU...")
        v_result = verify_one_mutant(
            kernel_code, mutated_code,
            analysis.suggested_test_code, problem_file,
        )

        killed = v_result.get("killed", False)
        round_record = {
            "round": round_num,
            "survival_reason": analysis.survival_reason,
            "kill_strategy": analysis.kill_strategy,
            "killable": True,
            "suggested_code": analysis.suggested_test_code,
            "killed": killed,
            "detail": {
                "ref_ok": v_result.get("ref_ok"),
                "original_ok": v_result.get("original_ok"),
                "mutant_ok": v_result.get("mutant_ok"),
                "diff_summary": v_result.get("diff_summary", ""),
                "error": v_result.get("error", ""),
            },
        }
        rounds_history.append(round_record)
        result["rounds"] = rounds_history

        if killed:
            P(f"      >>> KILLED in round {round_num}!")
            result["status"] = "killed_by_llm"
            result["killed_by_llm"] = True
            result["killing_round"] = round_num
            result["suggested_test_code"] = analysis.suggested_test_code

            # Extract test construction rule
            try:
                P(f"      Extracting test rule...")
                rule_prompt = build_test_rule_prompt(
                    op_name, kernel_name, analysis.survival_reason,
                    analysis.suggested_test_code,
                    analysis.suggested_test_desc or analysis.kill_strategy,
                )
                rule_raw = caller(rule_prompt)
                rule_data = parse_llm_response(rule_raw)
                if rule_data:
                    result["test_construction_rule"] = rule_data
                    P(f"      Rule: {rule_data.get('rule_name', 'N/A')}")
            except Exception as e:
                P(f"      Rule extraction error: {e}")

            return result
        else:
            P(f"      Not killed. diff={v_result.get('diff_summary', 'N/A')}")

    # All rounds exhausted → generate robustness suggestion
    result["status"] = "survived"
    result["rounds"] = rounds_history

    try:
        P(f"      Generating robustness suggestion...")
        rob_prompt = build_robustness_prompt(
            kernel_code, op_name, site,
            result["survival_reason"],
            len(rounds_history),
        )
        rob_raw = caller(rob_prompt)
        rob_data = parse_llm_response(rob_raw)
        if rob_data:
            result["robustness_suggestion"] = rob_data.get("robustness_suggestion", "")
            result["robustness_code"] = rob_data.get("robustness_code", "")
            P(f"      Robustness suggestion generated")
    except Exception as e:
        P(f"      Robustness generation error: {e}")

    return result


# ---------------------------------------------------------------------------
# Phase 3: Clustering
# ---------------------------------------------------------------------------
def run_clustering(survived_results: List[Dict], caller) -> Optional[Dict]:
    """Cluster all survival reasons into a taxonomy."""
    if not survived_results:
        return None

    P(f"\n  Clustering {len(survived_results)} survival reasons...")
    prompt = build_cluster_prompt(survived_results)

    prompt_path = OUTPUT_DIR / "prompts" / "_cluster_prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    try:
        raw = caller(prompt)
        taxonomy = parse_llm_response(raw)
        if taxonomy:
            # Apply labels back to results
            for cat in taxonomy.get("categories", []):
                label = cat.get("label", "")
                for mid in cat.get("mutant_ids", []):
                    for r in survived_results:
                        if r.get("mutant_id") == mid:
                            r["cluster_label"] = label
            return taxonomy
    except Exception as e:
        P(f"  Clustering error: {e}")

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not API_KEY:
        P("ERROR: Set LLM_API_KEY or DEEPSEEK_API_KEY environment variable")
        return

    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "details").mkdir(exist_ok=True)
    (OUTPUT_DIR / "prompts").mkdir(exist_ok=True)

    all_survived = collect_survived_mutants()
    targets = all_survived[:MAX_MUTANTS] if MAX_MUTANTS > 0 else all_survived

    # Resume support: check which mutants already have complete results
    already_done = {}
    for jf in (OUTPUT_DIR / "details").glob("*.json"):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            mid = d.get("mutant_id", jf.stem)
            if d.get("survival_reason") or d.get("killed_by_llm"):
                already_done[mid] = d
        except Exception:
            pass

    best_kernels = {}
    if BEST_KERNELS_FILE.exists():
        best_kernels = json.loads(BEST_KERNELS_FILE.read_text(encoding="utf-8"))

    P(f"\n{'='*70}")
    P(f"  Phase 2+3: LLM Iterative Analysis")
    P(f"{'='*70}")
    P(f"  Model: {MODEL}")
    P(f"  Max rounds per mutant: {MAX_ROUNDS}")
    P(f"  Total survived: {len(all_survived)}")
    P(f"  Targets: {len(targets)}")
    P(f"  Already done (will skip): {len(already_done)}")
    P("")

    caller = create_caller()

    all_results = []
    new_count = 0
    llm_errors = 0

    for idx, entry in enumerate(targets):
        mutant_id = entry["mutant_id"]
        kernel_name = entry["kernel_name"]
        op_name = entry["operator_name"]

        P(f"\n{'_'*70}")
        P(f"  [{idx+1}/{len(targets)}] {mutant_id}")
        P(f"    kernel={kernel_name}  op={op_name}")

        if mutant_id in already_done:
            P(f"    SKIP (already analyzed)")
            all_results.append(already_done[mutant_id])
            continue

        if not entry["detail"].get("mutated_code"):
            P(f"    SKIP: no mutated_code")
            continue

        try:
            result = process_one_mutant(entry, caller, best_kernels, MAX_ROUNDS)
        except Exception as e:
            P(f"    FATAL ERROR: {e}")
            llm_errors += 1
            result = {
                "mutant_id": mutant_id, "kernel_name": kernel_name,
                "operator_name": op_name, "status": "error",
                "error": str(e)[:500],
            }

        # Save per-mutant result
        detail_path = OUTPUT_DIR / "details" / f"{mutant_id}.json"
        detail_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        all_results.append(result)
        new_count += 1

    # -----------------------------------------------------------------------
    # Phase 3: Clustering
    # -----------------------------------------------------------------------
    survived_for_cluster = [
        r for r in all_results
        if r.get("status") == "survived" and r.get("survival_reason")
    ]

    taxonomy = None
    if survived_for_cluster:
        try:
            taxonomy = run_clustering(survived_for_cluster, caller)
            if taxonomy:
                (OUTPUT_DIR / "taxonomy.json").write_text(
                    json.dumps(taxonomy, indent=2, ensure_ascii=False),
                    encoding="utf-8")
                P(f"  Taxonomy saved ({len(taxonomy.get('categories', []))} categories)")

                # Update detail files with cluster labels
                for r in survived_for_cluster:
                    if r.get("cluster_label"):
                        dp = OUTPUT_DIR / "details" / f"{r['mutant_id']}.json"
                        if dp.exists():
                            d = json.loads(dp.read_text(encoding="utf-8"))
                            d["cluster_label"] = r["cluster_label"]
                            dp.write_text(json.dumps(d, indent=2, ensure_ascii=False),
                                          encoding="utf-8")
        except Exception as e:
            P(f"  Clustering failed: {e}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start

    killed_by_llm = [r for r in all_results if r.get("killed_by_llm")]
    survived_final = [r for r in all_results if r.get("status") == "survived"]
    has_robustness = [r for r in survived_final if r.get("robustness_suggestion")]
    has_rules = [r for r in killed_by_llm if r.get("test_construction_rule")]

    # Collect robustness suggestions
    robustness_list = []
    for r in has_robustness:
        robustness_list.append({
            "mutant_id": r["mutant_id"],
            "kernel_name": r["kernel_name"],
            "operator": r["operator_name"],
            "survival_reason": r.get("survival_reason", ""),
            "suggestion": r["robustness_suggestion"],
            "fixed_code": r.get("robustness_code", ""),
        })
    if robustness_list:
        (OUTPUT_DIR / "robustness_suggestions.json").write_text(
            json.dumps(robustness_list, indent=2, ensure_ascii=False),
            encoding="utf-8")

    # Collect test rules
    rules_list = []
    for r in has_rules:
        rule = r["test_construction_rule"]
        rule["mutant_id"] = r["mutant_id"]
        rule["kernel_name"] = r["kernel_name"]
        rules_list.append(rule)
    if rules_list:
        (OUTPUT_DIR / "test_construction_rules.json").write_text(
            json.dumps(rules_list, indent=2, ensure_ascii=False),
            encoding="utf-8")

    summary = {
        "total_analyzed": len(all_results),
        "newly_analyzed": new_count,
        "killed_by_llm": len(killed_by_llm),
        "survived_final": len(survived_final),
        "has_robustness_suggestion": len(has_robustness),
        "has_test_rule": len(has_rules),
        "llm_errors": llm_errors,
        "taxonomy_categories": len(taxonomy.get("categories", [])) if taxonomy else 0,
        "elapsed_min": round(elapsed / 60, 1),
        "model": MODEL,
        "max_rounds": MAX_ROUNDS,
    }

    P(f"\n{'='*70}")
    P(f"  Phase 2+3 Complete")
    P(f"{'='*70}")
    P(f"  Total analyzed:         {len(all_results)}")
    P(f"  Newly analyzed:         {new_count}")
    P(f"  Killed by LLM:          {len(killed_by_llm)}")
    P(f"  Survived (final):       {len(survived_final)}")
    P(f"  With robustness advice: {len(has_robustness)}")
    P(f"  With test rules:        {len(has_rules)}")
    P(f"  Taxonomy categories:    {summary['taxonomy_categories']}")
    P(f"  LLM errors:             {llm_errors}")
    P(f"  Elapsed: {elapsed/60:.1f} min")

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    P(f"  Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
