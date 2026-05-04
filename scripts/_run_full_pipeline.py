"""Full pipeline: Layer 1-4 stress test → Layer 5 LLM attribution + verification.

Run inside WSL with GPU:
  conda run -n RLHF python scripts/_run_full_pipeline.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STRESS_RESULT_DIR = PROJECT_ROOT / "stress_enhance_results"
LLM_RESULT_DIR = PROJECT_ROOT / "llm_attribution_results"

def P(msg):
    print(msg, flush=True)


def run_phase1():
    """Phase 1: 4-layer stress testing (all 322 survived mutants)."""
    P("\n" + "=" * 70)
    P("  PHASE 1: 4-Layer Stress Enhancement (322 mutants)")
    P("=" * 70 + "\n")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_stress_enhance.py")],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0


def collect_still_survived():
    """After stress testing, collect mutants that still survived."""
    details_dir = STRESS_RESULT_DIR / "details"
    if not details_dir.exists():
        return []
    still_survived = []
    for jf in sorted(details_dir.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        killed = d.get("killed") or d.get("dtype_killed") or \
                 d.get("repeated_killed") or d.get("training_killed")
        if not killed:
            still_survived.append(d)
    return still_survived


def run_phase2_llm(still_survived: list):
    """Phase 2: LLM attribution for mutants that survived all 4 layers."""
    P("\n" + "=" * 70)
    P(f"  PHASE 2: LLM Attribution ({len(still_survived)} still-survived mutants)")
    P("=" * 70 + "\n")

    # Write the list of target mutant IDs for the LLM script
    target_ids = [d.get("mutant_id", d.get("id", "")) for d in still_survived]
    target_file = STRESS_RESULT_DIR / "still_survived_ids.json"
    target_file.write_text(json.dumps(target_ids, indent=2), encoding="utf-8")
    P(f"  Wrote {len(target_ids)} target IDs to {target_file}")

    # Set env and run the LLM attribution script
    env = os.environ.copy()
    env["LLM_MAX_MUTANTS"] = "0"
    env["LLM_TARGET_IDS"] = str(target_file)
    # API key should be set in environment before running

    if not env.get("DEEPSEEK_API_KEY"):
        P("  WARNING: DEEPSEEK_API_KEY not set. Skipping LLM phase.")
        P("  Run manually: python scripts/_pilot_llm20.py")
        return False

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "_pilot_llm20.py")],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    return result.returncode == 0


def run_phase3_verify():
    """Phase 3: Verify LLM-suggested inputs."""
    P("\n" + "=" * 70)
    P("  PHASE 3: Verify LLM-Suggested Inputs")
    P("=" * 70 + "\n")

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "_verify_llm_suggestions.py")],
        cwd=str(PROJECT_ROOT),
    )
    return result.returncode == 0


def main():
    t_start = time.time()

    # Phase 1: Stress testing
    ok = run_phase1()
    if not ok:
        P("Phase 1 failed!")
        return

    # Collect still-survived
    still_survived = collect_still_survived()
    summary_path = STRESS_RESULT_DIR / "stress_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        total = summary.get("total_survived_tested", "?")
        killed = summary.get("killed_by_stress", "?")
    else:
        total = "?"
        killed = "?"

    P(f"\n  Phase 1 complete: {total} tested, {killed} killed, "
      f"{len(still_survived)} still survived")

    if not still_survived:
        P("  All mutants killed! No LLM attribution needed.")
        return

    # Phase 2: LLM attribution
    ok = run_phase2_llm(still_survived)

    # Phase 3: Verify (only if Phase 2 succeeded and we're in GPU env)
    if ok:
        run_phase3_verify()

    elapsed = time.time() - t_start
    P(f"\n{'='*70}")
    P(f"  Full Pipeline Complete ({elapsed/3600:.1f} hours)")
    P(f"{'='*70}")


if __name__ == "__main__":
    main()
