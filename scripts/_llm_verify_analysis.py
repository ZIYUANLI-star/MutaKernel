"""Deep analysis of LLM verification failures — why 0 kills."""
import json
import glob
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DETAILS_DIR = PROJECT / "llm_analysis_results" / "details"

ref_crash_examples = []
success_examples = []
other_failures = []

for jf in sorted(DETAILS_DIR.glob("*.json")):
    d = json.loads(jf.read_text())
    for r in d.get("rounds", []):
        detail = r.get("detail", {})
        error = detail.get("error", "")
        code = r.get("suggested_code", "")
        
        if not code:
            continue
        
        if error.startswith("ref crash"):
            # Extract the return statement from suggested code
            return_lines = []
            for line in code.split("\n"):
                stripped = line.strip()
                if stripped.startswith("return"):
                    return_lines.append(stripped)
            
            ref_crash_examples.append({
                "mutant": d["mutant_id"],
                "round": r["round"],
                "error": error[:120],
                "return_lines": return_lines,
            })
        elif r.get("killed"):
            success_examples.append({
                "mutant": d["mutant_id"],
                "round": r["round"],
            })
        else:
            other_failures.append({
                "mutant": d["mutant_id"],
                "round": r["round"],
                "error": error[:120] if error else "no error field",
                "ref_ok": detail.get("ref_ok"),
                "orig_ok": detail.get("original_ok"),
                "mut_ok": detail.get("mutant_ok"),
            })

print("=" * 70)
print("LLM VERIFICATION DEEP ANALYSIS")
print("=" * 70)
print()

print(f"Total ref crashes: {len(ref_crash_examples)}")
print(f"Total other failures: {len(other_failures)}")
print(f"Total successful kills: {len(success_examples)}")
print()

# Core problem: what are the return statements?
print("=== Ref Crash — LLM Return Statements (first 10) ===")
for ex in ref_crash_examples[:10]:
    print(f"  {ex['mutant']} R{ex['round']}: {ex['error']}")
    for rl in ex["return_lines"]:
        print(f"    {rl[:120]}")
    print()

# Other failures
print("=== Other Failures ===")
for ex in other_failures:
    print(f"  {ex['mutant']} R{ex['round']}: ref_ok={ex['ref_ok']} orig_ok={ex['orig_ok']} mut_ok={ex['mut_ok']} err={ex['error']}")
