"""Analyze LLM Phase 2 results — verification details."""
import json
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LLM_DIR = PROJECT / "llm_analysis_results" / "details"

total = 0
killed_by_llm = 0
not_killed = 0
no_verify = 0
killable_true_total = 0
killable_false_total = 0
rounds_total = 0
verify_details = []

for jf in sorted(LLM_DIR.glob("*.json")):
    d = json.loads(jf.read_text())
    total += 1
    rounds = d.get("rounds", [])
    rounds_total += len(rounds)
    
    if d.get("killed_by_llm"):
        killed_by_llm += 1
    
    killable = d.get("killable")
    if killable is True:
        killable_true_total += 1
    elif killable is False:
        killable_false_total += 1
    
    # Check each round's verification
    for r in rounds:
        if r.get("killed"):
            verify_details.append({
                "mutant": d["mutant_id"],
                "round": r["round"],
                "result": "KILLED",
            })
        elif r.get("detail", {}).get("error") == "no_problem_file":
            no_verify += 1
        elif r.get("suggested_code") and not r.get("killed"):
            detail = r.get("detail", {})
            verify_details.append({
                "mutant": d["mutant_id"],
                "round": r["round"],
                "result": "not killed",
                "ref_ok": detail.get("ref_ok"),
                "orig_ok": detail.get("original_ok"),
                "mut_ok": detail.get("mutant_ok"),
                "diff": detail.get("diff_summary", ""),
                "error": detail.get("error", ""),
            })

print(f"=== LLM Phase 2 Results ===")
print(f"Total completed: {total}")
print(f"Final killable=True: {killable_true_total}")
print(f"Final killable=False: {killable_false_total}")
print(f"Total rounds: {rounds_total}")
print(f"Killed by LLM: {killed_by_llm}")
print(f"Skipped verification: {no_verify}")
print()

# Count verification outcomes
killed_verifs = [v for v in verify_details if v["result"] == "KILLED"]
failed_verifs = [v for v in verify_details if v["result"] == "not killed"]
print(f"=== GPU Verification Results ===")
print(f"Total verification attempts: {len(verify_details)}")
print(f"KILLED: {len(killed_verifs)}")
print(f"Not killed: {len(failed_verifs)}")
print()

if killed_verifs:
    print(f"=== Successfully Killed ===")
    for v in killed_verifs:
        print(f"  {v['mutant']} (round {v['round']})")
    print()

# Show failure reasons
error_counter = Counter()
for v in failed_verifs:
    if v.get("error"):
        error_counter[v["error"][:80]] += 1
    elif v.get("ref_ok") is False:
        error_counter["ref_fail"] += 1
    elif v.get("orig_ok") is False:
        error_counter["orig_also_fails"] += 1
    elif v.get("mut_ok") is True:
        error_counter["both_ok (mutant matches ref)"] += 1
    else:
        error_counter["unknown"] += 1

print(f"=== Verification Failure Reasons ===")
for reason, count in error_counter.most_common():
    print(f"  {reason}: {count}")

# Show a few examples with diff
print(f"\n=== Sample Verifications with Diff ===")
shown = 0
for v in failed_verifs:
    if v.get("diff") and shown < 5:
        print(f"  {v['mutant']} R{v['round']}: ref_ok={v.get('ref_ok')} orig_ok={v.get('orig_ok')} mut_ok={v.get('mut_ok')}")
        print(f"    diff: {v['diff'][:120]}")
        shown += 1
