"""Analyze quality of LLM's killable=false judgments.
For each: is it a genuine equivalent mutant analysis or a lazy give-up?"""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DETAILS_DIR = PROJECT / "llm_analysis_results" / "details"

cases = []
for jf in sorted(DETAILS_DIR.glob("*.json")):
    d = json.loads(jf.read_text())
    if d.get("killed_by_llm"):
        continue
    
    rounds = d.get("rounds", [])
    
    # Classify: did LLM try and fail, or immediately say unkillable?
    tried_and_failed = False
    immediate_unkillable = False
    gave_up_after_trying = False
    
    killable_rounds = [r for r in rounds if r.get("killable") is True]
    unkillable_rounds = [r for r in rounds if r.get("killable") is False]
    
    if len(rounds) == 1 and not rounds[0].get("killable"):
        immediate_unkillable = True
        category = "IMMEDIATE_UNKILLABLE"
    elif killable_rounds and unkillable_rounds:
        gave_up_after_trying = True
        category = "TRIED_THEN_GAVE_UP"
    elif killable_rounds and not unkillable_rounds:
        tried_and_failed = True
        category = "TRIED_ALL_ROUNDS_FAILED"
    elif len(rounds) > 1 and all(not r.get("killable") for r in rounds):
        immediate_unkillable = True
        category = "IMMEDIATE_UNKILLABLE"
    else:
        category = "OTHER"
    
    # Check verification results
    verify_attempted = 0
    verify_both_ok = 0
    verify_ref_crash = 0
    verify_orig_fail = 0
    for r in rounds:
        if r.get("suggested_code"):
            verify_attempted += 1
            detail = r.get("detail", {})
            if detail.get("ref_ok") and detail.get("original_ok") and detail.get("mutant_ok"):
                verify_both_ok += 1
            elif detail.get("error", "").startswith("ref crash"):
                verify_ref_crash += 1
            elif detail.get("original_ok") is False:
                verify_orig_fail += 1
    
    reason = d.get("survival_reason", "")
    
    cases.append({
        "id": d["mutant_id"],
        "op": d["operator_name"],
        "kernel": d["kernel_name"],
        "category": category,
        "n_rounds": len(rounds),
        "killable_rounds": len(killable_rounds),
        "unkillable_rounds": len(unkillable_rounds),
        "verify_attempted": verify_attempted,
        "verify_both_ok": verify_both_ok,
        "verify_ref_crash": verify_ref_crash,
        "verify_orig_fail": verify_orig_fail,
        "reason": reason,
        "final_killable": d.get("killable"),
    })

# Print analysis
from collections import Counter
cat_counter = Counter(c["category"] for c in cases)

print("=" * 80)
print(f"  LLM killable=false QUALITY ANALYSIS ({len(cases)} cases)")
print("=" * 80)

print(f"\n--- Category Distribution ---")
for cat, cnt in cat_counter.most_common():
    print(f"  {cat}: {cnt}")

print(f"\n{'=' * 80}")
print(f"  A. IMMEDIATE UNKILLABLE (Round 1 直接判定不可杀)")
print(f"{'=' * 80}")
imm = [c for c in cases if c["category"] == "IMMEDIATE_UNKILLABLE"]
for c in imm:
    reason_short = c["reason"][:250] if c["reason"] else "NO REASON"
    print(f"\n  {c['id']} (op={c['op']})")
    print(f"    Reason: {reason_short}...")

print(f"\n{'=' * 80}")
print(f"  B. TRIED THEN GAVE UP (先判可杀, 验证失败后改判不可杀)")
print(f"{'=' * 80}")
tried = [c for c in cases if c["category"] == "TRIED_THEN_GAVE_UP"]
for c in tried:
    reason_short = c["reason"][:250] if c["reason"] else "NO REASON"
    print(f"\n  {c['id']} (op={c['op']})")
    print(f"    Rounds: {c['n_rounds']} (killable={c['killable_rounds']}, unkillable={c['unkillable_rounds']})")
    print(f"    Verified: {c['verify_attempted']} (both_ok={c['verify_both_ok']}, ref_crash={c['verify_ref_crash']}, orig_fail={c['verify_orig_fail']})")
    print(f"    Final reason: {reason_short}...")

print(f"\n{'=' * 80}")
print(f"  C. TRIED ALL ROUNDS BUT NEVER GAVE UP (始终判可杀但都没杀死)")
print(f"{'=' * 80}")
all_tried = [c for c in cases if c["category"] == "TRIED_ALL_ROUNDS_FAILED"]
for c in all_tried:
    reason_short = c["reason"][:250] if c["reason"] else "NO REASON"
    print(f"\n  {c['id']} (op={c['op']})")
    print(f"    Rounds: {c['n_rounds']} (all killable=True)")
    print(f"    Verified: {c['verify_attempted']} (both_ok={c['verify_both_ok']}, ref_crash={c['verify_ref_crash']})")
    print(f"    Final reason: {reason_short}...")
