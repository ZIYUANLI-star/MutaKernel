#!/usr/bin/env python3
"""Detailed analysis of LLM iterative analysis results."""
import json, os
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "第二次实验汇总" / "stress_enhance_results" / "details"
LLM_RESP = ROOT / "第二次实验汇总" / "stress_enhance_results" / "llm_responses"
PROMPTS = ROOT / "第二次实验汇总" / "stress_enhance_results" / "prompts"

if not D.exists():
    print("No details dir"); exit()

# Gather all LLM analysis info
llm_results = []
for f in sorted(D.glob("*.json")):
    data = json.loads(f.read_text())
    llm = data.get("llm_iterative_analysis", {})
    if not llm.get("executed"):
        continue
    
    mid = data["mutant_id"]
    op = data.get("operator_name", "")
    tier = data.get("tier", "?")
    n_rounds = len(llm.get("rounds", []))
    llm_killed = llm.get("killed", False)
    
    rounds_detail = []
    for r in llm.get("rounds", []):
        rd = {
            "round": r.get("round"),
            "prompt_type": r.get("prompt_type", ""),
            "killable": r.get("killable"),
            "reason_category": r.get("reason_category", ""),
            "has_code": bool(r.get("suggested_code", "")),
            "killed": r.get("killed", False),
            "exec_error": "",
        }
        er = r.get("execution_result")
        if er:
            rd["exec_error"] = er.get("error", "")
            rd["exec_killed"] = er.get("killed", False)
            rd["ref_ok"] = er.get("ref_ok", True)
            rd["orig_ok"] = er.get("original_ok", True)
            rd["mut_ok"] = er.get("mutant_ok", True)
        rounds_detail.append(rd)
    
    llm_results.append({
        "mutant_id": mid,
        "operator": op,
        "tier": tier,
        "n_rounds": n_rounds,
        "killed": llm_killed,
        "rounds": rounds_detail,
    })

print(f"=== LLM Iterative Analysis Detail ===")
print(f"Total mutants with LLM analysis: {len(llm_results)}")
print(f"LLM killed: {sum(1 for r in llm_results if r['killed'])}")
print(f"LLM not killed: {sum(1 for r in llm_results if not r['killed'])}")

# Category distribution
cats = Counter()
for lr in llm_results:
    for rd in lr["rounds"]:
        if rd.get("reason_category"):
            cats[rd["reason_category"]] += 1

print(f"\n--- reason_category distribution ---")
for cat, cnt in cats.most_common():
    print(f"  {cat}: {cnt}")

# Killable analysis
killable_count = 0
unkillable_count = 0
code_generated = 0
code_executed = 0
code_killed = 0
code_errors = Counter()

for lr in llm_results:
    for rd in lr["rounds"]:
        if rd.get("killable"):
            killable_count += 1
            if rd.get("has_code"):
                code_generated += 1
                er = rd.get("exec_error", "")
                if rd.get("execution_result") is not None or "exec_killed" in rd:
                    code_executed += 1
                    if rd.get("exec_killed") or rd.get("killed"):
                        code_killed += 1
                    if er:
                        code_errors[er[:60]] += 1
        else:
            unkillable_count += 1

print(f"\n--- Killable assessment ---")
print(f"  LLM said killable:   {killable_count}")
print(f"  LLM said unkillable: {unkillable_count}")
print(f"  Code generated:      {code_generated}")
print(f"  Code executed:       {code_executed}")
print(f"  Code killed:         {code_killed}")

if code_errors:
    print(f"\n--- Execution errors ---")
    for err, cnt in code_errors.most_common():
        print(f"  [{cnt}x] {err}")

# Round distribution
round_dist = Counter()
for lr in llm_results:
    round_dist[lr["n_rounds"]] += 1

print(f"\n--- Rounds distribution ---")
for n, cnt in sorted(round_dist.items()):
    print(f"  {n} round(s): {cnt} mutants")

# Per-mutant detail
print(f"\n--- Per mutant detail ---")
for lr in llm_results:
    mid = lr["mutant_id"]
    op = lr["operator"]
    status = "KILLED" if lr["killed"] else "survived"
    rounds_str = []
    for rd in lr["rounds"]:
        k = "killable" if rd.get("killable") else "unkillable"
        cat = rd.get("reason_category", "?")
        code = "has_code" if rd.get("has_code") else "no_code"
        exec_r = ""
        if "exec_killed" in rd:
            exec_r = "->KILLED" if rd["exec_killed"] else "->survived"
            if rd.get("exec_error"):
                exec_r += f"(err:{rd['exec_error'][:40]})"
        rounds_str.append(f"R{rd['round']}:{k}/{cat}/{code}{exec_r}")
    
    print(f"  {mid} ({op}) [{status}]")
    for rs in rounds_str:
        print(f"    {rs}")
