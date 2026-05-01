#!/usr/bin/env python3
"""Check survived mutants and their LLM analysis status."""
import json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "第二次实验汇总" / "stress_enhance_results" / "details"

if not D.exists():
    print("No details dir"); exit()

survived = []
for f in sorted(D.glob("*.json")):
    data = json.loads(f.read_text())
    if not data.get("any_killed"):
        llm = data.get("llm_iterative_analysis", {})
        n_rounds = len(llm.get("rounds", []))
        trigger = llm.get("trigger", "")
        executed = llm.get("executed", False)
        last_killable = ""
        if llm.get("rounds"):
            last = llm["rounds"][-1]
            last_killable = last.get("killable", "?")
        survived.append((f.stem, data.get("operator_name",""), n_rounds, trigger, executed, last_killable))

print(f"Total survived: {len(survived)}")
for name, op, nr, tr, ex, lk in survived:
    print(f"  {name} ({op}): llm_exec={ex}, rounds={nr}, trigger={tr}, last_killable={lk}")
