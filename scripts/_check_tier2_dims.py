#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "第二次实验汇总" / "stress_enhance_results"
DETAILS = RES / "details"

for f in sorted(DETAILS.glob("*.json")):
    d = json.loads(f.read_text())
    if d.get("tier") != 2:
        continue
    mid = f.stem
    print(f"{'='*70}")
    print(f"{mid} (operator={d.get('operator_name','')})")
    
    # Check each dimension
    dims = [
        "tier1_replay", "value_stress", "dtype_stress",
        "training_stress", "repeated_run", "config_stress"
    ]
    for dim in dims:
        section = d.get(dim)
        if section is None:
            print(f"  {dim}: NOT PRESENT")
            continue
        
        if isinstance(section, dict):
            killed = section.get("killed", section.get("any_killed", "?"))
            policies_run = 0
            total_runs = 0
            
            if "policies" in section:
                policies_run = len(section["policies"])
                for p in section["policies"]:
                    if isinstance(p, dict) and "runs" in p:
                        total_runs += len(p["runs"])
            elif "results" in section:
                total_runs = len(section["results"])
            elif "runs" in section:
                total_runs = len(section["runs"])
            
            skip = section.get("skipped", False)
            reason = section.get("skip_reason", "")
            
            if skip or reason:
                print(f"  {dim}: SKIPPED ({reason})")
            else:
                print(f"  {dim}: killed={killed}, policies={policies_run}, runs={total_runs}")
        else:
            print(f"  {dim}: type={type(section).__name__}")
    
    # LLM
    llm = d.get("llm_iterative_analysis", {})
    print(f"  llm_iterative: executed={llm.get('executed')}, rounds={len(llm.get('rounds',[]))}, killed={llm.get('killed')}")
    print()
