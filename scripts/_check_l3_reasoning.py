#!/usr/bin/env python3
"""Print Layer 3 LLM reasoning for all equiv-checked mutants."""
import json
from pathlib import Path

result_dir = Path(__file__).resolve().parent.parent / "第二次实验汇总" / "full_block12_results" / "details"

for jf in sorted(result_dir.glob("*.json")):
    with open(jf) as f:
        d = json.load(f)
    print(f"\n{'='*60}")
    print(f"Kernel: {d.get('kernel_name', jf.stem)}")
    print(f"{'='*60}")
    for m in d["mutants"]:
        ed = m.get("equiv_detail", {})
        l3 = ed.get("layer3", {})
        if l3 or ed:
            print(f"\n  --- {m['operator_name']} @ L{ed.get('line_start','?')} ---")
            print(f"  final_status: {m['status']}")
            print(f"  decided_at: {ed.get('decided_at', '?')}")
            if l3:
                print(f"  L3 verdict: {l3.get('verdict')}")
                print(f"  L3 confidence: {l3.get('confidence')}")
                r = l3.get("reasoning", "")
                print(f"  L3 reasoning: {r[:400]}")
                ks = l3.get("kill_strategy")
                if ks:
                    print(f"  L3 kill_strategy: {str(ks)[:300]}")
