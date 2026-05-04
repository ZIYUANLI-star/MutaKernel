import json, os

d = os.path.join(os.path.dirname(__file__), "..", "full_block12_results", "details")
for cat in ["A", "B", "C"]:
    print(f"\n=== Category {cat} SURVIVED ===")
    count = 0
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        data = json.load(open(os.path.join(d, f)))
        kname = data["kernel"]["problem_name"]
        for m in data["mutants"]:
            if m["status"] == "survived" and m["operator_category"] == cat:
                mc = m.get("mutated_code", "")
                has_code = "YES" if mc else "NO"
                print(f"  {kname}: id={m['id']}  op={m['operator_name']}  "
                      f"L{m['site']['line_start']}  "
                      f"orig_snippet=\"{m['site']['original_code'][:80]}\"  "
                      f"has_mutated_code={has_code}")
                count += 1
                if count >= 5:
                    break
        if count >= 5:
            break
