"""Analyze why P39 kernel causes timeouts."""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
bk = json.load(open(PROJECT / "best_kernels.json"))

for key in bk:
    if "P39" in key:
        info = bk[key]
        kpath = Path(info["kernel_path"])
        print(f"Key: {key}")
        print(f"Path: {kpath}")
        print(f"Exists: {kpath.exists()}")
        if kpath.exists():
            code = kpath.read_text()
            print(f"Code length: {len(code)} chars, {len(code.splitlines())} lines")
            for indicator in ["for ", "while ", "__global__", "load_inline", "triton"]:
                count = code.count(indicator)
                if count > 0:
                    print(f"  '{indicator}': {count} occurrences")
            print()
            print("=== Full code ===")
            for i, line in enumerate(code.splitlines(), 1):
                print(f"{i:3d}| {line}")

# Also check block12 results for P39
b12 = PROJECT / "full_block12_results" / "details"
for jf in sorted(b12.glob("*.json")):
    if "P39" in jf.name:
        data = json.loads(jf.read_text())
        km = data["kernel"]
        print(f"\n=== Block12 info for {jf.name} ===")
        print(f"Problem: P{km['problem_id']}, Level: {km['level']}")
        print(f"Language: {km.get('language', '?')}")
        total = len(data.get("mutants", []))
        survived = sum(1 for m in data.get("mutants", []) if m["status"] == "survived")
        print(f"Mutants: {total} total, {survived} survived")
        for m in data.get("mutants", []):
            if m["status"] == "survived":
                site = m.get("site", {})
                print(f"  Survived: {m['id']:40s} | {m['operator_name']:18s} | L{site.get('line_start','?')}")

# Check previous completion time for nearby kernels
stress = PROJECT / "stress_enhance_results" / "details"
print("\n=== Execution times for recent kernels ===")
for jf in sorted(stress.glob("*.json")):
    d = json.loads(jf.read_text())
    t = d.get("total_time_ms", 0)
    mid = d["mutant_id"]
    if "P38" in mid or "P39" in mid:
        print(f"  {mid:45s} | {t/1000:.1f}s | killed={d.get('killed', False)}")
