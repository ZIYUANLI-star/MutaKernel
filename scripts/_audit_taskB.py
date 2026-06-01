"""Audit Task B results: detect 'real fixes' vs 'unexpected pass' artifacts.

For each kernel, classify:
- `truly_fixed`:   round0 confirmed >=1 buggy AND a later round passed V_stress+V_kb
- `pseudo_fixed`:  reported fixed BUT round0 had all/most inputs unexpectedly passing
- `failed`:        rounds done but did not pass V_stress+V_kb
- `error`:         pipeline failed
"""
import json, os, glob
from pathlib import Path

DET = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_b_regenerate/details")

rows = []
for f in sorted(DET.glob("*.json")):
    d = json.load(open(f))
    name = d.get("kernel_name", f.stem)
    n_failing = d.get("n_failing_inputs", 0)
    r0 = d.get("round0_stats", {})
    r0_total = r0.get("n_total", 0)
    r0_confirmed = r0.get("n_confirmed_buggy", 0)
    r0_unexp = r0.get("n_unexpected_pass", 0)
    r0_reffail = r0.get("n_ref_failed", 0)
    final_status = d.get("final_status", "?")
    final_round = d.get("final_round", 0)
    rounds = d.get("rounds", [])
    elapsed = d.get("elapsed_sec", 0)

    # Per-round V_stress / V_kb summary
    round_brief = []
    for r in rounds:
        rn = r.get("round")
        vs = r.get("v_stress") or {}
        vk = r.get("v_kb") or {}
        round_brief.append(f"R{rn}: Vs={vs.get('n_pass',0)}/{vs.get('n_total',0)} "
                            f"Vkb={vk.get('n_pass',0)}/{vk.get('n_total',0)}"
                            f"{' [PASS]' if r.get('round_pass') else ''}")

    # Classification
    if final_status.startswith("fixed"):
        if r0_confirmed == 0 and r0_unexp > 0:
            klass = "PSEUDO_FIX (round0 all unexpected_pass)"
        elif r0_unexp >= r0_total * 0.5 and r0_total > 0:
            klass = f"PARTIAL_PSEUDO ({r0_unexp}/{r0_total} unexpected_pass)"
        else:
            klass = "TRULY_FIXED"
    elif final_status == "failed_after_3_rounds":
        klass = "FAILED"
    elif final_status == "skipped_no_failing_inputs":
        klass = "SKIPPED_NO_BUGGY"
    else:
        klass = f"OTHER:{final_status}"

    rows.append({
        "name": name,
        "n_in": n_failing,
        "r0_confirmed": r0_confirmed,
        "r0_unexp": r0_unexp,
        "r0_reffail": r0_reffail,
        "final_status": final_status,
        "final_round": final_round,
        "klass": klass,
        "rounds": "  ".join(round_brief),
        "elapsed_min": round(elapsed/60, 1),
    })

# Sort by class then name
order = {"TRULY_FIXED": 0, "PARTIAL_PSEUDO": 1, "PSEUDO_FIX (round0 all unexpected_pass)": 2,
         "FAILED": 3, "SKIPPED_NO_BUGGY": 4}
rows.sort(key=lambda r: (order.get(r["klass"].split(" ")[0], 99), r["name"]))

print(f"{'Kernel':<10} {'#in':>4} {'r0_cf':>5} {'r0_un':>5} {'r0_rf':>5} "
      f"{'final':<22} {'class':<35} {'min':>5}")
print("-" * 130)
for r in rows:
    print(f"{r['name']:<10} {r['n_in']:>4} {r['r0_confirmed']:>5} {r['r0_unexp']:>5} "
          f"{r['r0_reffail']:>5} {r['final_status']:<22} {r['klass']:<35} "
          f"{r['elapsed_min']:>5}")
print()
for r in rows:
    print(f"  {r['name']:<10} {r['rounds']}")

# Summary
from collections import Counter
klass_counter = Counter(r["klass"].split(" ")[0] for r in rows)
print()
print("=" * 60)
print("Classification summary:")
for k, v in klass_counter.most_common():
    print(f"  {k:<35} {v:>3}")
print(f"  TOTAL                              {len(rows):>3}")

print()
print("=== Drilling into PSEUDO_FIX details (R0 unexpected_pass) ===")
for r in rows:
    if r["klass"].startswith("PSEUDO_FIX") or r["klass"].startswith("PARTIAL_PSEUDO"):
        print(f"  {r['name']}: n_failing={r['n_in']}, r0_confirmed={r['r0_confirmed']}, "
              f"r0_unexpected_pass={r['r0_unexp']}, r0_ref_failed={r['r0_reffail']}")
