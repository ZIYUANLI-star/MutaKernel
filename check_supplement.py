"""Check supplement test progress and results."""
import json, os

details_dir = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\第二次实验汇总\stress_enhance_results\details'
completed_file = r'd:\doctor_learning\Academic_Project\paper_1\MutaKernel\第二次实验汇总\stress_enhance_results\completed.json'

original_499 = [
    "L1_P10__arith_replace__23", "L1_P12__arith_replace__17",
    "L1_P13__arith_replace__4", "L1_P13__cast_remove__0", "L1_P13__cast_remove__1",
    "L1_P14__arith_replace__0", "L1_P14__arith_replace__30", "L1_P14__arith_replace__33",
    "L1_P15__arith_replace__11", "L1_P19__arith_replace__3",
    "L1_P20__arith_replace__18", "L1_P23__init_modify__0",
    "L1_P24__cast_remove__0", "L1_P24__init_modify__0",
    "L1_P25__arith_replace__12", "L1_P27__arith_replace__2", "L1_P27__arith_replace__6",
    "L1_P29__arith_replace__16", "L1_P29__arith_replace__19", "L1_P29__cast_remove__4",
    "L1_P3__arith_replace__31", "L1_P31__arith_replace__3",
    "L1_P32__arith_replace__9", "L1_P32__arith_replace__8",
    "L1_P33__arith_replace__58", "L1_P38__cast_remove__0", "L1_P38__cast_remove__3",
    "L1_P42__init_modify__0", "L1_P5__arith_replace__5",
    "L1_P51__arith_replace__15", "L1_P6__arith_replace__0",
    "L1_P91__arith_replace__30", "L2_P58__arith_replace__9",
    "L2_P66__arith_replace__14", "L2_P66__scale_modify__0",
]

completed = set(json.loads(open(completed_file).read()))
total_details = len([f for f in os.listdir(details_dir) if f.endswith('.json')])

done = []
pending = []
killed_list = []
survived_list = []

for mid in original_499:
    fp = os.path.join(details_dir, f"{mid}.json")
    if os.path.exists(fp):
        d = json.loads(open(fp, encoding='utf-8').read())
        ak = d.get('any_killed', False)
        fkm = d.get('first_kill_mode', '')
        done.append(mid)
        if ak:
            killed_list.append((mid, fkm))
        else:
            survived_list.append(mid)
    else:
        pending.append(mid)

print(f"=== Supplement Test Progress ===")
print(f"Total detail files: {total_details}")
print(f"Completed (in completed.json): {len(completed)}")
print(f"")
print(f"--- 35 Missing Mutants ---")
print(f"Done:    {len(done)}/35")
print(f"Pending: {len(pending)}/35")
print(f"")
print(f"--- Results ---")
print(f"KILLED:   {len(killed_list)}")
for mid, fkm in killed_list:
    print(f"  * {mid}  (first_kill: {fkm})")
print(f"SURVIVED: {len(survived_list)}")
print(f"")
if pending:
    print(f"--- Still Pending ---")
    for mid in pending:
        print(f"  - {mid}")
