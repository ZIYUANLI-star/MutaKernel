"""For each of A/B/C/D high-level class in §5.1.2, count:
- Total
- Opus 5轮 all killable=False
- Opus 任一轮 killable=True
- Actually killed by any Task (A or C)
"""
import re
import json
from pathlib import Path

REPORT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/未杀死变异体逐项分析.md")
TASKA = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details")
TASKC = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/details")

# Parse the §5.1.2 table to get { mutant_id -> class }
content = REPORT.read_text(encoding="utf-8")
pat = re.compile(r"\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*[^|]+\|\s*([ABCD])\s*\|")
mapping = {}
for m in pat.finditer(content):
    mapping[m.group(1)] = m.group(2)

# Only keep entries in the §5.1.2 table by limiting to the segment between #### 5.1.2 and #### 5.1.3
start = content.index("#### 5.1.2 逐变异体分类表")
end = content.index("#### 5.1.3")
section = content[start:end]
mapping = {}
for m in pat.finditer(section):
    mapping[m.group(1)] = m.group(2)
print(f"Parsed mapping size: {len(mapping)}")

# Stats per class
classes = {"A": [], "B": [], "C": [], "D": []}
for mid, cls in mapping.items():
    classes[cls].append(mid)

def get_taskA(mid):
    p = TASKA / f"{mid}.json"
    if not p.exists(): return None
    d = json.load(open(p))
    seq = [r.get("killable") for r in d.get("rounds", [])]
    return {
        "all_False": all(k is False for k in seq) and len(seq) > 0,
        "any_True": any(k is True for k in seq),
        "killed": d.get("killed", False),
    }

def get_taskC(mid):
    p = TASKC / f"{mid}.json"
    if not p.exists(): return None
    d = json.load(open(p))
    return d.get("killed", False)

print(f"\n{'Class':<6}{'Total':<8}{'all_False':<11}{'any_True':<10}{'taskA_kill':<12}{'taskC_kill':<12}")
print("-" * 60)
total_all = {"all_False": 0, "any_True": 0, "taskA_kill": 0, "taskC_kill": 0}
for cls in "ABCD":
    al = at = ak = ck = 0
    for mid in classes[cls]:
        a = get_taskA(mid)
        c = get_taskC(mid)
        if a:
            if a["all_False"]: al += 1
            if a["any_True"]: at += 1
            if a["killed"]: ak += 1
        if c: ck += 1
    print(f"{cls:<6}{len(classes[cls]):<8}{al:<11}{at:<10}{ak:<12}{ck:<12}")
    total_all["all_False"] += al
    total_all["any_True"] += at
    total_all["taskA_kill"] += ak
    total_all["taskC_kill"] += ck

print("-" * 60)
total_count = sum(len(classes[c]) for c in "ABCD")
print(f"{'Total':<6}{total_count:<8}{total_all['all_False']:<11}{total_all['any_True']:<10}{total_all['taskA_kill']:<12}{total_all['taskC_kill']:<12}")
