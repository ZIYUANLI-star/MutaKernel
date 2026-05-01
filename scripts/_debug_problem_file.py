"""Debug why resolve_problem_file returns None."""
import json
import glob
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
bk = json.load(open(PROJECT / "best_kernels.json"))

test_kernels = ["L1_P1", "L1_P23", "L1_P100", "L2_P41", "L1_P39"]
for key in test_kernels:
    info = bk.get(key, {})
    pid = info.get("problem_id", "MISSING")
    level = info.get("level", "MISSING")
    
    # Simulate resolve_problem_file logic
    if not pid or pid == "MISSING":
        parts = key.split("_P")
        if len(parts) == 2:
            pid = parts[1]
    
    kb_root = "/home/kbuser/projects/KernelBench-0"
    level_str = str(level)
    level_num = level_str[1:] if level_str.startswith("L") else level_str
    problem_dir = f"{kb_root}/KernelBench/level{level_num}"
    pattern = f"{problem_dir}/{pid}_*.py"
    matches = glob.glob(pattern)
    
    print(f"{key}: level={level}, pid={pid}")
    print(f"  Pattern: {pattern}")
    print(f"  Matches: {matches}")
    print(f"  best_kernels keys: {list(info.keys())}")
    print()
