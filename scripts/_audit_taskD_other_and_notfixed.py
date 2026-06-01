"""Spot-check OTHER kernels and look at NOT_FIXED kernels in detail."""
import json
import re
from pathlib import Path

CKPT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第四次实验汇总/CUDA-Agent实验补充/results/checkpoint.json")
data = json.load(open(CKPT))

OTHER = [
    "cuda_agent__L1_T8", "cuda_agent__L1_T22", "cuda_agent__L1_T33", "cuda_agent__L1_T24",
    "cuda_agent__L1_T40", "cuda_agent__L1_T56", "cuda_agent__L1_T64", "cuda_agent__L1_T66",
    "cuda_agent__L1_T59", "cuda_agent__L1_T77", "cuda_agent__L1_T81", "cuda_agent__L1_T89",
    "cuda_agent__L2_T6", "cuda_agent__L2_T31", "cuda_agent__L2_T26", "cuda_agent__L2_T54",
    "cuda_agent__L2_T63", "cuda_agent__L2_T73", "cuda_agent__L2_T83",
    "cuda_agent__L3_T18", "cuda_agent__L3_T11", "cuda_agent__L3_T20", "cuda_agent__L3_T25",
    "cuda_agent__L3_T23", "cuda_agent__L3_T30", "cuda_agent__L3_T31",
    "cuda_agent__L1_T52", "cuda_agent__L1_T88",
]

print("=== OTHER kernels (no recognized pattern) — first 8 lines of source ===\n")
for k in OTHER[:10]:
    entry = data.get(k, {})
    fr = entry.get("fixed_round")
    r = entry.get("rounds", {}).get(str(fr)) or entry.get("rounds", {}).get(fr) or {}
    src = r.get("kernel_source", "")
    # show class definition + forward method head
    print(f"--- {k} (fixed_round={fr}, total chars={len(src)}) ---")
    lines = src.split("\n")
    head = "\n".join(lines[:30])
    print(head)
    print()

print("\n=== NOT_FIXED kernels: status and final round info ===\n")
for kid, entry in data.items():
    if entry.get("status") not in ("NOT_FIXED", "TEST_TIMEOUT"):
        continue
    rounds = entry.get("rounds", {})
    last_round = max(int(k) for k in rounds.keys()) if rounds else 0
    r = rounds.get(str(last_round), {})
    tr = r.get("test_result", {})
    print(f"{kid}: status={entry.get('status')} rounds={last_round}")
    for dim, d in tr.items():
        if not isinstance(d, dict): continue
        n_cases = len(d.get("test_cases", []) or [])
        if n_cases == 0: continue
        n_disc = d.get("discrepancies") or 0
        n_pass = d.get("passes") or 0
        print(f"  {dim:<20}: cases={n_cases}, discrepancies={n_disc}, passes={n_pass}")
