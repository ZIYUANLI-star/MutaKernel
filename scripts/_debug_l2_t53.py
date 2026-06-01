"""Debug the regex matching for L2_T53."""
import re
import json
CKPT = json.load(open("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第四次实验汇总/CUDA-Agent实验补充/results/checkpoint.json"))
entry = CKPT["cuda_agent__L2_T53"]
fr = entry["fixed_round"]
src = entry["rounds"][str(fr)]["kernel_source"]
m = re.search(r"def forward\(self[^)]*\).*?(?=\n    def |\nclass |\Z)", src, re.DOTALL)
if m:
    forward = m.group()
    print(f"Forward body length: {len(forward)}")
    print("=== forward body (first 800 chars) ===")
    print(forward[:800])
    print("...")
    print(f"\nContains 'fused_all_forward': {'fused_all_forward' in forward}")
