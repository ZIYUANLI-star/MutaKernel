import json
d = json.load(open("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第四次实验汇总/CUDA-Agent实验补充/results/checkpoint.json"))
e = d["cuda_agent__L2_T53"]
print("fixed_round:", e["fixed_round"])
print("rounds keys:", list(e["rounds"].keys()))
for k in e["rounds"]:
    src = e["rounds"][k].get("kernel_source", "")
    print(f"--- round {k}: kernel_source has cuda_extension.fused_all_forward: {'cuda_extension.fused_all_forward' in src} ---")
    print(f"    has __global__: {'__global__' in src}")
    print(f"    forward last 300 chars:")
    idx = src.find("def forward")
    if idx >= 0:
        print(src[idx:idx+500])
