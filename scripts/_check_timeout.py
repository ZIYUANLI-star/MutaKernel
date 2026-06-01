import json
d = json.load(open("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel/第四次实验汇总/CUDA-Agent实验补充/results/checkpoint.json"))
for k, v in d.items():
    if v.get("status") == "TEST_TIMEOUT":
        print(k)
        rounds = v.get("rounds", {})
        print("  rounds:", list(rounds.keys()))
        for rk in rounds:
            tr = rounds[rk].get("test_result", {})
            err = tr.get("error", "no error field")
            print(f"  round {rk} error: {err}")
