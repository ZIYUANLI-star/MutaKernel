"""验证 5 个快速测试变异体的等价判断是否合理。
交叉比对 full_block12_results + llm_analysis_results + stress_enhance_results。
"""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\doctor_learning\Academic_Project\paper_1\MutaKernel"
FIRST = os.path.join(ROOT, "第一次实验汇总")

TARGET_IDS = [
    "L1_P1__relop_replace__2",
    "L1_P1__relop_replace__7",
    "L1_P1__const_perturb__0",
    "L1_P1__const_perturb__1",
    "L1_P100__arith_replace__7",
]

print("=" * 70)
print("  交叉验证 5 个变异体的等价判断")
print("=" * 70)

# 1. full_block12 details
details_dir = os.path.join(FIRST, "full_block12_results", "details")
for f in sorted(os.listdir(details_dir)):
    if not f.endswith(".json"):
        continue
    data = json.load(open(os.path.join(details_dir, f), encoding="utf-8"))
    for m in data.get("mutants", []):
        if m["id"] in TARGET_IDS:
            mc = m.get("mutated_code", "")
            diff_lines = []
            if mc:
                oc = m.get("original_code", "")
                diff_lines = [l for l in mc.split('\n') if l.strip() and l not in (oc or "")]
            print(f"\n[Phase0 初始测试] {m['id']}")
            print(f"  status: {m['status']}")
            print(f"  operator: {m['operator_name']} ({m['operator_category']})")
            print(f"  描述: {m.get('description', 'N/A')}")

# 2. stress_enhance_results
stress_dir = os.path.join(FIRST, "stress_enhance_results", "details")
if os.path.isdir(stress_dir):
    for f in sorted(os.listdir(stress_dir)):
        if not f.endswith(".json"):
            continue
        data = json.load(open(os.path.join(stress_dir, f), encoding="utf-8"))
        mid = data.get("mutant_id", "")
        if mid in TARGET_IDS:
            print(f"\n[Phase1 增强测试] {mid}")
            print(f"  overall_killed: {data.get('overall_killed', 'N/A')}")
            pr = data.get("policy_results", [])
            if isinstance(pr, list):
                killed_policies = [p.get("policy", "?") for p in pr if p.get("killed")]
                print(f"  killed by policies: {killed_policies if killed_policies else 'NONE'}")
            print(f"  dtype_killed: {data.get('dtype_killed', 'N/A')}")
            print(f"  repeated_run_killed: {data.get('repeated_run_killed', 'N/A')}")

# 3. llm_analysis_results
llm_dir = os.path.join(FIRST, "llm_analysis_results", "details")
if os.path.isdir(llm_dir):
    for f in sorted(os.listdir(llm_dir)):
        if not f.endswith(".json"):
            continue
        data = json.load(open(os.path.join(llm_dir, f), encoding="utf-8"))
        mid = data.get("mutant_id", "")
        if mid in TARGET_IDS:
            print(f"\n[Phase2 LLM分析] {mid}")
            print(f"  killable: {data.get('killable', 'N/A')}")
            print(f"  killed_by_llm: {data.get('killed_by_llm', 'N/A')}")
            diag = data.get("diagnosis", {})
            if isinstance(diag, dict):
                print(f"  LLM判断: {diag.get('equivalence_assessment', 'N/A')[:200]}")
                print(f"  LLM理由: {diag.get('reasoning', 'N/A')[:300]}")
            elif isinstance(diag, str):
                print(f"  LLM诊断: {diag[:300]}")
            # Check iterations
            iters = data.get("iterations", [])
            if iters:
                last = iters[-1] if iters else {}
                print(f"  迭代次数: {len(iters)}")
                if isinstance(last, dict):
                    print(f"  最终判断: killable={last.get('killable', 'N/A')}")

print("\n" + "=" * 70)
print("  结论")
print("=" * 70)
print("""
新等价检测判断:
  L1_P1__relop_replace__2:  EQUIVALENT
  L1_P1__relop_replace__7:  EQUIVALENT
  L1_P1__const_perturb__0:  EQUIVALENT
  L1_P1__const_perturb__1:  TIMEOUT (60s)
  L1_P100__arith_replace__7: TIMEOUT (60s)
""")
