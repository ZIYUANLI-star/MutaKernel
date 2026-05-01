"""查看 LLM 对 L1_P1 relop_replace__2 和 const_perturb__0/1 的详细诊断。"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\doctor_learning\Academic_Project\paper_1\MutaKernel"
llm_dir = ROOT + r"\第一次实验汇总\llm_analysis_results\details"

for mid in ['L1_P1__relop_replace__2', 'L1_P1__const_perturb__0', 'L1_P1__const_perturb__1']:
    try:
        data = json.load(open(f"{llm_dir}/{mid}.json", encoding='utf-8'))
    except:
        print(f"\n{mid}: 文件不存在")
        continue
    print(f"\n{'='*60}")
    print(f"  {mid}")
    print(f"  killable: {data.get('killable')}")
    print(f"  killed_by_llm: {data.get('killed_by_llm')}")
    print(f"  survival_reason: {data.get('survival_reason', '')[:300]}")
    rounds = data.get('rounds', [])
    for r in rounds:
        print(f"  Round {r['round']}: killable={r.get('killable')}, killed={r.get('killed')}")
        print(f"    strategy: {r.get('kill_strategy', '')[:200]}")
