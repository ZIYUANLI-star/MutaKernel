"""显示变异体代码差异 — 直接对比 mutated_code 之间的差异行。"""
import json, sys, difflib
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\doctor_learning\Academic_Project\paper_1\MutaKernel"
f = ROOT + r"\第一次实验汇总\full_block12_results\details\L1_P1.json"
data = json.load(open(f, encoding='utf-8'))

IDS = ('L1_P1__relop_replace__2', 'L1_P1__relop_replace__7',
       'L1_P1__const_perturb__0', 'L1_P1__const_perturb__1')

mutants = {m['id']: m for m in data['mutants'] if m['id'] in IDS}

# 使用第一个变异体和另一个的差异来推断原始代码 vs 变异代码
# 更好的方案：用任意两个不同变异体的 mutated_code 对比找出各自的变异点
# 但最简单的办法：直接展示每个变异体 mutated_code 的特征行

# 取 relop_replace__2 的代码作为参考（它的变异在 L26）
ref_id = 'L1_P1__relop_replace__2'
ref_code = mutants[ref_id]['mutated_code']
ref_lines = ref_code.splitlines()

for mid in IDS:
    m = mutants[mid]
    mc = m['mutated_code']
    mc_lines = mc.splitlines()
    print(f"\n{'='*60}")
    print(f"  {mid}")
    print(f"  描述: {m.get('description', '')}")
    print(f"{'='*60}")
    
    if mid == ref_id:
        # 显示 L26 附近
        for i in range(max(0, 24), min(len(mc_lines), 29)):
            print(f"  L{i+1}: {mc_lines[i]}")
    else:
        # 和 ref 对比差异
        diff = list(difflib.unified_diff(ref_lines, mc_lines, n=1, lineterm=''))
        if diff:
            for line in diff:
                if line.startswith(('---', '+++', '@@', '-', '+')):
                    print(f"  {line}")
        else:
            print("  [与 relop_replace__2 完全相同]")
