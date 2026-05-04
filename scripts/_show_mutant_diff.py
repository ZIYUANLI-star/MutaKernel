"""显示指定变异体的代码差异。"""
import json, sys, difflib
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\doctor_learning\Academic_Project\paper_1\MutaKernel"
f = ROOT + r"\第一次实验汇总\full_block12_results\details\L1_P1.json"
data = json.load(open(f, encoding='utf-8'))

for m in data['mutants']:
    if m['id'] in ('L1_P1__relop_replace__2', 'L1_P1__relop_replace__7',
                    'L1_P1__const_perturb__0', 'L1_P1__const_perturb__1'):
        oc = m.get('original_code', '')
        mc = m.get('mutated_code', '')
        print(f"\n{'='*60}")
        print(f"  {m['id']}")
        print(f"  operator: {m['operator_name']}")
        print(f"  描述: {m.get('description', '')}")
        print(f"{'='*60}")
        if oc and mc:
            orig_lines = oc.splitlines(keepends=True)
            mut_lines = mc.splitlines(keepends=True)
            diff = difflib.unified_diff(orig_lines, mut_lines,
                                        fromfile='original', tofile='mutant',
                                        lineterm='')
            diff_text = ''.join(diff)
            if diff_text:
                for line in diff_text.split('\n')[:30]:
                    print(line)
            else:
                print("  [无差异]")
        else:
            print("  [代码缺失]")
