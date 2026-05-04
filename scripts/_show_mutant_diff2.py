"""显示指定变异体的代码差异 - 使用 mutated_code 字段。"""
import json, sys, difflib
sys.stdout.reconfigure(encoding='utf-8')

ROOT = r"D:\doctor_learning\Academic_Project\paper_1\MutaKernel"
f = ROOT + r"\第一次实验汇总\full_block12_results\details\L1_P1.json"
data = json.load(open(f, encoding='utf-8'))

# 获取原始 kernel 代码
kernel_code = ""
bk = json.load(open(ROOT + r"\best_kernels.json"))
kp = bk["L1_P1"]["kernel_path"]
try:
    kernel_code = open(kp, encoding='utf-8').read()
except:
    pass

IDS = ('L1_P1__relop_replace__2', 'L1_P1__relop_replace__7',
       'L1_P1__const_perturb__0', 'L1_P1__const_perturb__1')

for m in data['mutants']:
    if m['id'] in IDS:
        mc = m.get('mutated_code', '')
        print(f"\n{'='*60}")
        print(f"  {m['id']}")
        print(f"  operator: {m['operator_name']}")
        print(f"  描述: {m.get('description', '')}")
        print(f"  fields: {[k for k in m.keys()]}")
        print(f"{'='*60}")
        if kernel_code and mc:
            orig_lines = kernel_code.splitlines(keepends=True)
            mut_lines = mc.splitlines(keepends=True)
            diff = list(difflib.unified_diff(orig_lines, mut_lines,
                                             fromfile='original', tofile='mutant', n=2))
            if diff:
                for line in diff[:25]:
                    print(line, end='')
            else:
                print("  [代码完全相同]")
        elif mc:
            print(f"  mutated_code 长度: {len(mc)}")
        else:
            print(f"  [mutated_code 为空]")
