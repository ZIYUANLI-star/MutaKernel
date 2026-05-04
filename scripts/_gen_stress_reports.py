"""Generate stress-enhance reports:
  Folder 1 (killed_report):   mutants killed by 4-layer stress test
  Folder 2 (survived_report): mutants that survived 4-layer stress test
Each contains: full original code + full mutated code + concrete input info.
"""
import json, re, glob, os
from pathlib import Path
from collections import defaultdict

PROJECT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
STRESS_DIR = PROJECT / "stress_enhance_results" / "details"
BLOCK12_DIR = PROJECT / "full_block12_results" / "details"
BEST_KERNELS_FILE = PROJECT / "best_kernels.json"
KB_ROOT = "/home/kbuser/projects/KernelBench-0/KernelBench"

OUT_KILLED_DIR = PROJECT / "stress_enhance_results" / "killed_report"
OUT_SURVIVED_DIR = PROJECT / "stress_enhance_results" / "survived_report"
OUT_KILLED_DIR.mkdir(parents=True, exist_ok=True)
OUT_SURVIVED_DIR.mkdir(parents=True, exist_ok=True)

best_kernels = json.loads(BEST_KERNELS_FILE.read_text(encoding="utf-8"))

# ─── helpers ───────────────────────────────────────────────────────────────
def resolve_problem_file(kernel_name):
    bk = best_kernels.get(kernel_name, {})
    pid = bk.get("problem_id", "")
    level_raw = str(bk.get("level", 1))
    if not pid:
        parts = kernel_name.split("_P")
        if len(parts) == 2:
            pid = parts[1]
    level_num = level_raw[1:] if level_raw.startswith("L") else level_raw
    matches = glob.glob(f"{KB_ROOT}/level{level_num}/{pid}_*.py")
    return matches[0] if matches else None


def load_original_code(kernel_name):
    bk = best_kernels.get(kernel_name, {})
    kpath = bk.get("kernel_path", "")
    if kpath:
        p = Path(kpath)
        if p.exists():
            return p.read_text(encoding="utf-8")
    return None


def extract_get_inputs(code):
    if not code:
        return None
    lines = code.split("\n")
    gi_start = gi_end = None
    for i, line in enumerate(lines):
        if re.match(r'^def get_inputs', line):
            gi_start = i
            continue
        if gi_start is not None and gi_end is None:
            if line.strip() == "":
                continue
            if line and not line[0].isspace():
                gi_end = i
                break
    if gi_start is None:
        return None
    if gi_end is None:
        gi_end = len(lines)
    gi_body = lines[gi_start:gi_end]
    gi_text = "\n".join(gi_body)
    global_defs = []
    for line in lines[:gi_start]:
        stripped = line.strip()
        m = re.match(r'^([A-Za-z_]\w*)\s*=\s*(.+)', stripped)
        if m and m.group(1) in gi_text:
            global_defs.append(stripped)
    result = []
    if global_defs:
        result.extend(global_defs)
        result.append("")
    result.extend(gi_body)
    return "\n".join(result).rstrip()


def get_input_specs(kernel_name):
    """Run get_inputs() with seeds to get concrete tensor specs."""
    pf = resolve_problem_file(kernel_name)
    if not pf:
        return None
    try:
        code = Path(pf).read_text(encoding="utf-8")
    except Exception:
        return None
    lines = code.split("\n")
    gi_start = gi_end = None
    for i, line in enumerate(lines):
        if re.match(r'^def get_inputs', line):
            gi_start = i
            continue
        if gi_start is not None and gi_end is None:
            if line.strip() == "":
                continue
            if line and not line[0].isspace():
                gi_end = i
                break
    if gi_start is None:
        return None
    if gi_end is None:
        gi_end = len(lines)

    import torch
    namespace = {"torch": torch}
    for line in lines[:gi_start]:
        stripped = line.strip()
        m_var = re.match(r'^([A-Za-z_]\w*)\s*=\s*(.+)', stripped)
        if m_var:
            try:
                exec(stripped, namespace)
            except Exception:
                pass
    try:
        exec("\n".join(lines[gi_start:gi_end]), namespace)
    except Exception:
        return None
    fn = namespace.get("get_inputs")
    if not fn:
        return None

    seed_specs = []
    for seed in [42, 43, 44]:
        torch.manual_seed(seed)
        try:
            inputs = fn()
            parts = []
            for ti, t in enumerate(inputs):
                if isinstance(t, torch.Tensor):
                    parts.append(
                        f"  - inputs[{ti}]: shape={list(t.shape)}, "
                        f"dtype={t.dtype}, range=[{t.min().item():.4g}, {t.max().item():.4g}]"
                    )
                else:
                    parts.append(f"  - inputs[{ti}]: {type(t).__name__} = {t}")
            seed_specs.append(f"seed={seed}:\n" + "\n".join(parts))
        except Exception as e:
            seed_specs.append(f"seed={seed}: ERROR {e}")
    return "\n".join(seed_specs)


POLICY_DESC = {
    "near_zero": "values ~1e-7",
    "large_magnitude": "values ~1e6",
    "near_overflow": "values ~1e38",
    "denormals": "values ~1e-40 (subnormal)",
    "all_negative": "all values < 0",
    "all_positive": "all values > 0",
    "mixed_extremes": "mix of 1e6 and 1e-6",
    "alternating_sign": "[+big, -big, +big, ...]",
    "sparse": "90% zeros, 10% random",
    "uniform_constant": "all elements = same constant",
    "structured_ramp": "linearly increasing 0→N",
    "boundary_last_element": "last element set to extreme",
    "head_heavy": "first 10% large, rest small",
    "tail_heavy": "last 10% large, rest small",
}


def format_policy_results(policy_results):
    """Format Layer 1 stress test results into readable table."""
    if not policy_results:
        return "(无详细策略结果)"
    by_policy = defaultdict(list)
    for pr in policy_results:
        by_policy[pr["policy"]].append(pr)

    lines = ["| 策略 | 种子数 | 结果 | 描述 |",
             "|------|--------|------|------|"]
    for policy, results in by_policy.items():
        n = len(results)
        ref_ok_all = all(r.get("ref_ok", False) for r in results)
        orig_ok_all = all(r.get("original_ok", False) for r in results)
        mut_ok_all = all(r.get("mutant_ok", True) for r in results)
        if not ref_ok_all:
            status = "REF_FAIL (参考实现崩溃)"
        elif not orig_ok_all:
            status = "ORIG_ALSO_FAILS (原始也失败)"
        elif mut_ok_all:
            status = "both_ok (未区分)"
        else:
            status = "**KILLED** (杀死变异体)"
        desc = POLICY_DESC.get(policy, policy)
        lines.append(f"| {policy} | ×{n} | {status} | {desc} |")
    return "\n".join(lines)


# ─── Load mutated_code from block12 results ─────────────────────────────
print("Loading mutated code from block12 results...", flush=True)
mutant_code_map = {}  # mutant_id -> mutated_code
mutant_meta_map = {}  # mutant_id -> {operator_name, description, site}
for jf in sorted(BLOCK12_DIR.glob("*.json")):
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except Exception:
        continue
    for m in data.get("mutants", []):
        mid = m["id"]
        mutant_code_map[mid] = m.get("mutated_code", "")
        mutant_meta_map[mid] = {
            "operator_name": m.get("operator_name", ""),
            "operator_category": m.get("operator_category", ""),
            "description": m.get("description", ""),
            "site": m.get("site", {}),
        }

# ─── Load all stress results ─────────────────────────────────────────────
print("Loading stress results...", flush=True)
killed_list = []
survived_list = []
for sf in sorted(STRESS_DIR.glob("*.json")):
    try:
        sd = json.loads(sf.read_text(encoding="utf-8"))
    except Exception:
        continue
    if sd.get("killed"):
        killed_list.append(sd)
    else:
        survived_list.append(sd)

print(f"Killed by stress: {len(killed_list)}, Survived stress: {len(survived_list)}", flush=True)

# ─── Cache per-kernel data ───────────────────────────────────────────────
all_kernels = set()
for sd in killed_list + survived_list:
    all_kernels.add(sd["kernel_name"])

print(f"Caching original code & inputs for {len(all_kernels)} kernels...", flush=True)
kernel_original = {}
kernel_gi_src = {}
kernel_gi_specs = {}
for ki, kn in enumerate(sorted(all_kernels)):
    print(f"  [{ki+1}/{len(all_kernels)}] {kn}", flush=True)
    kernel_original[kn] = load_original_code(kn)
    pf = resolve_problem_file(kn)
    if pf:
        try:
            pcode = Path(pf).read_text(encoding="utf-8")
            kernel_gi_src[kn] = extract_get_inputs(pcode)
        except Exception:
            kernel_gi_src[kn] = None
    else:
        kernel_gi_src[kn] = None
    kernel_gi_specs[kn] = get_input_specs(kn)


# ═══════════════════════════════════════════════════════════════════════════
# Report 1: KILLED by stress test
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating killed report...", flush=True)
out = []
out.append("# 增强测试后被杀死的变异体")
out.append("")
out.append(f"**总计**: {len(killed_list)} 个变异体在 4 层增强压力测试中被杀死")
out.append("")
out.append("## 增强测试配置")
out.append("")
out.append("| 层 | 内容 | 策略 |")
out.append("|-----|------|------|")
out.append("| Layer 1 | 值分布压力 | 14 种策略 × 3 种子 = 42 次测试 |")
out.append("| Layer 2 | dtype 切换 | float16 + bfloat16 × 3 种子 |")
out.append("| Layer 3 | 重复执行 | 10 次重复 × 3 种子，检测不一致 |")
out.append("| Layer 4 | 训练模式 | .train() 模式运行 |")
out.append("")
out.append("---")

for idx, sd in enumerate(killed_list, 1):
    mid = sd["mutant_id"]
    kn = sd["kernel_name"]
    meta = mutant_meta_map.get(mid, {})
    op = meta.get("operator_name", sd.get("operator_name", ""))
    cat = meta.get("operator_category", sd.get("operator_category", ""))
    desc = meta.get("description", "")
    site = meta.get("site", {})
    line_start = site.get("line_start", 0)
    orig_fragment = site.get("original_code", "")
    node_type = site.get("node_type", "")

    killing_layer = sd.get("killing_layer", "")
    killing_policy = sd.get("killing_policy", "")
    killing_seed = sd.get("killing_seed", "")
    killing_mode = sd.get("killing_mode", "")

    orig_code = kernel_original.get(kn, "")
    mut_code = mutant_code_map.get(mid, "")

    out.append("")
    out.append(f"## {idx}. `{mid}`")
    out.append("")
    out.append(f"- **算子**: `{op}` (Category {cat})")
    out.append(f"- **描述**: {desc}")
    out.append(f"- **变异行**: Line {line_start}, 原始片段 `{orig_fragment}`, 节点类型 `{node_type}`")
    out.append(f"- **杀死层**: `{killing_layer}`")
    out.append(f"- **杀死策略**: `{killing_policy}`")
    out.append(f"- **杀死种子**: `{killing_seed}`")
    out.append(f"- **杀死模式**: `{killing_mode}`")
    out.append("")

    # Part 1
    out.append("### Part 1: 未变异的完整源代码")
    out.append("")
    if orig_code:
        out.append("```python")
        out.append(orig_code)
        out.append("```")
    else:
        out.append("*(原始 kernel 文件不可读)*")
    out.append("")

    # Part 2
    out.append("### Part 2: 变异后的完整源代码")
    out.append("")
    if mut_code:
        out.append("```python")
        out.append(mut_code)
        out.append("```")
    else:
        out.append("*(无变异代码数据)*")
    out.append("")

    # Part 3: killing input
    out.append("### Part 3: 杀死该变异体的输入")
    out.append("")
    out.append(f"**杀死方式**: 层={killing_layer}, 策略={killing_policy}, 种子={killing_seed}, 模式={killing_mode}")
    out.append("")
    if killing_policy:
        pdesc = POLICY_DESC.get(killing_policy, killing_policy)
        out.append(f"**杀死策略描述**: {pdesc}")
        out.append("")
    gi_src = kernel_gi_src.get(kn)
    if gi_src:
        out.append("**基础输入生成代码** (压力策略在此基础上修改值分布)：")
        out.append("")
        out.append("```python")
        out.append(gi_src)
        out.append("```")
    out.append("")
    out.append("---")

killed_file = OUT_KILLED_DIR / "killed_mutants_report.md"
killed_file.write_text("\n".join(out), encoding="utf-8")
print(f"Killed report: {killed_file} ({len(out)} lines)")

# ═══════════════════════════════════════════════════════════════════════════
# Report 2: SURVIVED stress test
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating survived report...", flush=True)
out2 = []
out2.append("# 增强测试后仍存活的变异体")
out2.append("")
out2.append(f"**总计**: {len(survived_list)} 个变异体在 4 层增强压力测试中均未被杀死")
out2.append("")
out2.append("## 增强测试配置")
out2.append("")
out2.append("| 层 | 内容 | 策略 |")
out2.append("|-----|------|------|")
out2.append("| Layer 1 | 值分布压力 | 14 种策略 × 3 种子 = 42 次测试 |")
out2.append("| Layer 2 | dtype 切换 | float16 + bfloat16 × 3 种子 |")
out2.append("| Layer 3 | 重复执行 | 10 次重复 × 3 种子，检测不一致 |")
out2.append("| Layer 4 | 训练模式 | .train() 模式运行 |")
out2.append("")
out2.append("---")

for idx, sd in enumerate(survived_list, 1):
    mid = sd["mutant_id"]
    kn = sd["kernel_name"]
    meta = mutant_meta_map.get(mid, {})
    op = meta.get("operator_name", sd.get("operator_name", ""))
    cat = meta.get("operator_category", sd.get("operator_category", ""))
    desc = meta.get("description", "")
    site = meta.get("site", {})
    line_start = site.get("line_start", 0)
    orig_fragment = site.get("original_code", "")
    node_type = site.get("node_type", "")

    orig_code = kernel_original.get(kn, "")
    mut_code = mutant_code_map.get(mid, "")

    out2.append("")
    out2.append(f"## {idx}. `{mid}`")
    out2.append("")
    out2.append(f"- **算子**: `{op}` (Category {cat})")
    out2.append(f"- **描述**: {desc}")
    out2.append(f"- **变异行**: Line {line_start}, 原始片段 `{orig_fragment}`, 节点类型 `{node_type}`")
    out2.append("")

    # Part 1
    out2.append("### Part 1: 未变异的完整源代码")
    out2.append("")
    if orig_code:
        out2.append("```python")
        out2.append(orig_code)
        out2.append("```")
    else:
        out2.append("*(原始 kernel 文件不可读)*")
    out2.append("")

    # Part 2
    out2.append("### Part 2: 变异后的完整源代码")
    out2.append("")
    if mut_code:
        out2.append("```python")
        out2.append(mut_code)
        out2.append("```")
    else:
        out2.append("*(无变异代码数据)*")
    out2.append("")

    # Part 3: all inputs tried
    out2.append("### Part 3: 未杀死该变异体的输入")
    out2.append("")
    gi_src = kernel_gi_src.get(kn)
    gi_specs = kernel_gi_specs.get(kn)
    if gi_src:
        out2.append("**基础输入生成代码** (所有压力策略在此基础上仅修改值分布，shape 不变)：")
        out2.append("")
        out2.append("```python")
        out2.append(gi_src)
        out2.append("```")
        out2.append("")
    if gi_specs:
        out2.append("**初始 3 组输入的 Tensor 规格 (seed=42/43/44)：**")
        out2.append("")
        out2.append("```")
        out2.append(gi_specs)
        out2.append("```")
        out2.append("")

    # Layer 1 detail
    policy_results = sd.get("policy_results", [])
    if policy_results:
        out2.append("**Layer 1 值分布压力测试 (14 策略 × 3 种子 = 42 次) 详细结果：**")
        out2.append("")
        out2.append(format_policy_results(policy_results))
        out2.append("")

    # Layer 2-4
    l2 = "杀死" if sd.get("dtype_killed") else "未杀死"
    l3 = "不一致" if sd.get("repeated_killed") else "一致"
    l4 = "杀死" if sd.get("training_killed") else "未杀死"
    out2.append(f"**Layer 2 (dtype 切换)**: {l2}")
    out2.append(f"**Layer 3 (重复执行)**: {l3}")
    out2.append(f"**Layer 4 (训练模式)**: {l4}")

    orig_failures = sd.get("original_failures", [])
    if orig_failures:
        out2.append(f"**原始 kernel 也失败的策略**: {', '.join(orig_failures)}")
    out2.append("")
    out2.append("---")

survived_file = OUT_SURVIVED_DIR / "survived_mutants_report.md"
survived_file.write_text("\n".join(out2), encoding="utf-8")
print(f"Survived report: {survived_file} ({len(out2)} lines)")
print("\nDone!")
