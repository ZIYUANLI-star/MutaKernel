"""Generate survived mutants report:
  Part 1: full original source code
  Part 2: full mutated source code
  Part 3: concrete inputs used (seed + get_inputs code + resulting tensor specs)
"""
import json, re, glob, sys
from pathlib import Path

PROJECT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
DETAILS_DIR = PROJECT / "full_block12_results" / "details"
BEST_KERNELS_FILE = PROJECT / "best_kernels.json"
OUTPUT_FILE = PROJECT / "full_block12_results" / "survived_mutants_report.md"
KB_ROOT = "/home/kbuser/projects/KernelBench-0/KernelBench"

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
    """Load original kernel source code."""
    bk = best_kernels.get(kernel_name, {})
    kpath = bk.get("kernel_path", "")
    if kpath:
        p = Path(kpath)
        if p.exists():
            return p.read_text(encoding="utf-8")
    return None


def get_input_description(kernel_name):
    """Extract get_inputs() and run it to get concrete tensor specs."""
    pf = resolve_problem_file(kernel_name)
    if not pf:
        return None, None

    try:
        code = Path(pf).read_text(encoding="utf-8")
    except Exception:
        return None, None

    # Extract get_inputs function + globals
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
        return None, None
    if gi_end is None:
        gi_end = len(lines)

    gi_body = lines[gi_start:gi_end]
    gi_text = "\n".join(gi_body)

    # Collect referenced globals
    global_defs = []
    for line in lines[:gi_start]:
        stripped = line.strip()
        m = re.match(r'^([A-Za-z_]\w*)\s*=\s*(.+)', stripped)
        if m and m.group(1) in gi_text:
            global_defs.append(stripped)

    gi_source = []
    if global_defs:
        gi_source.extend(global_defs)
        gi_source.append("")
    gi_source.extend(gi_body)
    gi_source_str = "\n".join(gi_source).rstrip()

    # Run it to get concrete tensor info for each seed
    import torch
    namespace = {"torch": torch}
    for gd in global_defs:
        try:
            exec(gd, namespace)
        except Exception:
            pass

    try:
        exec("\n".join(gi_body), namespace)
    except Exception:
        return gi_source_str, None

    get_inputs_fn = namespace.get("get_inputs")
    if not get_inputs_fn:
        return gi_source_str, None

    seed_specs = []
    for seed in [42, 43, 44]:
        torch.manual_seed(seed)
        try:
            inputs = get_inputs_fn()
            parts = []
            for ti, t in enumerate(inputs):
                if isinstance(t, torch.Tensor):
                    parts.append(
                        f"  - inputs[{ti}]: shape={list(t.shape)}, "
                        f"dtype={t.dtype}, "
                        f"range=[{t.min().item():.4g}, {t.max().item():.4g}]"
                    )
                else:
                    parts.append(f"  - inputs[{ti}]: {type(t).__name__} = {t}")
            seed_specs.append(f"seed={seed}:\n" + "\n".join(parts))
        except Exception as e:
            seed_specs.append(f"seed={seed}: ERROR {e}")

    return gi_source_str, "\n".join(seed_specs)


# ─── collect survived mutants ─────────────────────────────────────────────
survived = []
for jf in sorted(DETAILS_DIR.glob("*.json")):
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
    except Exception:
        continue
    kernel_name = data.get("kernel", {}).get("problem_name", jf.stem)
    for m in data.get("mutants", []):
        if m.get("status") == "survived":
            survived.append({
                "id": m["id"],
                "kernel_name": kernel_name,
                "operator_name": m.get("operator_name", ""),
                "operator_category": m.get("operator_category", ""),
                "description": m.get("description", ""),
                "site": m.get("site", {}),
                "mutated_code": m.get("mutated_code", ""),
            })

print(f"Found {len(survived)} survived mutants", flush=True)

# ─── cache per-kernel: original code + input info ─────────────────────────
kernel_original = {}
kernel_input_source = {}
kernel_input_specs = {}

all_kernels = sorted(set(s["kernel_name"] for s in survived))
for ki, kn in enumerate(all_kernels):
    print(f"  [{ki+1}/{len(all_kernels)}] {kn} ...", flush=True)
    kernel_original[kn] = load_original_code(kn)
    gi_src, gi_specs = get_input_description(kn)
    kernel_input_source[kn] = gi_src
    kernel_input_specs[kn] = gi_specs

orig_found = sum(1 for v in kernel_original.values() if v)
gi_found = sum(1 for v in kernel_input_source.values() if v)
print(f"Original code: {orig_found}/{len(all_kernels)}")
print(f"get_inputs: {gi_found}/{len(all_kernels)}")

# ─── generate report ─────────────────────────────────────────────────────
out = []
out.append("# 存活变异体完整报告")
out.append("")
out.append(f"**总计**: {len(survived)} 个存活变异体，来自 {len(all_kernels)} 个 kernel")
out.append("")
out.append("## 初始变异测试配置")
out.append("")
out.append("| 参数 | 值 |")
out.append("|------|-----|")
out.append("| 测试输入数量 | 3 组（seed = 42, 43, 44）|")
out.append("| 绝对容差 (atol) | 1e-2 |")
out.append("| 相对容差 (rtol) | 1e-2 |")
out.append("| 输入生成方式 | `torch.manual_seed(seed)` → `get_inputs()` |")
out.append("| 判定逻辑 | `torch.allclose(ref_output, mut_output, atol, rtol)` |")
out.append("| 判定标准 | 3 组输入全部通过 → survived；任一不通过 → killed |")
out.append("")
out.append("---")

for idx, s in enumerate(survived, 1):
    mid = s["id"]
    kn = s["kernel_name"]
    op = s["operator_name"]
    cat = s["operator_category"]
    desc = s["description"]
    site = s["site"]
    line_start = site.get("line_start", 0)
    orig_fragment = site.get("original_code", "")
    node_type = site.get("node_type", "")
    mut_code = s["mutated_code"]
    orig_code = kernel_original.get(kn, "")
    gi_src = kernel_input_source.get(kn)
    gi_specs = kernel_input_specs.get(kn)

    out.append("")
    out.append(f"## {idx}. `{mid}`")
    out.append("")
    out.append(f"- **算子**: `{op}` (Category {cat})")
    out.append(f"- **描述**: {desc}")
    out.append(f"- **变异行**: Line {line_start}, 原始片段 `{orig_fragment}`, 节点类型 `{node_type}`")
    out.append("")

    # ── Part 1: Full original source code ──
    out.append("### Part 1: 未变异的完整源代码")
    out.append("")
    if orig_code:
        out.append("```python")
        out.append(orig_code)
        out.append("```")
    else:
        out.append("*(原始 kernel 文件不可读)*")
    out.append("")

    # ── Part 2: Full mutated source code ──
    out.append("### Part 2: 变异后的完整源代码")
    out.append("")
    if mut_code:
        out.append("```python")
        out.append(mut_code)
        out.append("```")
    else:
        out.append("*(无变异代码数据)*")
    out.append("")

    # ── Part 3: Concrete inputs ──
    out.append("### Part 3: 未杀死该变异体的具体输入")
    out.append("")
    out.append("使用以下 `get_inputs()` 函数，分别以 seed=42, 43, 44 生成 3 组输入，均未杀死该变异体：")
    out.append("")
    if gi_src:
        out.append("**输入生成代码：**")
        out.append("")
        out.append("```python")
        out.append(gi_src)
        out.append("```")
        out.append("")
    if gi_specs:
        out.append("**3 组具体输入的 Tensor 规格：**")
        out.append("")
        out.append("```")
        out.append(gi_specs)
        out.append("```")
    elif not gi_src:
        out.append("*(未能提取输入信息)*")
    out.append("")
    out.append("---")

OUTPUT_FILE.write_text("\n".join(out), encoding="utf-8")
print(f"\nDone! {OUTPUT_FILE}")
print(f"Total: {len(survived)} mutants, {len(out)} lines")
