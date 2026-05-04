"""Generate LLM analysis reports:
  File 1: mutants killed by LLM   (full code + killing input + diagnosis process)
  File 2: mutants survived LLM    (full code + tried inputs + diagnosis process)
"""
import json, re, glob
from pathlib import Path

PROJECT = Path("/mnt/d/doctor_learning/Academic_Project/paper_1/MutaKernel")
LLM_DIR = PROJECT / "llm_analysis_results" / "details"
BLOCK12_DIR = PROJECT / "full_block12_results" / "details"
BEST_KERNELS_FILE = PROJECT / "best_kernels.json"
KB_ROOT = "/home/kbuser/projects/KernelBench-0/KernelBench"
OUT_DIR = PROJECT / "llm_analysis_results"

best_kernels = json.loads(BEST_KERNELS_FILE.read_text(encoding="utf-8"))


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


# ─── Load mutated code from block12 ──────────────────────────────────────
print("Loading mutated code from block12...", flush=True)
mutant_code_map = {}
mutant_meta_map = {}
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

# ─── Load LLM analysis results ──────────────────────────────────────────
print("Loading LLM analysis results...", flush=True)
killed_list = []
survived_list = []
for lf in sorted(LLM_DIR.glob("*.json")):
    try:
        ld = json.loads(lf.read_text(encoding="utf-8"))
    except Exception:
        continue
    if ld.get("killed_by_llm"):
        killed_list.append(ld)
    else:
        survived_list.append(ld)

print(f"Killed by LLM: {len(killed_list)}, Survived LLM: {len(survived_list)}", flush=True)

# ─── Cache per-kernel data ───────────────────────────────────────────────
all_kernels = set()
for ld in killed_list + survived_list:
    all_kernels.add(ld["kernel_name"])

print(f"Caching data for {len(all_kernels)} kernels...", flush=True)
kernel_original = {}
kernel_gi_src = {}
for kn in sorted(all_kernels):
    kernel_original[kn] = load_original_code(kn)
    pf = resolve_problem_file(kn)
    if pf:
        try:
            kernel_gi_src[kn] = extract_get_inputs(Path(pf).read_text(encoding="utf-8"))
        except Exception:
            kernel_gi_src[kn] = None
    else:
        kernel_gi_src[kn] = None


def format_round(r, is_last=False):
    """Format one LLM diagnosis round."""
    lines = []
    round_num = r.get("round", "?")
    killed = r.get("killed", False)
    killable = r.get("killable", None)

    if killed:
        status_icon = "**KILLED**"
    elif killable is False:
        status_icon = "判定不可杀"
    else:
        status_icon = "未杀死"

    lines.append(f"#### Round {round_num} ({status_icon})")
    lines.append("")

    lines.append(f"**LLM 存活原因分析：**")
    lines.append("")
    lines.append(f"> {r.get('survival_reason', '(无)')}")
    lines.append("")

    lines.append(f"**LLM 杀死策略：**")
    lines.append("")
    lines.append(f"> {r.get('kill_strategy', '(无)')}")
    lines.append("")

    code = r.get("suggested_code")
    if code:
        lines.append(f"**LLM 建议的输入代码：**")
        lines.append("")
        lines.append("```python")
        lines.append(code)
        lines.append("```")
        lines.append("")

    detail = r.get("detail", {})
    if detail:
        ref_ok = detail.get("ref_ok", "")
        orig_ok = detail.get("original_ok", "")
        mut_ok = detail.get("mutant_ok", "")
        diff = detail.get("diff_summary", "")
        error = detail.get("error", "")
        lines.append(f"**GPU 验证结果：**")
        lines.append("")
        lines.append(f"- ref_ok: `{ref_ok}`, original_ok: `{orig_ok}`, mutant_ok: `{mut_ok}`")
        if diff:
            lines.append(f"- diff: `{diff}`")
        if error:
            lines.append(f"- error: `{error}`")
        lines.append("")

    return "\n".join(lines)


def write_report(entries, output_path, title, is_killed):
    out = []
    out.append(f"# {title}")
    out.append("")
    out.append(f"**总计**: {len(entries)} 个变异体")
    out.append(f"**模型**: DeepSeek-R1 (deepseek-reasoner)")
    out.append(f"**最大迭代轮数**: 3")
    out.append("")
    out.append("---")

    for idx, ld in enumerate(entries, 1):
        mid = ld["mutant_id"]
        kn = ld["kernel_name"]
        meta = mutant_meta_map.get(mid, {})
        op = meta.get("operator_name", ld.get("operator_name", ""))
        cat = meta.get("operator_category", "")
        desc = meta.get("description", "")
        site = meta.get("site", {})
        line_start = site.get("line_start", 0)
        orig_fragment = site.get("original_code", "")
        node_type = site.get("node_type", "")
        killing_round = ld.get("killing_round", 0)
        cluster_label = ld.get("cluster_label", "")

        orig_code = kernel_original.get(kn, "")
        mut_code = mutant_code_map.get(mid, "")

        out.append("")
        out.append(f"## {idx}. `{mid}`")
        out.append("")
        out.append(f"- **Kernel**: `{kn}`")
        out.append(f"- **算子**: `{op}` (Category {cat})")
        out.append(f"- **描述**: {desc}")
        out.append(f"- **变异行**: Line {line_start}, 原始片段 `{orig_fragment}`, 节点类型 `{node_type}`")
        if is_killed:
            out.append(f"- **杀死轮次**: Round {killing_round}")
        if cluster_label:
            out.append(f"- **聚类标签**: {cluster_label}")
        out.append("")

        # Part 1: original code
        out.append("### Part 1: 未变异的完整源代码")
        out.append("")
        if orig_code:
            out.append("```python")
            out.append(orig_code)
            out.append("```")
        else:
            out.append("*(原始 kernel 文件不可读)*")
        out.append("")

        # Part 2: mutated code
        out.append("### Part 2: 变异后的完整源代码")
        out.append("")
        if mut_code:
            out.append("```python")
            out.append(mut_code)
            out.append("```")
        else:
            out.append("*(无变异代码数据)*")
        out.append("")

        # Part 3: inputs
        if is_killed:
            out.append("### Part 3: 杀死该变异体的输入")
            out.append("")
            # Find the killing round's code
            rounds = ld.get("rounds", [])
            for r in rounds:
                if r.get("killed"):
                    code = r.get("suggested_code", "")
                    if code:
                        out.append(f"**杀死输入代码 (Round {r.get('round', '?')})：**")
                        out.append("")
                        out.append("```python")
                        out.append(code)
                        out.append("```")
                        out.append("")
                    detail = r.get("detail", {})
                    if detail:
                        diff = detail.get("diff_summary", "")
                        if diff:
                            out.append(f"**输出差异**: `{diff}`")
                            out.append("")
                    break
        else:
            out.append("### Part 3: 未杀死该变异体的输入")
            out.append("")
            gi_src = kernel_gi_src.get(kn)
            if gi_src:
                out.append("**基础 get_inputs() 代码：**")
                out.append("")
                out.append("```python")
                out.append(gi_src)
                out.append("```")
                out.append("")
            # Show all rounds' suggested codes
            rounds = ld.get("rounds", [])
            tried_codes = [r for r in rounds if r.get("suggested_code")]
            if tried_codes:
                out.append(f"**LLM 尝试的 {len(tried_codes)} 个输入均未杀死：**")
                out.append("")
                for r in tried_codes:
                    out.append(f"Round {r.get('round', '?')} 建议的输入：")
                    out.append("")
                    out.append("```python")
                    out.append(r["suggested_code"])
                    out.append("```")
                    out.append("")
            # Final verdict
            out.append(f"**LLM 最终存活原因**: {ld.get('survival_reason', '(无)')}")
            out.append("")
            out.append(f"**LLM 最终判定**: {'不可杀 (equivalent/unkillable)' if not ld.get('killable') else '可杀但未成功'}")
            out.append("")

        # Part 4: LLM diagnosis process
        out.append("### Part 4: 大模型诊断流程")
        out.append("")
        rounds = ld.get("rounds", [])
        if rounds:
            out.append(f"共 {len(rounds)} 轮迭代分析：")
            out.append("")
            for ri, r in enumerate(rounds):
                out.append(format_round(r, is_last=(ri == len(rounds) - 1)))
        else:
            out.append("*(无诊断轮次数据)*")

        # Robustness suggestion (for survived)
        if not is_killed:
            rob = ld.get("robustness_suggestion", "")
            if rob:
                out.append("#### 鲁棒性增强建议")
                out.append("")
                out.append(f"> {rob}")
                out.append("")

        # Test construction rule (for killed)
        if is_killed:
            rule = ld.get("test_construction_rule")
            if rule:
                out.append("#### 测试构造规则")
                out.append("")
                out.append(f"- **规则名**: `{rule.get('rule_name', '')}`")
                out.append(f"- **描述**: {rule.get('rule_description', '')}")
                out.append(f"- **适用算子**: {', '.join(rule.get('applicable_operators', []))}")
                if rule.get("policy_code"):
                    out.append("")
                    out.append("```python")
                    out.append(rule["policy_code"])
                    out.append("```")
                out.append("")

        out.append("")
        out.append("---")

    output_path.write_text("\n".join(out), encoding="utf-8")
    print(f"Report: {output_path} ({len(out)} lines)")


# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating killed report...", flush=True)
write_report(
    killed_list,
    OUT_DIR / "llm_killed_report.md",
    "LLM 分析后新杀死的变异体",
    is_killed=True,
)

print("\nGenerating survived report...", flush=True)
write_report(
    survived_list,
    OUT_DIR / "llm_survived_report.md",
    "LLM 分析后仍存活的变异体",
    is_killed=False,
)

print("\nDone!")
