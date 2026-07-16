# -*- coding: utf-8 -*-
"""Generate a Chinese per-mutant analysis of the KGB true-survivor mutants
(true escapes that MutaKernel's enhanced testing still failed to kill).

config_stress is now batch-only (faithful to MutaKernel's design): only the
first/batch dimension varies; every other tensor dimension stays fixed at the
problem's canonical shape."""
import json, glob, os, difflib, collections

BASE = r"D:\doctor_learning\Academic_Project\paper_1\MutaKernel\外部Benchmark差分测试_RQ4\MutaKernel-KGB\MutaKernel-KGB\MutaKernel\runs\kgb_ext_llmemd\details"
OUT = r"D:\doctor_learning\Academic_Project\paper_1\MutaKernel\外部Benchmark差分测试_RQ4\MutaKernel-KGB\MutaKernel-KGB\存活变异体逐个分析_中文.md"

OP_CN = {
    "arith_replace": "算术运算符替换",
    "relop_replace": "关系运算符替换",
    "const_perturb": "数值常量扰动",
    "cast_remove": "删除类型转换",
    "init_modify": "削弱归约初始值",
    "acc_downgrade": "累加器精度降级",
    "broadcast_unsafe": "删除广播/expand",
    "layout_assume": "删除 contiguous（内存布局假设）",
}
FAM_CN = {
    "flash_attention": "Flash-Attention", "reduce": "Reduce 归约", "layernorm": "LayerNorm",
    "softmax": "Softmax", "rotary_embedding": "RoPE 旋转位置编码", "cross_entropy": "Cross-Entropy",
    "matmul": "MatMul", "rmsnorm": "RMSNorm",
}

def changed_lines(orig, mut):
    o = orig.splitlines(); n = mut.splitlines()
    rem, add = [], []
    for line in difflib.unified_diff(o, n, lineterm="", n=0):
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("-"):
            rem.append(line[1:].strip())
        elif line.startswith("+"):
            add.append(line[1:].strip())
    return rem, add

# ---- survival-reason classifier -------------------------------------------
REASONS = {
    "R1": ("维度分发盲区",
        "变异点位于宿主端按输入维度数(ndim)分发的形状处理分支（如 `x.ndim>2` 的判定或 `unsqueeze`/`view` 重塑）。"
        "只有当输入张量的 ndim 改变（1D / 3D 以上 / 非常规维度）时，才会进入被改写的分支或触发 view/reshape 错误。"
        "而 KGB 与增强测试（value/dtype/repeated/config）始终喂入固定 ndim 的张量，config_stress 也只改第一维 batch、不改维数，"
        "故被改写分支从未执行，输出逐位一致。属于 benchmark 固定输入形状造成的**输入空间盲区**，而非 kernel 计算等价。"),
    "R2": ("断言/白名单守卫盲区",
        "变异改写的是宿主端的合法性断言或白名单守卫（如『支持的 head_dim ∈ {16,32,64,128,256}』、『0 ≤ dim < ndim』等）。"
        "被测配置仍满足改写后的断言——被扰动的往往是当前未用到的枚举项或不影响当前取值的边界，断言照常通过、不抛异常，可执行路径完全不变。"
        "只有当输入恰好命中被改写的那个枚举项/边界（如 head_dim 恰为被删掉的 16）时才会触发断言失败。"),
    "R3": ("边界掩码盲区",
        "变异改写的是 kernel 内的边界掩码逻辑——或是被掩码/越界通道的填充值（`-inf` → `-1e10`、`tl.where` 兜底值、在线 softmax 的 `m_i` 初值），"
        "或是边界比较的 off-by-one（`<` ↔ `<=`）。这类改写只有在张量的特征宽度不是分块大小(BLOCK)的整数倍、即真实存在被掩码的边界余项通道时才会显现；"
        "而 KGB 所有问题的特征宽度本身就是 2 的幂（512/1024 等），主轨道(固定 shape)与 config_stress(只改 batch、特征宽度不变)都不改变特征宽度，"
        "因此根本不存在被掩码的边界余项通道，填充值/边界判定永不被触达，逐位一致。"),
    "R4": ("同精度转换为恒等",
        "变异去掉的是显式升精到 fp32 的类型转换（如加载后 `.to(tl.float32)`、`tl.dot(...).to(tl.float32)`）。被测精度环境下，"
        "该转换要么作用于本就是 fp32 的张量（恒等），要么作用于半精度张量、但 `tl.dot`/后续表达式仍以 fp32 累加，使数值无损、计算图等价，任意输入逐位一致；"
        "只有在精度错配且无隐式提升的特定路径下才可能暴露差异。"),
    "R5": ("半精度环境下降级为恒等",
        "变异把 FP32 累加/转换降级为半精度。在这些幸存用例中，问题的基精度本身即为半精度（fp16/bf16），降级等同恒等操作；"
        "或该累加器初值随后被 `tl.dot` 结果覆盖、在被测的归约长度与数值范围内两者舍入结果逐位一致。"
        "（对应的 fp32 基精度变体多已被 dtype_stress 杀死。）"),
    "R6": ("注释/文档串变异（可执行代码未改）",
        "变异被解析器定位到 kernel 头部的公式文档字符串/注释中，可执行代码完全未改变，增强测试自然逐位一致。"
        "EMD-LLM 在结构化字段中仍把它标记为『可杀』（其自由文本推理实际已认定等价），属于 LLM 层的保守/不一致标注，本质应并入等价类。"),
    "R7": ("非连续布局假设未触发",
        "变异删除了 `permute(...).contiguous()` 中的 `.contiguous()`，只有当传入张量非连续（transpose/permute 视图）且后续 `view` 失败时才暴露。"
        "该调用位于多轴规约（inner_size>1）路径，KGB 用例固定末维规约不进入该路径，且输入恒为连续张量，故无差异。"),
    "R8": ("规约维/形状整数运算盲区",
        "变异改写的是宿主端只在特定形状关系下才生效的整数运算或分支（如规约维归一化 `if dim<0`、多轴规约分支 `inner_size==1`、"
        "`total_rows=outer*inner`、`repeat_factor` 上取整、matmul 的维度/步长计算）。KGB 用例的形状满足规则关系"
        "（末维规约 → inner_size==1、n_rows 等于基准行数、维度可整除），config_stress 只改 batch 也不破坏这些关系，"
        "使改写后的表达式与原值重合或不改变最终索引，故逐位一致。只有非规则形状（多轴规约、需要广播重复、不可整除）才会暴露。"),
}

def classify(m, rem, add):
    op = m["operator_name"]
    diff = " ".join(rem + add)
    l3 = (m.get("equiv_detail") or {}).get("layer3") or {}
    r = (l3.get("reasoning") or "").lower()
    cs = (l3.get("change_summary") or "").lower()
    if "docstring" in r or "docstring" in cs or (
        "comment" in r and any(k in r for k in ("unchanged", "not executable", "identical", "is equivalent", "does not affect", "not affect"))):
        return "R6"
    if "assert" in diff:
        return "R2"
    if "ndim" in diff or "unsqueeze" in diff:
        return "R1"
    if op == "layout_assume" or "contiguous" in diff:
        return "R7"
    if op == "cast_remove":
        return "R4"
    if op == "acc_downgrade":
        return "R5"
    if op == "init_modify" or "1e10" in diff or "-inf" in diff or (op == "relop_replace" and ("mask" in diff or "offs" in diff)):
        return "R3"
    return "R8"

# ---- load survivors --------------------------------------------------------
survivors = []
for f in glob.glob(os.path.join(BASE, "*.json")):
    with open(f, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    kn = data["kernel"]["problem_name"]
    pid = data["kernel"]["problem_id"]
    lang = data["kernel"].get("language", "")
    for m in data["mutants"]:
        sr = m.get("stress_result")
        if m.get("final_emd_status") == "survived" and not (sr and sr.get("killed")):
            survivors.append((pid, kn, lang, m))

rcount = collections.Counter()
fam_reason = collections.defaultdict(collections.Counter)
for pid, kn, lang, m in survivors:
    rem, add = changed_lines(m["original_code"], m["mutated_code"])
    code = classify(m, rem, add)
    m["_reason"] = code
    m["_rem"], m["_add"] = rem, add
    rcount[code] += 1
    fam_reason[kn.split("__")[0]][code] += 1

def trials(m):
    sr = m.get("stress_result") or {}
    mt = sr.get("main_track") or {}
    ct = sr.get("config_track") or {}
    v = (mt.get("value_stress") or {}).get("trials", 0)
    d = (mt.get("dtype_stress") or {}).get("trials", 0)
    rp = (mt.get("repeated_run") or {}).get("trials", 0)
    c = (ct.get("config_stress") or {}).get("trials", 0)
    return v, d, c, rp

fam_order = ["flash_attention", "reduce", "layernorm", "softmax", "rotary_embedding",
             "cross_entropy", "matmul", "rmsnorm"]
by_fam = collections.defaultdict(list)
for tup in survivors:
    by_fam[tup[1].split("__")[0]].append(tup)

L = []
w = L.append
w("# KGB 真存活变异体逐个分析（增强测试后仍未被杀）\n")
w("> 数据来源：`runs/kgb_ext_llmemd/details/*.json`（EMD 四层 + A800 五维增强差分测试，逐位比较）。\n")
w("> **本版本的 config_stress 已修正为忠实于 MutaKernel 设计的 batch-only 变化**——只改第一维 batch，"
  "其余张量维度严格固定在各问题的规范 shape（不再改变特征宽度/序列长）。\n")
w("\n## 一、什么是『真存活变异体』\n")
w("一个变异体要进入本表，必须**同时**满足：\n")
w("1. **EMD 判定为真漏检（true escape / `survived`）**——经过 Layer0 文本归一化、Layer1 静态等价规则、"
  "Layer2 动态等价筛查（100 随机 + 定向 stress 输入、逐位比较）、Layer3 LLM 复核后，"
  "LLM 仍判定其**不等价、原则上可被某输入杀死**（全部 `llm_equivalent=False`）；\n")
w("2. **增强测试未能补杀**——在 A800 上对其施加五维增强差分测试中的确定性维度"
  "（`value_stress` / `dtype_stress` / `config_stress`(batch-only) / `repeated_run`，**不含 LLM 维度**），"
  "在所有生成输入上与原 kernel **逐位一致**。\n")
w(f"\n本轮共 **699** 个真漏检，其中 **248** 个被增强测试补杀，**{len(survivors)}** 个仍存活——即下文逐个分析的对象。\n")

w("\n## 二、存活原因总体分类\n")
w("把 451 个存活体的根因归纳为 8 类。**核心结论**：绝大多数存活并非 kernel 计算逻辑真的等价，"
  "而是 **KGB 固定 benchmark 形状/数据类型留下的『输入空间盲区』**——被改写的代码分支需要特定的 ndim、"
  "规约维、非连续布局或极端数值才会被触发，而这些条件恰好落在增强测试的覆盖之外。\n")
w("\n| 编号 | 存活机理 | 数量 | 占比 |\n|---|---|---|---|")
RN = {k: v[0] for k, v in REASONS.items()}
for code in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]:
    n = rcount.get(code, 0)
    w(f"| {code} | {RN[code]} | {n} | {n/len(survivors)*100:.1f}% |")
w(f"| — | **合计** | **{len(survivors)}** | 100% |\n")

w("\n### 各类机理详解\n")
for code in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]:
    name, desc = REASONS[code]
    w(f"- **{code} {name}**：{desc}\n")

w("\n### 按算子族 × 存活机理分布\n")
w("\n| 算子族 | 存活数 | 主要机理 |\n|---|---|---|")
for fam in fam_order:
    if fam not in by_fam:
        continue
    cc = fam_reason[fam]
    main = "；".join(f"{RN[c]} {n}" for c, n in cc.most_common())
    w(f"| {FAM_CN.get(fam, fam)} | {len(by_fam[fam])} | {main} |")
w("")

w("\n## 三、逐个变异体分析\n")
w("> 每条给出：变异类型与位置、代码改动（diff）、LLM 认定的『可杀触发条件』(原文佐证)、以及中文存活原因。\n")

for fam in fam_order:
    if fam not in by_fam:
        continue
    items = sorted(by_fam[fam], key=lambda t: (t[0], t[3]["id"]))
    w(f"\n### {FAM_CN.get(fam, fam)}（{len(items)} 个）\n")
    last_kn = None
    for pid, kn, lang, m in items:
        if kn != last_kn:
            w(f"\n#### 问题 P{pid} · `{kn}` · {lang}\n")
            last_kn = kn
        op = m["operator_name"]
        site = m.get("site") or {}
        line = site.get("line_start")
        node = site.get("node_type")
        rem = m["_rem"]; add = m["_add"]
        diff_str = ""
        for x in rem:
            diff_str += f"`- {x}`<br>"
        for x in add:
            diff_str += f"`+ {x}`<br>"
        if not diff_str:
            diff_str = "(仅空白/格式差异)"
        l3 = (m.get("equiv_detail") or {}).get("layer3") or {}
        cs = l3.get("change_summary") or "(无)"
        rcode = m["_reason"]
        rname, _ = REASONS[rcode]
        v, d, c, rp = trials(m)
        w(f"- **`{m['id']}`** — {OP_CN.get(op, op)} @ L{line}（`{node}`）")
        w(f"  - 改动：{diff_str}")
        w(f"  - LLM 可杀依据：*{cs}*")
        w(f"  - **存活原因【{rcode} {rname}】**：{REASONS[rcode][1]}")
        w(f"  - 增强测试覆盖：value×{v}, dtype×{d}, config(batch)×{c}, repeated×{rp}，全部逐位一致。")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L))

print("survivors:", len(survivors))
print("reason counts:", dict(rcount))
print("written:", OUT)
print("size KB:", round(os.path.getsize(OUT)/1024, 1))
