#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按论文口径重算 Phase II 杀伤数（移除 DeepSeek-R1 LLM 兜底）。

论文最终的 MutaKernel 杀伤链只含 5 个确定性 stress 维度（value/dtype/training/
repeated/config）+ tier1_replay，已移除 Phase II 内嵌的 DeepSeek-R1 迭代分析兜底。
本脚本读取 ../details/*.json 的原始记录（不修改），改以
`kill_summary.deterministic_killed` 为准重新统计，输出 stress_summary_论文口径_无LLM.json。

关键事实：LLM 仅在"5 维全部未杀"时触发（trigger=all_dimensions_survived），
因此 llm_killed=True 必然 deterministic_killed=False，两个杀伤集合不相交。
"""
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DETAILS = HERE.parent / "details"


def main() -> None:
    files = sorted(DETAILS.glob("*.json"))
    total = len(files)
    det_killed = 0          # 论文口径：确定性维度杀死（= 论文 MutaKernel 杀数）
    llm_only_killed = []     # deterministic_killed=False 但 llm_killed=True（论文移除）
    final_killed = 0         # 原始 any_killed（含 LLM，历史口径）
    both_flag = 0            # deterministic 与 llm 同时为真（理论上应为 0）
    by_tier = {}             # tier -> [count, det_killed]
    det_first_kill = Counter()

    for fp in files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        ks = d.get("kill_summary", {})
        det = bool(ks.get("deterministic_killed", False))
        llm = bool(ks.get("llm_killed", False))
        fin = bool(d.get("any_killed", False))
        tier = d.get("tier")

        if fin:
            final_killed += 1
        if det:
            det_killed += 1
            mode = d.get("first_kill_mode")
            if mode == "llm_iterative_analysis":
                # 不应发生：若确定性已杀，first_kill_mode 不应是 LLM
                mode = "(anomaly:det_killed_but_llm_first)"
            det_first_kill[mode] += 1
        if (not det) and llm:
            llm_only_killed.append(d.get("mutant_id", fp.stem))
        if det and llm:
            both_flag += 1

        t = by_tier.setdefault(tier, [0, 0])
        t[0] += 1
        if det:
            t[1] += 1

    summary = {
        "_说明": "论文口径（已移除 Phase II DeepSeek-R1 LLM 兜底）。原始 details/*.json 未改动。",
        "total_mutants_in_phase2": total,
        "paper_mutakernel_killed_deterministic": det_killed,
        "legacy_any_killed_incl_llm": final_killed,
        "llm_only_killed_removed_count": len(llm_only_killed),
        "llm_only_killed_ids": sorted(llm_only_killed),
        "deterministic_and_llm_both_killed (应为0)": both_flag,
        "deterministic_first_kill_mode_distribution": dict(det_first_kill),
        "by_tier (count / deterministic_killed)": {
            str(k): {"count": v[0], "deterministic_killed": v[1]}
            for k, v in sorted(by_tier.items(), key=lambda x: (x[0] is None, x[0]))
        },
        "note": (
            "这 3 个 llm_only_killed 已在论文中移出 MutaKernel 杀数（169->166），"
            "改由 Task A(RQ2 审计)处理：L1_P49__init_modify__0 与 L1_P23__init_modify__0 "
            "归为 MutaKernel-missed，L1_P49__arith_replace__11 归为 operationally_indistinguishable。"
        ),
    }

    out = HERE / "stress_summary_论文口径_无LLM.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"total Phase II mutants        = {total}")
    print(f"paper MutaKernel killed (det) = {det_killed}")
    print(f"legacy any_killed (incl LLM)  = {final_killed}")
    print(f"LLM-only kills removed        = {len(llm_only_killed)} -> {sorted(llm_only_killed)}")
    print(f"det&llm both (should be 0)     = {both_flag}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
