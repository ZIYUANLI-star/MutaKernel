"""Aggregate KGB LLM-EMD: fold Opus-4.5 Layer-3 verdicts back into details +
summaries, recompute true-escape / mutation scores, and emit a complete result
tree (details/, summary.json, emd_summary.json, checkpoint.json, REPORT.md)
mirroring the CUDA-L1 layout.

Reads:
  runs/kgb_ext/details/*.json                 (original deterministic details)
  runs/kgb_ext/llm_emd/llm_verdicts.jsonl     (Opus-4.5 verdicts)
  runs/kgb_ext/emd_summary.json               (fault categories, by_op)
Writes (new tree, original kept intact):
  runs/kgb_ext_llmemd/details/*.json
  runs/kgb_ext_llmemd/summary.json
  runs/kgb_ext_llmemd/emd_summary.json
  runs/kgb_ext_llmemd/checkpoint.json
  runs/kgb_ext_llmemd/REPORT.md
"""
from __future__ import annotations

import collections
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(
    ROOT, "外部Benchmark差分测试_RQ4", "MutaKernel-KGB", "MutaKernel-KGB",
    "MutaKernel", "runs",
)
SRC = os.path.join(BASE, "kgb_ext")
DET_SRC = os.path.join(SRC, "details")
VERDICTS = os.path.join(SRC, "llm_emd", "llm_verdicts.jsonl")
OUT = os.path.join(BASE, "kgb_ext_llmemd")
DET_OUT = os.path.join(OUT, "details")

# operator name -> family (mirrors emd_summary by_op keys)
def op_family(kernel_name: str) -> str:
    return kernel_name.split("__", 1)[0]


def load_verdicts():
    v = {}
    with open(VERDICTS, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v[r["uid"]] = r
    return v


def score(killed, denom):
    return round(killed / denom, 4) if denom > 0 else 0.0


def main():
    os.makedirs(DET_OUT, exist_ok=True)
    verdicts = load_verdicts()

    # global accumulators
    g = collections.Counter()
    by_op = collections.defaultdict(lambda: collections.Counter())
    by_cat_fault = collections.Counter()
    conflict_strict = []          # strict but LLM says non-equiv
    flipped_to_escape = []        # candidate -> true_escape (LLM non-equiv)
    flipped_to_equiv = []         # survived -> equiv (LLM equiv)
    llm_missing = []
    per_kernel_rows = []

    files = sorted(f for f in os.listdir(DET_SRC) if f.endswith(".json"))
    for fn in files:
        with open(os.path.join(DET_SRC, fn), encoding="utf-8") as f:
            d = json.load(f)
        kname = d["kernel"]["problem_name"]
        fam = op_family(kname)

        kc = collections.Counter()           # final-status counter for this kernel
        cat_mut = collections.defaultdict(lambda: {"killed": 0, "denom": 0})

        for m in d["mutants"]:
            base = m["status"]
            uid = f"{fn}::{m['id']}"
            verdict = verdicts.get(uid)

            final = base
            if base in ("survived", "candidate_equivalent", "strict_equivalent"):
                ed = m.setdefault("equiv_detail", {})
                if verdict and verdict.get("llm_equivalent") is not None:
                    ed["layer3"] = {
                        "llm_equivalent": verdict["llm_equivalent"],
                        "confidence": verdict.get("llm_confidence", ""),
                        "change_summary": verdict.get("llm_change_summary", ""),
                        "reasoning": verdict.get("llm_reasoning", ""),
                        "model": "claude-opus-4-5",
                    }
                    if base == "strict_equivalent":
                        final = "strict_equivalent"  # textual/static authoritative
                        if verdict["llm_equivalent"] is False:
                            conflict_strict.append(uid)
                    elif verdict["llm_equivalent"] is True:
                        final = "candidate_equivalent"  # LLM-confirmed equivalent
                        if base == "survived":
                            flipped_to_equiv.append(uid)
                    else:
                        final = "survived"              # LLM-confirmed real escape
                        if base == "candidate_equivalent":
                            flipped_to_escape.append(uid)
                else:
                    ed["layer3"] = {"llm_equivalent": None, "note": "no LLM verdict (fallback to deterministic)"}
                    if base in ("survived", "candidate_equivalent"):
                        llm_missing.append(uid)
            m["final_emd_status"] = final
            kc[final] += 1

            # category score (conservative: exclude stillborn + strict)
            cat = m["operator_category"]
            if final not in ("stillborn", "strict_equivalent"):
                cat_mut[cat]["denom"] += 1
            if final == "killed":
                cat_mut[cat]["killed"] += 1

            # global / by_op (over survivors only for eq/escape breakdown)
            if base in ("survived", "candidate_equivalent", "strict_equivalent"):
                if final == "survived":
                    by_op[fam]["true_escape"] += 1
                    g["true_escape"] += 1
                else:
                    by_op[fam]["equivalent"] += 1
                    g["equivalent"] += 1
                by_op[fam]["survived_raw"] += 1

        total = kc.get("killed", 0) + kc.get("survived", 0) + kc.get("stillborn", 0) \
            + kc.get("strict_equivalent", 0) + kc.get("candidate_equivalent", 0)
        killed = kc.get("killed", 0)
        survived = kc.get("survived", 0)
        stillborn = kc.get("stillborn", 0)
        strict = kc.get("strict_equivalent", 0)
        cand = kc.get("candidate_equivalent", 0)
        denom_cons = total - stillborn - strict
        denom_opt = total - stillborn - strict - cand

        sbc = {}
        for cat in ["A", "B", "C", "D"]:
            sbc[cat] = score(cat_mut[cat]["killed"], cat_mut[cat]["denom"])

        d["summary"] = {
            "total": total,
            "killed": killed,
            "survived": survived,
            "stillborn": stillborn,
            "strict_equivalent": strict,
            "candidate_equivalent": cand,
            "equivalent": strict + cand,
            "mutation_score": score(killed, denom_cons),
            "mutation_score_optimistic": score(killed, denom_opt),
            "score_by_category": sbc,
            "emd_layer3": "claude-opus-4-5",
        }
        with open(os.path.join(DET_OUT, fn), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

        per_kernel_rows.append({
            "kernel": fn[:-5],
            "name": kname,
            "total": total, "killed": killed, "survived": survived,
            "stillborn": stillborn, "strict_equivalent": strict,
            "candidate_equivalent": cand,
            "score": score(killed, denom_cons),
            "score_optimistic": score(killed, denom_opt),
        })
        g["killed"] += killed
        g["survived_final"] += survived
        g["stillborn"] += stillborn
        g["strict"] += strict
        g["candidate"] += cand
        g["total"] += total

    # ---- aggregate summary.json ----
    tot_killed = g["killed"]
    tot_stillborn = g["stillborn"]
    tot_strict = g["strict"]
    tot_cand = g["candidate"]
    tot_surv = g["survived_final"]
    tot_total = g["total"]
    overall_cons = score(tot_killed, tot_total - tot_stillborn - tot_strict)
    overall_opt = score(tot_killed, tot_total - tot_stillborn - tot_strict - tot_cand)
    # LLM-adjusted headline: killed / (killed + true_escape)
    overall_llm = score(tot_killed, tot_killed + g["true_escape"])

    summary = {
        "total_kernels": len(files),
        "total_mutants": tot_total,
        "total_killed": tot_killed,
        "total_stillborn": tot_stillborn,
        "total_strict_equivalent": tot_strict,
        "total_candidate_equivalent": tot_cand,
        "total_true_escape": tot_surv,
        "overall_mutation_score_conservative": overall_cons,
        "overall_mutation_score_optimistic": overall_opt,
        "overall_mutation_score_llm_adjusted": overall_llm,
        "emd_layer3_model": "claude-opus-4-5",
        "by_kernel": per_kernel_rows,
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ---- emd_summary.json (LLM-adjusted) ----
    # carry fault categories from original emd_summary for context
    orig_emd = {}
    p = os.path.join(SRC, "emd_summary.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            orig_emd = json.load(f)

    emd = {
        "total_survived_raw": g["equivalent"] + g["true_escape"],
        "strict_equivalent": tot_strict,
        "candidate_equivalent_llm_confirmed": g["equivalent"] - tot_strict,
        "equivalent_total_after_llm": g["equivalent"],
        "true_escape_after_llm": g["true_escape"],
        "llm_verdicts_used": len(verdicts),
        "reclassification": {
            "candidate_equiv__to__true_escape": len(flipped_to_escape),
            "survived__to__equivalent": len(flipped_to_equiv),
            "strict_conflict_llm_nonequiv": len(conflict_strict),
            "survivors_without_llm_verdict": len(llm_missing),
        },
        "by_op_after_llm": {k: dict(v) for k, v in sorted(by_op.items())},
        "original_deterministic_emd": {
            "strict_equivalent": orig_emd.get("strict_equivalent"),
            "candidate_equivalent": orig_emd.get("candidate_equivalent"),
            "true_escape": orig_emd.get("true_escape"),
            "by_fault_category": orig_emd.get("by_fault_category"),
            "by_op": orig_emd.get("by_op"),
        },
    }
    with open(os.path.join(OUT, "emd_summary.json"), "w", encoding="utf-8") as f:
        json.dump(emd, f, ensure_ascii=False, indent=2)

    # ---- checkpoint.json (mirror CUDA-L1 minimal completeness marker) ----
    with open(os.path.join(OUT, "checkpoint.json"), "w", encoding="utf-8") as f:
        json.dump({
            "completed_kernels": [r["kernel"] for r in per_kernel_rows],
            "n_kernels": len(files),
            "n_mutants": tot_total,
            "n_llm_verdicts": len(verdicts),
            "emd_layer3_model": "claude-opus-4-5",
        }, f, ensure_ascii=False, indent=2)

    print("=== LLM-EMD aggregation done ===")
    print(f"kernels={len(files)} mutants={tot_total}")
    print(f"killed={tot_killed} stillborn={tot_stillborn} strict={tot_strict} "
          f"candidate(LLM-eq)={tot_cand} true_escape(LLM)={tot_surv}")
    print(f"survivors equivalent_after_llm={g['equivalent']} true_escape_after_llm={g['true_escape']}")
    print(f"reclass: candidate->escape={len(flipped_to_escape)} "
          f"survived->equiv={len(flipped_to_equiv)} strict_conflict={len(conflict_strict)} "
          f"missing_llm={len(llm_missing)}")
    print(f"scores: conservative={overall_cons} optimistic={overall_opt} llm_adjusted={overall_llm}")
    print(f"written -> {OUT}")
    return summary, emd


if __name__ == "__main__":
    main()
