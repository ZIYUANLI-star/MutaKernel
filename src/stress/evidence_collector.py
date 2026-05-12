"""Task B evidence aggregator.

For a single kernel (e.g. ``L1_P100``), collect mutation testing evidence
from four stages:

- Phase I  — ``第二次实验汇总/full_block12_results/details/{kernel}.json``
- Phase II — ``第二次实验汇总/stress_enhance_results/details/{kernel}__*.json``
- Task A   — ``第二次实验汇总/第二次实验汇总_补充/task_a_phase2_rerun/details/{mutant}.json``
- Task C   — ``第二次实验汇总/第二次实验汇总_补充/task_c_phase1_direct/details/{mutant}.json``

Output: a dict of strings ready for ``str.format`` substitution into
``STRENGTHEN_PROMPT_B`` (see :pyfunc:`build_strengthen_prompt`).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Paths (default; can be overridden)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SUMMARY_DIR = PROJECT_ROOT / "第二次实验汇总"
SUPPL_DIR = SUMMARY_DIR / "第二次实验汇总_补充"

PHASE1_DETAILS = SUMMARY_DIR / "full_block12_results" / "details"
PHASE2_DETAILS = SUMMARY_DIR / "stress_enhance_results" / "details"
TASK_A_DETAILS = SUPPL_DIR / "task_a_phase2_rerun" / "details"
TASK_C_DETAILS = SUPPL_DIR / "task_c_phase1_direct" / "details"


# ---------------------------------------------------------------------------
# Phase I — load and classify
# ---------------------------------------------------------------------------
def load_phase1_kernel(kernel_name: str) -> Optional[Dict[str, Any]]:
    f = PHASE1_DETAILS / f"{kernel_name}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def classify_phase1(detail: Dict[str, Any]) -> Dict[str, Any]:
    """Return per-mutant Phase I final status (killed/equivalent/survived).

    Schema of returned dict::

        {
            "by_status": {"killed": [...], "survived": [...], ...},
            "mutant_index": {mutant_id: phase1_record},
        }
    """
    by_status: Dict[str, List[str]] = defaultdict(list)
    mutant_index: Dict[str, Dict[str, Any]] = {}
    for m in detail.get("mutants", []):
        mid = m.get("id")
        if not mid:
            continue
        mutant_index[mid] = m
        status = m.get("final_status") or m.get("status") or "unknown"
        by_status[status].append(mid)
    return {"by_status": dict(by_status), "mutant_index": mutant_index}


# ---------------------------------------------------------------------------
# Phase II — load all stress-enhance detail jsons for one kernel
# ---------------------------------------------------------------------------
def load_phase2_kernel(kernel_name: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    prefix = f"{kernel_name}__"
    for f in sorted(PHASE2_DETAILS.glob(f"{prefix}*.json")):
        try:
            results.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            continue
    return results


def classify_phase2(phase2_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate Phase II per-mutant outcomes.

    Returns::

        {
            "det_killed": [...],         # killed by 5-dimension deterministic
            "det_survived": [...],       # survived deterministic; LLM may have killed
            "llm_killed": [...],         # killed by 3-round DeepSeek-R1 iter
            "llm_survived": [...],
            "any_killed": [...],         # mutants Phase II killed (any path)
            "all_unkilled": [...],       # mutants Phase II failed to kill
            "by_mutant": {mid: detail},
        }
    """
    det_killed, det_survived = [], []
    llm_killed, llm_survived = [], []
    any_killed_list, all_unkilled = [], []
    by_mutant: Dict[str, Dict[str, Any]] = {}

    for d in phase2_records:
        mid = d.get("mutant_id")
        if not mid:
            continue
        by_mutant[mid] = d
        ks = d.get("kill_summary") or {}
        det_ok = bool(ks.get("deterministic_killed"))
        llm_ok = bool(ks.get("llm_killed"))
        any_ok = bool(d.get("any_killed"))

        (det_killed if det_ok else det_survived).append(mid)
        (llm_killed if llm_ok else llm_survived).append(mid)
        (any_killed_list if any_ok else all_unkilled).append(mid)

    return {
        "det_killed": det_killed, "det_survived": det_survived,
        "llm_killed": llm_killed, "llm_survived": llm_survived,
        "any_killed": any_killed_list, "all_unkilled": all_unkilled,
        "by_mutant": by_mutant,
    }


# ---------------------------------------------------------------------------
# Task A / Task C — load per-mutant results
# ---------------------------------------------------------------------------
def _load_details_dir(details_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read every JSON in details/, key by mutant_id."""
    out: Dict[str, Dict[str, Any]] = {}
    if not details_dir.exists():
        return out
    for f in sorted(details_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        mid = d.get("mutant_id")
        if mid:
            out[mid] = d
    return out


def load_taskA() -> Dict[str, Dict[str, Any]]:
    return _load_details_dir(TASK_A_DETAILS)


def load_taskC() -> Dict[str, Dict[str, Any]]:
    return _load_details_dir(TASK_C_DETAILS)


def filter_kernel(per_mutant: Dict[str, Dict[str, Any]],
                  kernel_name: str) -> Dict[str, Dict[str, Any]]:
    return {mid: d for mid, d in per_mutant.items()
            if mid.startswith(f"{kernel_name}__")}


def classify_taskAC(per_mutant: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    killed, survived = [], []
    by_reason: Dict[str, List[str]] = defaultdict(list)
    by_mutant: Dict[str, Dict[str, Any]] = {}
    for mid, d in per_mutant.items():
        by_mutant[mid] = d
        if d.get("killed"):
            killed.append(mid)
        else:
            survived.append(mid)
        for r in d.get("rounds", []):
            cat = r.get("reason_category")
            if cat:
                by_reason[cat].append(mid)
                break
    return {
        "killed": killed, "survived": survived,
        "by_reason": dict(by_reason),
        "by_mutant": by_mutant,
    }


# ---------------------------------------------------------------------------
# Reason clustering across A + C
# ---------------------------------------------------------------------------
def cluster_survival_reasons(
    taskA_by_mutant: Dict[str, Dict[str, Any]],
    taskC_by_mutant: Dict[str, Dict[str, Any]],
    final_unkilled: Set[str],
) -> str:
    """Markdown table grouping unkilled mutants by Opus reason_category.

    Combines A and C results; for a mutant present in both, prefer A.
    """
    per_mutant_reason: Dict[str, Tuple[str, str, str]] = {}
    for mid in sorted(final_unkilled):
        rec = taskA_by_mutant.get(mid) or taskC_by_mutant.get(mid)
        if not rec:
            continue
        rounds = rec.get("rounds") or []
        if not rounds:
            continue
        last = next((r for r in reversed(rounds)
                    if r.get("reason_category")), rounds[-1])
        cat = last.get("reason_category") or "unknown"
        proof = (last.get("proof_sketch") or "").strip()
        reason = (last.get("survival_reason") or "").strip()
        per_mutant_reason[mid] = (cat, proof, reason)

    by_cat: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for mid, (cat, proof, reason) in per_mutant_reason.items():
        by_cat[cat].append((mid, proof, reason))

    if not by_cat:
        return "(no Opus survival reasons available for this kernel)"

    lines = []
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        lines.append(f"\n### Category: `{cat}` ({len(items)} mutants)\n")
        for mid, proof, reason in items[:5]:
            short_reason = reason.replace("\n", " ")
            if len(short_reason) > 280:
                short_reason = short_reason[:280] + "..."
            short_proof = proof.replace("\n", " ")
            if len(short_proof) > 220:
                short_proof = short_proof[:220] + "..."
            lines.append(f"- **{mid}**")
            if short_proof:
                lines.append(f"  - Proof sketch: {short_proof}")
            lines.append(f"  - Survival reason: {short_reason}")
        if len(items) > 5:
            lines.append(f"- ...and {len(items)-5} more in this category")
    return "\n".join(lines).strip()


def select_top_unkilled_examples(
    final_unkilled: Set[str],
    phase1_idx: Dict[str, Dict[str, Any]],
    phase2_by_mutant: Dict[str, Dict[str, Any]],
    taskA_by_mutant: Dict[str, Dict[str, Any]],
    taskC_by_mutant: Dict[str, Dict[str, Any]],
    n: int = 5,
) -> str:
    """Pick top-N representative unkilled mutants with mutation diff + reason.

    Selection prioritizes tier 1 (most behavior-changing) and operator diversity.
    """
    candidates = []
    for mid in final_unkilled:
        ph2 = phase2_by_mutant.get(mid, {})
        tier = ph2.get("tier") or 99
        op = ph2.get("operator_name") or "unknown"
        candidates.append((tier, op, mid))
    candidates.sort()

    chosen: List[str] = []
    seen_ops: Set[str] = set()
    for tier, op, mid in candidates:
        if op in seen_ops and len(chosen) < n:
            continue
        chosen.append(mid)
        seen_ops.add(op)
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        for tier, op, mid in candidates:
            if mid in chosen:
                continue
            chosen.append(mid)
            if len(chosen) >= n:
                break

    blocks = []
    for mid in chosen:
        p1 = phase1_idx.get(mid, {})
        site = p1.get("site", {}) or {}
        op = (p1.get("operator_name")
              or phase2_by_mutant.get(mid, {}).get("operator_name", "?"))
        rec = taskA_by_mutant.get(mid) or taskC_by_mutant.get(mid) or {}
        reason_cat = "unknown"
        proof = ""
        survival = ""
        for r in (rec.get("rounds") or []):
            if r.get("reason_category"):
                reason_cat = r.get("reason_category")
                proof = (r.get("proof_sketch") or "").replace("\n", " ")
                survival = (r.get("survival_reason") or "").replace("\n", " ")
                break

        line_start = site.get("line_start", "?")
        orig_frag = (site.get("original_code") or "").strip()
        mutated_code = p1.get("mutated_code", "")
        original_code = p1.get("original_code", "")

        diff_hint = _short_diff(original_code, mutated_code)

        b = (
            f"\n### `{mid}`  (operator={op}, line {line_start})\n"
            f"- Mutated fragment from: `{orig_frag}`\n"
            f"- Single-line diff hint:\n```diff\n{diff_hint}\n```\n"
            f"- Opus 4.5 verdict: `{reason_cat}`\n"
            f"  - Proof: {proof[:300]}{'...' if len(proof) > 300 else ''}\n"
            f"  - Survival: {survival[:400]}{'...' if len(survival) > 400 else ''}\n"
        )
        blocks.append(b)
    return "\n".join(blocks).strip() if blocks else "(none)"


def _short_diff(a: str, b: str, ctx: int = 1) -> str:
    """Compute a tiny line-level diff (no library) suitable for prompts."""
    a_lines = a.splitlines()
    b_lines = b.splitlines()
    out = []
    n = max(len(a_lines), len(b_lines))
    for i in range(n):
        la = a_lines[i] if i < len(a_lines) else ""
        lb = b_lines[i] if i < len(b_lines) else ""
        if la != lb:
            out.append(f"- {la}")
            out.append(f"+ {lb}")
            if len(out) >= 8:
                out.append("(... diff truncated ...)")
                break
    return "\n".join(out) if out else "(no line-level diff; mutation may be sub-expression only)"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def collect_kernel_evidence(
    kernel_name: str,
    taskA_per_mutant: Optional[Dict[str, Dict[str, Any]]] = None,
    taskC_per_mutant: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """Build the full evidence package for one kernel.

    Returns a dict with two main sections:

    - ``summary``: ready-to-format strings keyed for STRENGTHEN_PROMPT_B
    - ``meta``: raw mutant lists and indexes for downstream analysis

    Pass ``taskA_per_mutant`` / ``taskC_per_mutant`` to share already-loaded
    A/C indexes across many kernels (avoid re-reading disk).
    """
    p1_detail = load_phase1_kernel(kernel_name)
    if not p1_detail:
        return None
    p1_cls = classify_phase1(p1_detail)
    p1_index: Dict[str, Dict[str, Any]] = p1_cls["mutant_index"]

    phase1_killed = set(p1_cls["by_status"].get("killed", []))
    phase1_equiv = (set(p1_cls["by_status"].get("equivalent", []))
                    | set(p1_cls["by_status"].get("candidate_equivalent", [])))
    phase1_survived = (set(p1_index.keys())
                       - phase1_killed - phase1_equiv)

    p2_records = load_phase2_kernel(kernel_name)
    p2_cls = classify_phase2(p2_records)
    p2_index = p2_cls["by_mutant"]
    p2_unkilled = set(p2_cls["all_unkilled"])

    if taskA_per_mutant is None:
        taskA_per_mutant = load_taskA()
    if taskC_per_mutant is None:
        taskC_per_mutant = load_taskC()
    tA_kernel = filter_kernel(taskA_per_mutant, kernel_name)
    tC_kernel = filter_kernel(taskC_per_mutant, kernel_name)
    tA_cls = classify_taskAC(tA_kernel)
    tC_cls = classify_taskAC(tC_kernel)
    tA_killed_set = set(tA_cls["killed"])
    tC_killed_set = set(tC_cls["killed"])

    final_unkilled: Set[str] = set()
    for mid in phase1_survived:
        if mid in phase1_killed or mid in phase1_equiv:
            continue
        killed_anywhere = False
        if mid in p2_index and p2_index[mid].get("any_killed"):
            killed_anywhere = True
        if mid in tA_killed_set or mid in tC_killed_set:
            killed_anywhere = True
        if not killed_anywhere:
            final_unkilled.add(mid)

    survival_clusters = cluster_survival_reasons(
        tA_cls["by_mutant"], tC_cls["by_mutant"], final_unkilled
    )
    top_examples = select_top_unkilled_examples(
        final_unkilled, p1_index, p2_index,
        tA_cls["by_mutant"], tC_cls["by_mutant"], n=5,
    )

    phase1_rounds = 0
    for m in p1_index.values():
        ed = m.get("equiv_detail") or {}
        l2 = ed.get("layer2") or {}
        rr = l2.get("total_rounds") or 0
        if rr > phase1_rounds:
            phase1_rounds = rr

    summary = {
        "phase1_rounds": phase1_rounds or "~112",
        "phase1_killed": len(phase1_killed),
        "phase1_survived": len(phase1_survived) + len(phase1_equiv),
        "phase2_killed_det": len(p2_cls["det_killed"]),
        "phase2_survived_det": len(p2_cls["det_survived"]),
        "phase2_killed_llm": len(p2_cls["llm_killed"]),
        "phase2_survived_llm": len(p2_cls["llm_survived"]),
        "taskA_killed": len(tA_killed_set),
        "taskA_survived": len(tA_cls["survived"]),
        "taskC_killed": len(tC_killed_set),
        "taskC_survived": len(tC_cls["survived"]),
        "final_unkilled": len(final_unkilled),
        "survival_reason_clusters": survival_clusters,
        "top_unkilled_examples": top_examples,
    }

    meta = {
        "kernel_name": kernel_name,
        "kernel_meta": p1_detail.get("kernel", {}),
        "phase1_index": p1_index,
        "phase1_killed_ids": sorted(phase1_killed),
        "phase1_equiv_ids": sorted(phase1_equiv),
        "phase1_survived_ids": sorted(phase1_survived),
        "phase2_by_mutant": p2_index,
        "phase2_unkilled_ids": sorted(p2_unkilled),
        "taskA_killed_ids": sorted(tA_killed_set),
        "taskC_killed_ids": sorted(tC_killed_set),
        "final_unkilled_ids": sorted(final_unkilled),
    }
    return {"summary": summary, "meta": meta}


def list_kernels_with_unkilled() -> List[Tuple[str, int]]:
    """Enumerate kernels that still have non-equivalent unkilled mutants.

    Returns sorted list of ``(kernel_name, final_unkilled_count)``.
    Requires Task A and Task C details to be present (run after they finish).
    """
    taskA = load_taskA()
    taskC = load_taskC()
    out: List[Tuple[str, int]] = []
    for f in sorted(PHASE1_DETAILS.glob("L*.json")):
        kernel = f.stem
        ev = collect_kernel_evidence(kernel, taskA, taskC)
        if not ev:
            continue
        n = ev["summary"]["final_unkilled"]
        if isinstance(n, int) and n > 0:
            out.append((kernel, n))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect Task B evidence for one kernel")
    parser.add_argument("kernel", nargs="?", default=None,
                        help="Kernel name like L1_P1; omit to list all kernels with unkilled")
    parser.add_argument("--show-prompt", action="store_true",
                        help="Print the rendered STRENGTHEN_PROMPT_B")
    args = parser.parse_args()

    if args.kernel is None:
        print("# Kernels with final_unkilled > 0 (run after Task A+C)\n")
        for k, n in list_kernels_with_unkilled():
            print(f"  {k}\t{n}")
    else:
        ev = collect_kernel_evidence(args.kernel)
        if not ev:
            print(f"ERROR: no Phase I record for {args.kernel}")
            raise SystemExit(2)
        print("# Summary fields")
        for k, v in ev["summary"].items():
            if isinstance(v, str) and "\n" in v:
                print(f"\n## {k}\n{v}\n")
            else:
                print(f"  {k}: {v}")
