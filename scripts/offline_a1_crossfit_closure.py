#!/usr/bin/env python3
"""Offline analyzer 1 — §5.6 cross-fitted closure rate (conservative lower
bound) + non-cross-fitted upper-bound curve.

For every witnessed blind-spot probe (equiv WITNESSED_NON_EQUIVALENT + CSE
FALSIFIED) it asks: does the held-out fold map's first k=8 planned cases
contain the probe's *recorded* witnessing case?  Because round logs stop at
the first sound divergence, this is a lower bound on true closure; the
full-data (non-cross-fitted) map gives the matching upper bound.  Witnessed
probes that stay unclosed because some planned case was never executed on
them are listed as small-scale verification-rerun candidates.

Usage:
  python scripts/offline_a1_crossfit_closure.py \
      --e1-dir <dir with equiv/cse/baseline jsonl> --out-dir <analysis dir> \
      [--cse-obs FILE ...] [--folds 5] [--planned-cases 8] [--final]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from offline_reuse_lib import (  # noqa: E402
    DEFAULT_SCOPE_INTERIM,
    build_map_records,
    closure_evaluation,
    executed_case_status,
    extract_witnesses,
    load_dataset,
    provenance,
    try_matplotlib,
    write_csv,
    write_json,
)


def run(e1_dir: Path, out_dir: Path, cse_files, folds: int,
        planned_cases: int, scope_label: str) -> dict:
    dataset = load_dataset(e1_dir, cse_files)
    witnesses = extract_witnesses(dataset["equiv"], dataset["cse"])
    executed = executed_case_status(dataset["equiv"], dataset["cse"])
    records = build_map_records(
        dataset["equiv"], dataset["cse"], dataset["baseline"],
        include_baseline_kills=True)

    result = closure_evaluation(
        records, witnesses, executed,
        k_folds=folds, planned_cases=planned_cases)
    result["provenance"] = provenance(dataset, scope_label)
    result["notes"] = [
        "closure criterion: recorded witness case within held-out top-k "
        "planned cases (offline conservative lower bound)",
        "map training includes baseline IID kills (run_e1_probe_study "
        "--phase map convention); evaluation is restricted to blind-spot "
        "witnesses (equiv WITNESSED + CSE FALSIFIED)",
        "unclosed_evidence_insufficient probes are the candidates for the "
        "small-scale verification rerun (planned cases never executed on "
        "them because of first-kill early exit or lane scope)",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "a1_crossfit_closure.json", result)

    write_csv(
        out_dir / "a1_closure_curve.csv",
        ["k", "closure_lower_bound", "closure_upper_bound"],
        [
            (lo["k"], lo["closure_rate"], up["closure_rate"])
            for lo, up in zip(result["closure_curve_lower"],
                              result["closure_curve_upper"])
        ])

    unclosed = [r for r in result["per_probe"] if not r["closed_at_planned_k"]]
    write_csv(
        out_dir / "a1_unclosed_witnessed.csv",
        ["probe_id", "fault_class", "source", "witness_policy",
         "classification", "crossfit_rank_of_witness",
         "planned_not_executed", "planned_inconclusive"],
        [
            (r["probe_id"], r["fault_class"], r["source"],
             r["witness_policy_label"], r["classification"],
             r["crossfit_rank_of_witness"],
             sum(1 for s in r["planned_case_status"] if s["recorded"] == "not_executed"),
             sum(1 for s in r["planned_case_status"] if s["recorded"] == "inconclusive"))
            for r in unclosed
        ])

    plt = try_matplotlib()
    if plt is not None:
        ks = [p["k"] for p in result["closure_curve_lower"]]
        lower = [p["closure_rate"] for p in result["closure_curve_lower"]]
        upper = [p["closure_rate"] for p in result["closure_curve_upper"]]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ks, upper, marker="s", label="non-cross-fitted (upper bound)")
        ax.plot(ks, lower, marker="o", label=f"{folds}-fold cross-fitted (lower bound)")
        ax.axvline(planned_cases, color="grey", linestyle="--", linewidth=1,
                   label=f"k = {planned_cases} planned cases")
        ax.set_xlabel("planned cases k")
        ax.set_ylabel("closure rate of witnessed blind spots")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title("Blueprint 5.6 closure curve (offline conservative)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "a1_closure_curve.png", dpi=150)
        plt.close(fig)

    lb = result["pooled_lower_bound"]
    ub = result["upper_bound_full_map"]
    print(f"a1 closure @k={planned_cases}: lower {lb['closed']}/{result['witnessed_total']}"
          f" = {lb['closure_rate']}, upper {ub['closed']}/{result['witnessed_total']}"
          f" = {ub['closure_rate']}, classes {result['classification_counts']}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e1-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cse-obs", type=Path, nargs="*", default=None,
                    help="CSE observation jsonl files (default: lanes 1+2)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--planned-cases", type=int, default=8)
    ap.add_argument("--final", action="store_true",
                    help="label outputs as final scope instead of interim")
    args = ap.parse_args()
    scope = "final" if args.final else DEFAULT_SCOPE_INTERIM
    run(args.e1_dir, args.out_dir, args.cse_obs, args.folds,
        args.planned_cases, scope)


if __name__ == "__main__":
    main()
