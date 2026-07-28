#!/usr/bin/env python3
"""Offline analyzer 3 — budget-recall curve (§5.7 saturation budget) and the
A13 dual accounting (per-round hit rate vs budget share, random vs directed).

Replays the equiv/CSE round logs in executed order:
  (a) cumulative witness recall as the per-probe round budget grows —
      phase-local curve (equiv search, 20 random + 12 directed rounds) and
      combined curve (a CSE witness first consumed the probe's full equiv
      budget);
  (b) A13: for every policy (random rounds kept separate) the per-round hit
      rate, its share of the round budget and its share of measured GPU wall
      time — the two-accounting rebuttal of the denominator-confusion attack.

Usage:
  python scripts/offline_a3_budget_recall.py --e1-dir <dir> --out-dir <dir>
      [--cse-obs FILE ...] [--final]
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
    extract_witnesses,
    load_dataset,
    policy_round_stats,
    provenance,
    recall_curve,
    try_matplotlib,
    witness_budget_indices,
    write_csv,
    write_json,
)


def run(e1_dir: Path, out_dir: Path, cse_files, scope_label: str) -> dict:
    dataset = load_dataset(e1_dir, cse_files)
    witnesses = extract_witnesses(dataset["equiv"], dataset["cse"])
    indices = witness_budget_indices(witnesses, dataset["equiv"])

    max_budget = max(
        (i["combined_round_index"] for i in indices
         if i["combined_round_index"] is not None), default=0)
    max_budget = max(max_budget, 32)
    phase_curve = recall_curve(
        [i["phase_round_index"] for i in indices], max_budget)
    combined_curve = recall_curve(
        [i["combined_round_index"] for i in indices], max_budget)

    yield_equiv = policy_round_stats(dataset["equiv"], "equiv")
    yield_cse = policy_round_stats(dataset["cse"], "cse")

    def _aggregate(stats):
        random_rows = [s for s in stats if s["policy"] == "random"]
        directed_rows = [s for s in stats if s["policy"] != "random"]

        def _agg(rows, label):
            rounds = sum(r["rounds"] for r in rows)
            hits = sum(r["witnesses"] for r in rows)
            return {"group": label, "rounds": rounds, "witnesses": hits,
                    "hit_rate_per_round": round(hits / rounds, 6) if rounds else None}
        return [_agg(random_rows, "random"), _agg(directed_rows, "directed")]

    result = {
        "witnessed_total": len(indices),
        "witness_indices": indices,
        "recall_curve_phase_local": phase_curve,
        "recall_curve_combined": combined_curve,
        "policy_yield_equiv": yield_equiv,
        "policy_yield_cse": yield_cse,
        "a13_group_comparison": {
            "equiv": _aggregate(yield_equiv),
            "cse": _aggregate(yield_cse),
        },
        "notes": [
            "recall denominator = witnessed blind spots recorded so far "
            "(equiv WITNESSED + CSE FALSIFIED); the curve is a replay of "
            "recorded logs, not a re-execution",
            "combined curve charges a CSE witness with the probe's full "
            "equiv-phase budget first (per-probe cumulative invocations)",
            "A13 accounting (a): hit_rate_per_round = witnesses / executed "
            "rounds of that policy; (b): budget_share_rounds and "
            "budget_share_wall_ms",
        ],
        "provenance": provenance(dataset, scope_label),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "a3_budget_recall.json", result)
    write_csv(
        out_dir / "a3_budget_recall_curve.csv",
        ["budget_rounds", "recall_phase_local", "recall_combined"],
        [
            (p["budget_rounds"], p["recall"], c["recall"])
            for p, c in zip(phase_curve, combined_curve)
        ])
    write_csv(
        out_dir / "a3_policy_yield.csv",
        ["source", "policy", "rounds", "conclusive_rounds", "witnesses",
         "hit_rate_per_round", "budget_share_rounds", "budget_share_wall_ms"],
        [
            (s["source"], s["policy"], s["rounds"], s["conclusive_rounds"],
             s["witnesses"], s["hit_rate_per_round"],
             s["budget_share_rounds"], s["budget_share_wall_ms"])
            for s in yield_equiv + yield_cse
        ])

    plt = try_matplotlib()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        xs = [p["budget_rounds"] for p in combined_curve]
        ax.step(xs, [p["recall"] for p in phase_curve], where="post",
                label="phase-local budget")
        ax.step(xs, [p["recall"] for p in combined_curve], where="post",
                linestyle="--", label="combined (equiv + CSE) budget")
        ax.set_xlabel("per-probe round budget (validator invocations)")
        ax.set_ylabel("cumulative witness recall")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_title("Budget-recall replay (offline, recorded witnesses)")
        fig.tight_layout()
        fig.savefig(out_dir / "a3_budget_recall_curve.png", dpi=150)
        plt.close(fig)

        rows = sorted(yield_equiv, key=lambda s: s["hit_rate_per_round"])
        fig, ax = plt.subplots(figsize=(8, 6))
        names = [s["policy"] for s in rows]
        ax.barh(names, [s["hit_rate_per_round"] for s in rows],
                color=["tab:grey" if n == "random" else "tab:blue" for n in names],
                label="hit rate per round")
        ax.barh(names, [-(s["budget_share_rounds"] or 0) for s in rows],
                color="tab:orange", alpha=0.7, label="budget share (rounds)")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("<- budget share (rounds)   |   per-round hit rate ->")
        ax.set_title("A13 dual accounting, equiv phase (interim)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "a3_policy_yield.png", dpi=150)
        plt.close(fig)

    groups = result["a13_group_comparison"]["equiv"]
    print(f"a3 witnesses={len(indices)}; equiv per-round hit rate: "
          + ", ".join(f"{g['group']}={g['hit_rate_per_round']}" for g in groups))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e1-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cse-obs", type=Path, nargs="*", default=None)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    scope = "final" if args.final else DEFAULT_SCOPE_INTERIM
    run(args.e1_dir, args.out_dir, args.cse_obs, scope)


if __name__ == "__main__":
    main()
