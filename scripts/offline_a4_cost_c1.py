#!/usr/bin/env python3
"""Offline analyzer 4 — §5.7 cost, C1 side.

Aggregates recorded wall_ms per phase (original controls, B1 baseline
replay, equiv search, CSE falsification) into per-probe median / p95 / mean,
per-round costs from trial timings, phase GPU-busy totals, and the
"validate 1,000 candidates" extrapolations the blueprint §5.7 sentence
needs.  GPU-busy seconds = sum of recorded wall_ms (drivers are serial per
lane; parallel lanes overlap in wall-clock, so busy-time is the meaningful
cost unit).

Usage:
  python scripts/offline_a4_cost_c1.py --e1-dir <dir> --out-dir <dir>
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
    MS_PER_HOUR,
    cost_stats,
    load_dataset,
    provenance,
    trial_total_ms,
    write_csv,
    write_json,
)


def _wall(rows, predicate=lambda r: True):
    return [r["wall_ms"] for r in rows
            if predicate(r) and isinstance(r.get("wall_ms"), (int, float))]


def _round_costs(rows):
    return [c for r in rows for t in (r.get("trials") or [])
            for c in [trial_total_ms(t)] if c is not None]


def run(e1_dir: Path, out_dir: Path, cse_files, scope_label: str) -> dict:
    dataset = load_dataset(e1_dir, cse_files)
    controls = dataset["controls"]

    sections = {
        "original_controls": cost_stats(
            [v["wall_ms"] for v in controls.values()
             if isinstance(v.get("wall_ms"), (int, float))]),
        "baseline_all_executed": cost_stats(_wall(dataset["baseline"])),
        "baseline_killed": cost_stats(
            _wall(dataset["baseline"], lambda r: r.get("status") == "killed")),
        "baseline_survived": cost_stats(
            _wall(dataset["baseline"], lambda r: r.get("status") == "survived")),
        "equiv_per_probe": cost_stats(_wall(dataset["equiv"])),
        "equiv_witnessed": cost_stats(
            _wall(dataset["equiv"],
                  lambda r: r.get("evidence_grade") == "WITNESSED_NON_EQUIVALENT")),
        "equiv_likely_equivalent": cost_stats(
            _wall(dataset["equiv"],
                  lambda r: r.get("evidence_grade") == "LIKELY_EQUIVALENT")),
        "cse_per_probe": cost_stats(_wall(dataset["cse"])),
        "equiv_per_round": cost_stats(_round_costs(dataset["equiv"])),
        "cse_per_round": cost_stats(_round_costs(dataset["cse"])),
    }

    phase_totals_ms = {
        "original_controls": sections["original_controls"].get("total_ms", 0.0),
        "baseline": sections["baseline_all_executed"].get("total_ms", 0.0),
        "equiv": sections["equiv_per_probe"].get("total_ms", 0.0),
        "cse_loaded_lanes": sections["cse_per_probe"].get("total_ms", 0.0),
    }
    total_ms = sum(v or 0.0 for v in phase_totals_ms.values())

    def _per_1000(stats):
        median = stats.get("median_ms")
        return round(median * 1000 / MS_PER_HOUR, 3) if median else None

    result = {
        "sections": sections,
        "phase_gpu_busy": {
            **{k: {"total_ms": v,
                   "gpu_hours": round((v or 0.0) / MS_PER_HOUR, 4)}
               for k, v in phase_totals_ms.items()},
            "all_loaded_phases": {"total_ms": round(total_ms, 1),
                                  "gpu_hours": round(total_ms / MS_PER_HOUR, 4)},
        },
        "per_1000_candidates_gpu_hours": {
            "b1_baseline_5draw": _per_1000(sections["baseline_all_executed"]),
            "equiv_search_deploy_like_early_exit": _per_1000(sections["equiv_per_probe"]),
            "cse_full_audit_like_103_rounds": _per_1000(sections["cse_per_probe"]),
        },
        "notes": [
            "GPU-busy = sum of recorded per-probe wall_ms (serial drivers); "
            "parallel lanes overlap in wall-clock so busy-time != elapsed time",
            "deploy-like figure uses the equiv search (early exit on first "
            "sound violation, 20 random + <=12 directed rounds); full-audit-"
            "like uses the CSE search (40 random + 21x3 stress rounds)",
            "wall_ms includes worker startup + JIT compilation; per-round "
            "trial timings (equiv_per_round/cse_per_round) exclude them",
            "cse_loaded_lanes covers only the CSE observation files passed "
            "in; in-flight lanes are excluded until the final rerun",
        ],
        "provenance": provenance(dataset, scope_label),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "a4_cost_c1.json", result)
    write_csv(
        out_dir / "a4_cost_c1.csv",
        ["section", "n", "median_ms", "p95_ms", "mean_ms", "total_ms",
         "total_gpu_hours"],
        [
            (name, s.get("n"), s.get("median_ms"), s.get("p95_ms"),
             s.get("mean_ms"), s.get("total_ms"), s.get("total_gpu_hours"))
            for name, s in sections.items()
        ])

    eq = sections["equiv_per_probe"]
    print(f"a4 equiv per-probe median {eq.get('median_ms')} ms "
          f"p95 {eq.get('p95_ms')} ms; loaded-phase GPU busy "
          f"{result['phase_gpu_busy']['all_loaded_phases']['gpu_hours']} h")
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
