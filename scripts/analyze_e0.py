#!/usr/bin/env python3
"""Aggregate E0 flip-rerun observations into the Table-3 strata.

Usage: python scripts/analyze_e0.py <run_dir>
Reads observations.jsonl + original_controls.json, prints the stratified
flip table (paired legacy-vs-corrected as primary; corrected-vs-historical
as secondary) and writes table3_summary.json next to the observations.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main():
    run_dirs = [Path(p) for p in sys.argv[1:]]
    records = []
    seen = set()
    for run_dir in run_dirs:
        obs = run_dir / "observations.jsonl"
        if not obs.exists():
            continue
        for line in obs.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["probe_id"] in seen:
                continue
            seen.add(r["probe_id"])
            records.append(r)
    run_dir = run_dirs[0]

    strata = defaultdict(lambda: {
        "probes": 0, "excluded_control": 0, "inconclusive": 0,
        "paired_n": 0, "paired_flips": 0,
        "k2s": 0, "s2k": 0,           # paired direction (legacy->corrected)
        "hist_n": 0, "hist_flips": 0,
    })

    def stratum_of(r):
        stateful = r.get("reference_stateful")
        s = {True: "stateful", False: "stateless", None: "unknown"}[stateful]
        return (f"L{r['level']}", s)

    for r in records:
        key = stratum_of(r)
        st = strata[key]
        st["probes"] += 1
        if not r.get("original_control_ok"):
            st["excluded_control"] += 1
            continue
        if r.get("flip_paired") is None:
            st["inconclusive"] += 1
        else:
            st["paired_n"] += 1
            if r["flip_paired"]:
                st["paired_flips"] += 1
                leg = r["legacy"]["status"]
                st["k2s" if leg == "killed" else "s2k"] += 1
        if r.get("flip_vs_historical") is not None:
            st["hist_n"] += 1
            if r["flip_vs_historical"]:
                st["hist_flips"] += 1

    def emit(rows):
        header = (f"{'Stratum':28s} {'probes':>6} {'excl':>5} {'inc':>4} "
                  f"{'paired n':>8} {'flips':>5} {'rate':>7} {'95% CI':>16} "
                  f"{'k->s':>4} {'s->k':>4} {'hist n':>6} {'hist flips':>10}")
        print(header)
        print("-" * len(header))
        for name, st in rows:
            lo, hi = wilson_ci(st["paired_flips"], st["paired_n"])
            rate = st["paired_flips"] / st["paired_n"] * 100 if st["paired_n"] else 0.0
            print(f"{name:28s} {st['probes']:>6} {st['excluded_control']:>5} "
                  f"{st['inconclusive']:>4} {st['paired_n']:>8} {st['paired_flips']:>5} "
                  f"{rate:>6.1f}% [{lo*100:5.1f},{hi*100:5.1f}]% "
                  f"{st['k2s']:>4} {st['s2k']:>4} {st['hist_n']:>6} {st['hist_flips']:>10}")

    rows = sorted(strata.items(), key=lambda kv: str(kv[0]))
    named = [(f"{lvl}, {s} reference", st) for (lvl, s), st in rows]
    overall = defaultdict(int)
    for _, st in rows:
        for k, v in st.items():
            overall[k] += v
    named.append(("Overall", dict(overall)))
    emit(named)

    summary = {
        "strata": {f"{lvl}|{s}": st for (lvl, s), st in rows},
        "overall": dict(overall),
        "records": len(records),
    }
    (run_dir / "table3_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwritten: {run_dir / 'table3_summary.json'}")


if __name__ == "__main__":
    main()
