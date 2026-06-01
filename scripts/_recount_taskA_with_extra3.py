#!/usr/bin/env python3
"""Recount Task A audit results across all 368 mutants
(365 original + 3 supplement). Produces the numbers that
feed RQ3 in the paper.

Definitions (per the paper):
- provably_equivalent: every recorded round (with non-null killable)
  returns killable=False.
- operationally_indistinguishable: at least one round has killable=True
  yet no executed kill succeeded across all rounds.
- ourtool-missed: at least one round produced a verified kill
  (round.killed == True), i.e. Opus found an in-contract killing input
  that MutaKernel's stress dimensions failed to discover.

Round-level reason_category counts only rounds whose LLM response
parsed successfully and emitted a reason_category label. (parse errors
and prompt-build errors are excluded.)
"""
from __future__ import annotations
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DETAILS = (ROOT / "第二次实验汇总" / "第二次实验汇总_补充" /
           "task_a_phase2_rerun" / "details")


def classify_mutant(d: dict) -> str:
    rounds = d.get("rounds", []) or []
    # Any verified kill -> ourtool-missed.
    for r in rounds:
        if r.get("killed") is True:
            return "ourtool_missed"
        ex = r.get("execution_result")
        if isinstance(ex, dict) and ex.get("killed") is True:
            return "ourtool_missed"
    # No kill: look at killable verdicts across rounds.
    killables = [r.get("killable") for r in rounds
                 if r.get("killable") is not None]
    if not killables:
        return "no_verdict"
    if any(k is True for k in killables):
        return "operationally_indistinguishable"
    return "provably_equivalent"


def main():
    files = sorted(DETAILS.glob("*.json"))
    print(f"Total detail files: {len(files)}")

    by_class = collections.Counter()
    by_class_tier = collections.defaultdict(collections.Counter)
    reason_cat_round = collections.Counter()
    reason_cat_mutant_first = collections.Counter()
    killing_round_dist = collections.Counter()
    parse_errors = 0
    rounds_total = 0
    rounds_with_label = 0
    rounds_killable_true = 0
    rounds_killable_false = 0
    supplement_ids = []

    # tier-level mutant-killed lookup not available here; we just bucket
    # by tier reported in detail file.
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] parse fail {f.name}: {e}")
            continue

        cls = classify_mutant(d)
        tier = d.get("tier", "?")
        by_class[cls] += 1
        by_class_tier[cls][tier] += 1

        if d.get("supplement_run"):
            supplement_ids.append((d.get("mutant_id"), cls, tier,
                                   d.get("killing_round")))

        first_label = None
        for r in d.get("rounds", []) or []:
            rounds_total += 1
            if r.get("survival_reason") == "parse_error":
                parse_errors += 1
                continue
            rc = r.get("reason_category")
            if rc:
                reason_cat_round[rc] += 1
                rounds_with_label += 1
                if first_label is None:
                    first_label = rc
            k = r.get("killable")
            if k is True:
                rounds_killable_true += 1
            elif k is False:
                rounds_killable_false += 1
        if first_label:
            reason_cat_mutant_first[first_label] += 1

        if d.get("killed"):
            killing_round_dist[d.get("killing_round", 0)] += 1

    print("\n=== Mutant-level classification (n={}) ===".format(sum(by_class.values())))
    for k in ("provably_equivalent", "operationally_indistinguishable",
              "ourtool_missed", "no_verdict"):
        print(f"  {k:35s} {by_class.get(k,0):4d}")

    print("\n=== Mutant-level by tier ===")
    for cls in ("provably_equivalent", "operationally_indistinguishable",
                "ourtool_missed", "no_verdict"):
        if not by_class_tier[cls]:
            continue
        tiers = dict(sorted(by_class_tier[cls].items(),
                            key=lambda x: str(x[0])))
        print(f"  {cls}: {tiers}")

    print("\n=== Round-level reason_category (rounds with label) ===")
    print(f"  total rounds (any): {rounds_total}")
    print(f"  parse_error rounds: {parse_errors}")
    print(f"  rounds with reason_category label: {rounds_with_label}")
    for k, v in reason_cat_round.most_common():
        print(f"    {k:35s} {v:4d}")

    print("\n=== Round-level killable verdicts (rounds with non-null) ===")
    print(f"  killable=true:  {rounds_killable_true}")
    print(f"  killable=false: {rounds_killable_false}")

    print("\n=== Mutant-level FIRST round reason_category ===")
    for k, v in reason_cat_mutant_first.most_common():
        print(f"    {k:35s} {v:4d}")

    print("\n=== Killing round distribution (for ourtool-missed) ===")
    for k in sorted(killing_round_dist):
        print(f"  round {k}: {killing_round_dist[k]}")

    print("\n=== Supplement (3 mutants) breakdown ===")
    for mid, cls, tier, kr in supplement_ids:
        print(f"  {mid:35s} class={cls:35s} tier={tier} killing_round={kr}")


if __name__ == "__main__":
    main()
