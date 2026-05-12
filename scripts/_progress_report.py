"""Quick analysis of Task A / Task C live progress.

For each completed mutant detail, report:
  - killed yes/no
  - max round reached
  - whether LLM ever judged "killable=true" (i.e. actually tried inputs)
  - reason_category distribution
"""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKA_DIR = ROOT / "第二次实验汇总" / "第二次实验汇总_补充" / "task_a_phase2_rerun" / "details"
TASKC_DIR = ROOT / "第二次实验汇总" / "第二次实验汇总_补充" / "task_c_phase1_direct" / "details"


def analyze(name: str, details_dir: Path) -> None:
    files = sorted(details_dir.glob("*.json")) if details_dir.exists() else []
    print(f"\n{'='*70}")
    print(f"  {name}   completed={len(files)}")
    print(f"{'='*70}")
    if not files:
        return

    killed_ids = []
    max_round_dist = Counter()
    ever_tried_input = 0  # at least one round had killable=true
    reason_cats = Counter()
    total_rounds = 0
    deepseek_baseline_killed = 0  # Task A only

    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("killed"):
            killed_ids.append((d["mutant_id"], d.get("killing_round")))
        rounds = d.get("rounds") or []
        if rounds:
            max_round_dist[len(rounds)] += 1
            total_rounds += len(rounds)
            tried = any(r.get("killable") is True for r in rounds)
            if tried:
                ever_tried_input += 1
            for r in rounds:
                cat = r.get("reason_category")
                if cat:
                    reason_cats[cat] += 1
        baseline = d.get("deepseek_baseline") or {}
        if baseline.get("killed"):
            deepseek_baseline_killed += 1

    print(f"  Killed by Opus 4.5     : {len(killed_ids)} / {len(files)}")
    for mid, kr in killed_ids:
        print(f"      -> {mid}  (killing_round={kr})")
    print(f"  Ever tried an input    : {ever_tried_input} / {len(files)}  "
          f"(rest gave killable=false on round 1, early-stopped)")
    print(f"  Avg rounds per mutant  : {total_rounds/len(files):.2f}")
    print(f"  Round distribution     : {dict(sorted(max_round_dist.items()))}")
    print(f"  reason_category counts : {dict(reason_cats.most_common())}")
    if deepseek_baseline_killed:
        print(f"  (Task A only) DeepSeek baseline killed: "
              f"{deepseek_baseline_killed} of same set")


analyze("Task A", TASKA_DIR)
analyze("Task C", TASKC_DIR)
