#!/usr/bin/env python3
"""One-shot runner for all four offline round-log reuse analyzers.

Interim (local example):
  python scripts/offline_reuse_all.py \
      --e1-dir "MutakernelV2/实验/重跑实验数据/离线分析/data/e1" \
      --out-dir "MutakernelV2/实验/重跑实验数据/离线分析/outputs"

Final rerun after CSE completion (remote A800; CPU-only, read-only over the
run directory, writes to the analysis dir):
  /root/miniconda3/bin/python /root/mk_v2_runs/e1/analysis/scripts/offline_reuse_all.py \
      --e1-dir /root/mk_v2_runs/e1 --out-dir /root/mk_v2_runs/e1/analysis \
      --cse-obs /root/mk_v2_runs/e1/cse_observations_lane*.jsonl --final
"""

from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import offline_a1_crossfit_closure  # noqa: E402
import offline_a2_sole_detector  # noqa: E402
import offline_a3_budget_recall  # noqa: E402
import offline_a4_cost_c1  # noqa: E402
from offline_reuse_lib import DEFAULT_SCOPE_INTERIM  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e1-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cse-obs", nargs="*", default=None,
                    help="CSE observation jsonl files or globs "
                         "(default: lanes 1+2 under --e1-dir)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--planned-cases", type=int, default=8)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()

    cse_files = None
    if args.cse_obs:
        cse_files = []
        for pattern in args.cse_obs:
            expanded = sorted(globmod.glob(str(pattern)))
            if expanded:
                cse_files.extend(Path(p) for p in expanded)
            else:
                cse_files.append(Path(pattern))

    scope = "final" if args.final else DEFAULT_SCOPE_INTERIM
    offline_a1_crossfit_closure.run(
        args.e1_dir, args.out_dir, cse_files, args.folds,
        args.planned_cases, scope)
    offline_a2_sole_detector.run(args.e1_dir, args.out_dir, cse_files, scope)
    offline_a3_budget_recall.run(args.e1_dir, args.out_dir, cse_files, scope)
    offline_a4_cost_c1.run(args.e1_dir, args.out_dir, cse_files, scope)
    print("OFFLINE_REUSE_ALL_DONE")


if __name__ == "__main__":
    main()
