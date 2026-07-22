#!/usr/bin/env python3
"""Build the M-dir (mutakernel-directed) per-subject plans for E4 (CPU-only).

Closes the audited E4 gap "M-dir not integrated into the planner": consumes
the frozen FaultToStressMap (E1 --phase map output) plus each subject's
static site fingerprint, and derives the deterministic 70/30
directed/general case plan via src.validator.site_directed (already
unit-tested).  Output is one JSON per subject plus an aggregate artifact
that a strategy-matrix row can reference.

Freeze discipline (blueprint §5.1.1): the map MUST be the deployment-grade
map built on all of C1 and content-hashed *before* any C2-C5 subject is
executed; this script records the map digest into every plan.

Remaining integration TODO (documented, not hidden):
  * add a `mutakernel-directed` row to configs/fse_strategy_matrix.json that
    references these per-subject plans (the matrix schema currently assumes
    subject-independent case lists — extend protocol.build_experiment_plan
    to accept per-subject case streams, or emit one plan file per subject
    and register them as per-subject strategy parameters);
  * contract gate: this script uses assert_case_in_contract as the
    is_authorized callback when a contract is supplied with the subject.

Usage:
  python scripts/build_mdir_strategy_row.py \
      --map /root/mk_v2_runs/e1/fault_to_stress_map.json \
      --subjects subjects.jsonl --out-dir /root/mk_v2_runs/e4_mdir \
      [--budget 32] [--directed-fraction 0.7]

``subjects.jsonl``: one {"subject_id", "candidate_path"[, "contract"]} per
line; candidate_path is read for fingerprinting only (no execution).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--map", required=True, type=Path)
    ap.add_argument("--subjects", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--budget", type=int, default=32)
    ap.add_argument("--directed-fraction", type=float, default=0.7)
    args = ap.parse_args()

    from src.experiments.contract import ContractError, assert_case_in_contract, validate_contract
    from src.mutengine.fingerprint import build_site_fingerprint
    from src.validator.site_directed import derive_site_directed_plan

    map_bytes = args.map.read_bytes()
    fault_map = json.loads(map_bytes)
    map_sha256 = hashlib.sha256(map_bytes).hexdigest()

    plans_dir = args.out_dir / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)

    index = []
    with open(args.subjects, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            subject = json.loads(line)
            subject_id = subject["subject_id"]
            source = Path(subject["candidate_path"]).read_text(
                encoding="utf-8", errors="replace")
            fingerprint = build_site_fingerprint(source, subject_id)

            contract = subject.get("contract")
            if contract:
                contract = validate_contract(contract)

                def is_authorized(case, _contract=contract):
                    try:
                        assert_case_in_contract(case, _contract)
                        return True
                    except ContractError:
                        return False
            else:
                def is_authorized(case):
                    return True

            plan = derive_site_directed_plan(
                subject_id=subject_id,
                fingerprint=fingerprint,
                fault_to_stress_map=fault_map,
                budget_candidate_calls=args.budget,
                is_authorized=is_authorized,
                directed_fraction=args.directed_fraction,
            )
            plan["map_sha256"] = map_sha256
            plan_path = plans_dir / f"{subject_id}.json"
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            index.append({
                "subject_id": subject_id,
                "plan_path": str(plan_path),
                "directed_calls": plan["budget"]["directed_calls"],
                "general_calls": plan["budget"]["general_calls"],
                "skipped_unauthorized": len(plan.get("skipped_unauthorized", [])),
            })
            print(f"[mdir] {subject_id}: directed={plan['budget']['directed_calls']} "
                  f"general={plan['budget']['general_calls']}", flush=True)

    aggregate = {
        "created_at": _now(),
        "strategy_name": "mutakernel-directed",
        "map": {"path": str(args.map), "sha256": map_sha256,
                "map_version": fault_map.get("map_version")},
        "budget_candidate_calls": args.budget,
        "directed_fraction": args.directed_fraction,
        "subjects": index,
        "integration_todo": [
            "register a mutakernel-directed row in configs/fse_strategy_matrix.json",
            "extend protocol.build_experiment_plan for per-subject case streams",
        ],
    }
    (args.out_dir / "mdir_strategy_row.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({"subjects": len(index)}, indent=2))


if __name__ == "__main__":
    main()
