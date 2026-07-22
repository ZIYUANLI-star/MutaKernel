#!/usr/bin/env python3
"""E1 counterexample-search falsification of LIKELY_EQUIVALENT probes (GPU).

Blueprint §5.5: "Counterexample search additionally falsified the equivalence
hypothesis for ? of ? probes the evidence pipeline had graded
LIKELY_EQUIVALENT".  This driver takes the LIKELY_EQUIVALENT set from
run_e1_probe_study --phase equiv and runs a strictly *stronger* search than
the first pass:

  * all 21 value stress policies (not just the operator-directed six),
  * more random rounds and more repeats per policy,
  * (TODO, listed in the run manifest) dtype/train/config/repeated dimension
    cases via scripts/_stress_worker.py once the audit stress phase lands.

Only a sound SPEC_VIOLATION-style FAIL from src.validation counts as a
falsification; tolerance-conforming bitwise divergence never does (the equiv
worker already enforces this).  Reuses the E0 execution pattern: serial,
checkpointed, classified INCONCLUSIVE, full trial evidence.

Usage:
  python scripts/run_e1_cse_falsify.py --out-dir /root/mk_v2_runs/e1 \
      --kernelbench-root KernelBench [--timeout 900]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.audit.inconclusive import classify_observation  # noqa: E402
from scripts.run_e1_probe_study import (  # noqa: E402
    BASELINE_PROTOCOL,
    _load_probe_files,
    _now,
    _resolve_problem_file,
    _write_worker_log,
    environment_fingerprint,
    run_worker,
)

FALSIFY_EQUIV_RUNS = 40
FALSIFY_BASE_SEED = 50000
FALSIFY_STRESS_REPEATS = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--kernelbench-root", required=True, type=Path)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    out = args.out_dir
    scratch = Path(tempfile.mkdtemp(prefix="e1cse_", dir=str(out)))
    obs_path = out / "cse_falsify_observations.jsonl"
    done_path = out / "cse_falsify_completed.json"
    completed = set(json.loads(done_path.read_text()) if done_path.exists() else [])

    from src.stress.policy_bank import get_all_policy_names
    all_policies = get_all_policy_names()

    likely = {}
    with open(out / "equiv_observations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("evidence_grade") == "LIKELY_EQUIVALENT":
                likely[row["probe_id"]] = row

    kernel_files = _load_probe_files(out)
    targets = [
        (kf, probe)
        for kf in kernel_files
        for probe in kf["probes"]
        if probe["probe_id"] in likely
    ]
    print(f"[{_now()}] cse-falsify: {len(targets)} LIKELY_EQUIVALENT targets, "
          f"{len(completed)} done", flush=True)

    manifest = {
        "phase": "cse_falsify",
        "run_id": f"e1-cse-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "started_at": _now(),
        "search": {
            "equiv_runs": FALSIFY_EQUIV_RUNS,
            "base_seed": FALSIFY_BASE_SEED,
            "stress_policies": all_policies,
            "stress_repeats": FALSIFY_STRESS_REPEATS,
        },
        "todo": [
            "dtype/train/config/repeated dimension cases via _stress_worker "
            "(audit stress phase); value-dimension-only until then",
        ],
        "environment": environment_fingerprint(),
    }
    (out / "cse_falsify_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    counter = Counter()
    for kf, probe in targets:
        probe_id = probe["probe_id"]
        if probe_id in completed:
            continue
        kernel = kf["kernel"]
        problem_file = _resolve_problem_file(
            args.kernelbench_root, kernel["level"], kernel["problem_id"])
        cfg = {
            "mode": "equiv",
            "mutant_id": probe_id,
            "problem_file": str(problem_file),
            "kernel_code": kf["kernel_source"],
            "mutated_code": probe["mutated_code"],
            "operator_name": probe["operator_name"],
            "device": BASELINE_PROTOCOL["device"],
            "equiv_runs": FALSIFY_EQUIV_RUNS,
            "base_seed": FALSIFY_BASE_SEED,
            "stress_policies": all_policies,
            "stress_repeats": FALSIFY_STRESS_REPEATS,
            "atol": BASELINE_PROTOCOL["atol"],
            "rtol": BASELINE_PROTOCOL["rtol"],
        }
        result, timed_out, wall_ms, so, se = run_worker(cfg, args.timeout, scratch)
        validation_status = "inconclusive" if timed_out else (
            (result or {}).get("validation_status", "inconclusive"))
        outcome = {
            "fail": "FALSIFIED",
            "pass": "STILL_LIKELY_EQUIVALENT",
        }.get(validation_status, "INCONCLUSIVE")
        record = {
            "probe_id": probe_id,
            "kernel": kernel["problem_name"],
            "operator_name": probe["operator_name"],
            "fault_class": probe["fault_class"],
            "outcome": outcome,
            "validation_status": validation_status,
            "inconclusive_class": (
                classify_observation(result or {}, timed_out=timed_out)
                if outcome == "INCONCLUSIVE" else None),
            "divergence": (result or {}).get("divergence"),
            "trials": (result or {}).get("trials"),
            "valid_rounds": (result or {}).get("valid_rounds"),
            "wall_ms": round(wall_ms, 1),
            "timed_out": timed_out,
            "finished_at": _now(),
        }
        if timed_out or result is None or se.strip():
            _write_worker_log(out, f"{probe_id}_cse", so, se)
        with open(obs_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        completed.add(probe_id)
        done_path.write_text(json.dumps(sorted(completed)), encoding="utf-8")
        counter[outcome] += 1
        if len(completed) % 10 == 0:
            print(f"[{_now()}] cse-falsify: {len(completed)}/{len(targets)} "
                  f"{dict(counter)}", flush=True)

    (out / "cse_falsify_summary.json").write_text(json.dumps({
        "finished_at": _now(),
        "targets": len(targets),
        "outcomes": dict(counter),
    }, indent=2), encoding="utf-8")
    print(f"[{_now()}] cse-falsify DONE: {dict(counter)}", flush=True)


if __name__ == "__main__":
    main()
