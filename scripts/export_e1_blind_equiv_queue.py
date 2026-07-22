#!/usr/bin/env python3
"""Export the blinded equivalence-audit queue for E1 (CPU-only).

Blueprint §5.5: probes that remain equivalence-uncertain after the evidence
pipeline (grade LIKELY_EQUIVALENT or INCONCLUSIVE) go to a *blinded* human
equivalence audit.  Annotators see the original/mutated source pair under a
neutral ID; they never see the operator name, fault class, injection site
metadata, historical verdict, or any dynamic-evidence grade — those all go
into the sealed mapping (integrity-hashed) for post-unblinding analysis.

Input:  an E1 run directory produced by run_e1_probe_study (probes/ +
        equiv_observations.jsonl).
Output: <output-dir>/blind_equiv_queue/<neutral_id>/{pair.json,original.py,
        mutated.py} and <output-dir>/sealed/equiv_mapping.json(.sha256).

Usage:
  python scripts/export_e1_blind_equiv_queue.py --e1-dir /root/mk_v2_runs/e1 \
      --output-dir /root/mk_v2_runs/e1_blind --salt <secret>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def neutral_id(probe_id: str, salt: str) -> str:
    return "eq-" + hashlib.sha256(f"{salt}|{probe_id}".encode("utf-8")).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e1-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--salt", required=True,
                    help="secret salt; store only with the sealed mapping")
    ap.add_argument("--grades", nargs="+",
                    default=["LIKELY_EQUIVALENT", "INCONCLUSIVE"])
    args = ap.parse_args()

    grades = {}
    with open(args.e1_dir / "equiv_observations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            grades[row["probe_id"]] = row["evidence_grade"]

    queue_dir = args.output_dir / "blind_equiv_queue"
    sealed_dir = args.output_dir / "sealed"
    queue_dir.mkdir(parents=True, exist_ok=True)
    sealed_dir.mkdir(parents=True, exist_ok=True)

    mapping = {}
    exported = 0
    for probe_file in sorted((args.e1_dir / "probes").glob("*.json")):
        data = json.loads(probe_file.read_text(encoding="utf-8"))
        for probe in data["probes"]:
            grade = grades.get(probe["probe_id"])
            if grade not in args.grades:
                continue
            nid = neutral_id(probe["probe_id"], args.salt)
            target = queue_dir / nid
            target.mkdir(parents=True, exist_ok=True)
            (target / "original.py").write_text(
                data["kernel_source"], encoding="utf-8")
            (target / "mutated.py").write_text(
                probe["mutated_code"], encoding="utf-8")
            (target / "pair.json").write_text(json.dumps({
                "blind_schema_version": "1.0",
                "neutral_id": nid,
                "language": data["kernel"].get("language", "cuda"),
                "task": (
                    "Decide whether mutated.py is semantically equivalent to "
                    "original.py for all in-contract inputs.  Labels: "
                    "EQUIVALENT / NOT_EQUIVALENT (attach reasoning or a "
                    "counterexample sketch) / CANNOT_DECIDE."
                ),
            }, indent=2), encoding="utf-8")
            mapping[nid] = {
                "probe_id": probe["probe_id"],
                "kernel": data["kernel"]["problem_name"],
                "operator_name": probe["operator_name"],
                "operator_category": probe["operator_category"],
                "fault_class": probe["fault_class"],
                "site": probe["site"],
                "evidence_grade": grade,
                "historical_status": probe["historical_status"],
            }
            exported += 1

    payload = json.dumps(mapping, indent=2, sort_keys=True)
    (sealed_dir / "equiv_mapping.json").write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (sealed_dir / "equiv_mapping.sha256").write_text(digest + "\n", encoding="utf-8")
    print(json.dumps({"exported": exported, "mapping_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
