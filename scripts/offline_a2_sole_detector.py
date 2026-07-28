#!/usr/bin/env python3
"""Offline analyzer 2 — Table 11 sole-detector column (offline part) and the
§5.6 ">=2 dimensions cross-confirmation" rate.

Maps the 21 value-stress policies onto the five blueprint dimensions
(value / dtype / training / repetition / configuration) via the documented
proxy in offline_reuse_lib.POLICY_DIMENSION_PROXY (basis: target fault
classes of src/stress/policy_metadata.py matched against the execution-
context dimension targets), and reports the strict RIPR accounting alongside
(under which every executed E1 case is dimension "value").

Usage:
  python scripts/offline_a2_sole_detector.py --e1-dir <dir> --out-dir <dir>
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
    provenance,
    witness_dimension_summary,
    write_csv,
    write_json,
)


def run(e1_dir: Path, out_dir: Path, cse_files, scope_label: str) -> dict:
    dataset = load_dataset(e1_dir, cse_files)
    witnesses = extract_witnesses(dataset["equiv"], dataset["cse"])
    result = witness_dimension_summary(witnesses)
    result["provenance"] = provenance(dataset, scope_label)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "a2_sole_detector.json", result)
    write_csv(
        out_dir / "a2_witness_dimensions.csv",
        ["probe_id", "fault_class", "source", "witness_policy",
         "dimension_proxy", "dimension_strict"],
        [
            (r["probe_id"], r["fault_class"], r["source"], r["witness_policy"],
             "|".join(r["dimensions_proxy"]), "|".join(r["dimensions_strict"]))
            for r in result["per_defect"]
        ])

    proxy = result["proxy"]
    print(f"a2 defects={proxy['defects_total']} "
          f"sole-by-dimension(proxy)={proxy['sole_detector_defects']} "
          f"cross>=2dims={proxy['cross_confirmed_ge2_dims']} "
          f"({proxy['cross_confirmed_rate']})")
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
