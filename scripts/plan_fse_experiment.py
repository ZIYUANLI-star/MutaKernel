#!/usr/bin/env python3
"""Create an immutable, compute-matched FSE experiment plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.protocol import ProtocolError, plan_from_files, write_plan_once


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", required=True, type=Path, help="Subject manifest JSON")
    parser.add_argument(
        "--strategy-matrix",
        type=Path,
        default=PROJECT_ROOT / "configs" / "fse_strategy_matrix.json",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        plan = plan_from_files(args.subjects, args.strategy_matrix)
        write_plan_once(args.output, plan)
    except FileExistsError:
        print(f"refusing to overwrite existing experiment plan: {args.output}", file=sys.stderr)
        return 2
    except (OSError, ProtocolError, TypeError, ValueError) as exc:
        print(f"failed to plan FSE experiment: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plan_sha256": plan["plan_sha256"],
                "test_case_count": plan["test_case_count"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
