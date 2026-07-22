#!/usr/bin/env python3
"""B11: NVIDIA Compute Sanitizer wrapper (native mode, separately metered).

Runs each of the four sanitizer tools (memcheck / racecheck / synccheck /
initcheck) over a candidate-executing replay command and converts the text
report into structured alarm records.  B11 alarms cover memory/race
properties and are reported by alarm type, never merged with tolerance-based
verdicts (blueprint Table 2 / §5.3).

The subject program is a self-contained replay script (e.g. an FSE replay
bundle's command, or a minimal "load candidate, run get_inputs once"
driver).  This wrapper never interprets sanitizer output as PASS/FAIL of the
numerical contract — zero sanitizer errors is *not* a correctness proof.

GPU REQUIRED: do not run while E0 run5 owns the GPU.

Usage:
  python scripts/b11_compute_sanitizer.py --subject-id L1_P1-cand42 \
      --out-dir /root/mk_v2_runs/e3_b11 \
      -- /root/miniconda3/bin/python replay.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOOLS = ("memcheck", "racecheck", "synccheck", "initcheck")

_ERROR_SUMMARY = re.compile(r"ERROR SUMMARY: (\d+) error", re.IGNORECASE)
_ALARM_HEAD = re.compile(
    r"^========= (?!ERROR SUMMARY)(?!COMPUTE-SANITIZER)(.+)$", re.MULTILINE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_sanitizer() -> str:
    binary = shutil.which("compute-sanitizer")
    if binary:
        return binary
    fallback = "/usr/local/cuda/bin/compute-sanitizer"
    if Path(fallback).exists():
        return fallback
    raise FileNotFoundError("compute-sanitizer not found on PATH or in /usr/local/cuda/bin")


def run_tool(sanitizer: str, tool: str, command, timeout: int, log_path: Path):
    argv = [sanitizer, "--tool", tool, "--error-exitcode", "99", *command]
    t0 = time.time()
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
            env={**os.environ},
        )
        timed_out = False
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        exit_code = None
    wall_s = time.time() - t0
    log_path.write_text(f"ARGV: {argv}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n",
                        encoding="utf-8")

    combined = stdout + "\n" + stderr
    summary_match = _ERROR_SUMMARY.search(combined)
    error_count = int(summary_match.group(1)) if summary_match else None
    alarm_lines = [line.strip() for line in _ALARM_HEAD.findall(combined)]
    return {
        "tool": tool,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "wall_s": round(wall_s, 2),
        "error_count": error_count,
        "alarm_headline_sample": alarm_lines[:20],
        "status": (
            "inconclusive_timeout" if timed_out
            else "alarms" if (error_count or 0) > 0 or exit_code == 99
            else "clean" if error_count == 0 or exit_code == 0
            else "inconclusive"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject-id", required=True)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--tools", nargs="+", default=list(TOOLS), choices=TOOLS)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="replay command after '--'")
    args = ap.parse_args()

    command = [token for token in args.command if token != "--"]
    if not command:
        ap.error("provide the replay command after '--'")

    out = args.out_dir / args.subject_id
    out.mkdir(parents=True, exist_ok=True)
    sanitizer = find_sanitizer()

    results = []
    for tool in args.tools:
        result = run_tool(sanitizer, tool, command, args.timeout,
                          out / f"{tool}.log")
        results.append(result)
        print(f"[{_now()}] {args.subject_id} {tool}: {result['status']} "
              f"(errors={result['error_count']}, {result['wall_s']}s)", flush=True)

    record = {
        "subject_id": args.subject_id,
        "sanitizer": sanitizer,
        "command": command,
        "finished_at": _now(),
        "tools": results,
        "any_alarm": any(r["status"] == "alarms" for r in results),
        "note": "GPU-seconds metered separately from budget-matched rows",
    }
    (out / "b11_result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({t["tool"]: t["status"] for t in results}, indent=2))


if __name__ == "__main__":
    main()
