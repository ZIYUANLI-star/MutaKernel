#!/usr/bin/env python3
"""B11: NVIDIA Compute Sanitizer wrapper (native mode, separately metered).

Runs each of the four sanitizer tools (memcheck / racecheck / synccheck /
initcheck) over a candidate-executing replay command and converts the text
report into structured alarm records, bucketed by alarm type.  B11 alarms
cover memory/race properties and are reported by alarm type, never merged
with tolerance-based verdicts (blueprint Table 2 / §5.3).

The subject program is a self-contained replay script (e.g. an FSE replay
bundle's command, or a minimal "load candidate, run get_inputs once"
driver).  This wrapper never interprets sanitizer output as PASS/FAIL of the
numerical contract — zero sanitizer errors is *not* a correctness proof.

Parsing (:func:`parse_sanitizer_report`) is pure and unit-tested CPU-only;
only the CLI entry point needs a GPU.  GPU REQUIRED for actual runs: do not
run while E1 owns the GPU.

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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

TOOLS = ("memcheck", "racecheck", "synccheck", "initcheck")

_ERROR_SUMMARY = re.compile(r"ERROR SUMMARY: (\d+) error", re.IGNORECASE)
_RACECHECK_SUMMARY = re.compile(
    r"RACECHECK SUMMARY: (\d+) hazard", re.IGNORECASE)
_LEAK_SUMMARY = re.compile(
    r"LEAK SUMMARY: (\d+) bytes? leaked", re.IGNORECASE)
_ALARM_HEAD = re.compile(
    r"^========= (?!ERROR SUMMARY)(?!RACECHECK SUMMARY)(?!LEAK SUMMARY)"
    r"(?!COMPUTE-SANITIZER)(?!Saved host backtrace)(?!Host Frame)"
    r"(?!\s)(.+)$",
    re.MULTILINE)

# Alarm-type buckets (blueprint Table 2: "alarm types reported separately").
# Order matters: the first matching pattern names the bucket.
_ALARM_CATEGORIES = (
    ("invalid_memory_access", re.compile(
        r"Invalid __(global|shared|local)__|Invalid global|Invalid shared|"
        r"Invalid local|out-of-bounds|Address .* is out of bounds|"
        r"Invalid atomic|misaligned", re.IGNORECASE)),
    ("memory_leak", re.compile(r"leaked|leak of", re.IGNORECASE)),
    ("uninitialized_read", re.compile(r"uninitialized", re.IGNORECASE)),
    ("race_hazard", re.compile(
        r"race reported|hazard detected|(RAW|WAR|WAW) hazard", re.IGNORECASE)),
    ("barrier_sync_error", re.compile(
        r"barrier error|divergent thread|invalid barrier|"
        r"deadlock", re.IGNORECASE)),
    ("api_error", re.compile(
        r"program hit|API error|invalid argument|driver api", re.IGNORECASE)),
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_alarm(headline: str) -> str:
    for category, pattern in _ALARM_CATEGORIES:
        if pattern.search(headline):
            return category
    return "other"


def parse_sanitizer_report(text: str) -> Dict:
    """Parse one compute-sanitizer text report into structured alarms.

    Returns error/hazard counts from the summary lines, the alarm headlines
    (`========= <headline>` blocks), and a per-alarm-type histogram.  A
    report with no summary line at all is marked ``summary_found=False`` so
    the caller can classify the run as inconclusive rather than clean.
    """

    error_match = _ERROR_SUMMARY.search(text)
    race_match = _RACECHECK_SUMMARY.search(text)
    leak_match = _LEAK_SUMMARY.search(text)

    error_count: Optional[int] = None
    if error_match:
        error_count = int(error_match.group(1))
    hazard_count = int(race_match.group(1)) if race_match else None
    leaked_bytes = int(leak_match.group(1)) if leak_match else None

    alarms: List[Dict[str, str]] = []
    for raw in _ALARM_HEAD.findall(text):
        headline = raw.strip()
        if not headline:
            continue
        alarms.append({"headline": headline, "category": classify_alarm(headline)})

    by_category: Dict[str, int] = {}
    for alarm in alarms:
        by_category[alarm["category"]] = by_category.get(alarm["category"], 0) + 1

    total = 0
    for value in (error_count, hazard_count):
        if value:
            total += value
    return {
        "summary_found": bool(error_match or race_match or leak_match),
        "error_count": error_count,
        "hazard_count": hazard_count,
        "leaked_bytes": leaked_bytes,
        "total_reported": total,
        "alarms": alarms,
        "by_category": by_category,
    }


def report_status(parsed: Dict, exit_code: Optional[int], timed_out: bool) -> str:
    """Map one tool run onto {alarms, clean, inconclusive_*} statuses."""

    if timed_out:
        return "inconclusive_timeout"
    if parsed["total_reported"] > 0 or exit_code == 99 or parsed["alarms"]:
        return "alarms"
    if parsed["summary_found"] and parsed["total_reported"] == 0:
        return "clean"
    if exit_code == 0:
        # Sanitizer ran the program but printed no summary (e.g. no kernels
        # were launched); the program itself succeeded.
        return "clean_no_kernel_activity"
    return "inconclusive"


def find_sanitizer() -> str:
    binary = shutil.which("compute-sanitizer")
    if binary:
        return binary
    fallback = "/usr/local/cuda/bin/compute-sanitizer"
    if Path(fallback).exists():
        return fallback
    raise FileNotFoundError(
        "compute-sanitizer not found on PATH or in /usr/local/cuda/bin")


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

    parsed = parse_sanitizer_report(stdout + "\n" + stderr)
    return {
        "tool": tool,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "wall_s": round(wall_s, 2),
        "error_count": parsed["error_count"],
        "hazard_count": parsed["hazard_count"],
        "leaked_bytes": parsed["leaked_bytes"],
        "by_category": parsed["by_category"],
        "alarm_headline_sample": [a["headline"] for a in parsed["alarms"][:20]],
        "status": report_status(parsed, exit_code, timed_out),
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
              f"(errors={result['error_count']}, hazards={result['hazard_count']}, "
              f"{result['wall_s']}s)", flush=True)

    merged_by_category: Dict[str, int] = {}
    for result in results:
        for category, count in result["by_category"].items():
            merged_by_category[category] = merged_by_category.get(category, 0) + count

    record = {
        "subject_id": args.subject_id,
        "sanitizer": sanitizer,
        "command": command,
        "finished_at": _now(),
        "tools": results,
        "any_alarm": any(r["status"] == "alarms" for r in results),
        "alarms_by_category": merged_by_category,
        "note": "GPU-seconds metered separately from budget-matched rows",
    }
    (out / "b11_result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({t["tool"]: t["status"] for t in results}, indent=2))


if __name__ == "__main__":
    main()
