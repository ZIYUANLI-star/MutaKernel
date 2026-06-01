#!/usr/bin/env python3
"""Pre-run for Task B: collect train mode ref_ok + diff_summary.

For every (kernel, policy, seed) candidate in
``task_b_buggy_kernels_from_existing_data.json`` whose ``mode == train_value``,
call ``_stress_worker.py`` (mode=training_stress) to get the authoritative
``ref_ok`` and ``diff_summary``. Filter to ``ref_ok=True AND original_ok=False``
and write the resulting events to ``task_b_regenerate/_train_refok_supplement.json``.

Run AFTER Task C finishes (the GPU is now free).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORKER = SCRIPT_DIR / "_stress_worker.py"
PY_INTERP = sys.executable

KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
}

BUGGY_JSON = (PROJECT_ROOT / "第二次实验汇总" / "第二次实验汇总_补充" /
              "task_b_buggy_kernels_from_existing_data.json")
BEST_KERNELS_JSON = PROJECT_ROOT / "best_kernels.json"
OUT_DIR = (PROJECT_ROOT / "第二次实验汇总" / "第二次实验汇总_补充" /
           "task_b_regenerate")
OUT_FILE = OUT_DIR / "_train_refok_supplement.json"

DEFAULT_TIMEOUT = 180
ATOL = 1e-2
RTOL = 1e-2


def find_problem_file(level: str, problem_id) -> Path | None:
    pid = str(problem_id)
    pdir = PROBLEM_DIRS.get(level)
    if pdir is None or not pdir.exists():
        return None
    for f in pdir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def run_one(cfg: Dict[str, Any], timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Spawn _stress_worker.py with mode=training_stress, return result dict."""
    with tempfile.TemporaryDirectory(prefix="taskb_supp_") as td:
        cfg_path = os.path.join(td, "cfg.json")
        res_path = os.path.join(td, "res.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        try:
            proc = subprocess.run(
                [PY_INTERP, "-u", str(WORKER), cfg_path, res_path],
                cwd=str(PROJECT_ROOT),
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                    "error": f"TIMEOUT({timeout}s)"}
        try:
            with open(res_path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                    "error": f"worker no output (rc={proc.returncode}): "
                             f"{proc.stderr[:200]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-kernel", default="", help="run only this kernel name")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    ap.add_argument("--resume", action="store_true",
                    help="skip events already present in OUT_FILE")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    buggy = json.loads(BUGGY_JSON.read_text(encoding="utf-8"))
    best = json.loads(BEST_KERNELS_JSON.read_text(encoding="utf-8"))

    existing: Dict[str, Dict[str, Any]] = {}
    if args.resume and OUT_FILE.exists():
        try:
            saved = json.loads(OUT_FILE.read_text(encoding="utf-8"))
            for k, evs in saved.get("supplemented", {}).items():
                existing[k] = {(e["policy"], e["seed"]): e for e in evs}
        except Exception:
            existing = {}

    supplemented: Dict[str, List[Dict[str, Any]]] = {}
    stats = {"kernels_attempted": 0, "events_total": 0,
             "events_ref_ok_true": 0, "events_buggy": 0,
             "events_ref_failed": 0, "events_orig_ok": 0,
             "events_skipped_resume": 0, "events_errored": 0,
             "elapsed_sec": 0.0}
    t_start = time.time()

    targets = []
    for kname, events in buggy["buggy_kernels"].items():
        if args.only_kernel and kname != args.only_kernel:
            continue
        train_events = [e for e in events
                        if e.get("mode") == "train_candidate_no_refok"]
        if not train_events:
            continue
        targets.append((kname, train_events))

    print(f"[INIT] {len(targets)} kernels with train candidates to verify",
          flush=True)
    for kname, evs in targets:
        print(f"  {kname}: {len(evs)} candidate event(s)", flush=True)

    for kname, train_events in targets:
        meta = best.get(kname)
        if meta is None:
            print(f"[SKIP] {kname}: not in best_kernels.json", flush=True)
            continue
        problem_file = find_problem_file(meta["level"], meta["problem_id"])
        if problem_file is None:
            print(f"[SKIP] {kname}: no problem_file for {meta['level']}/"
                  f"P{meta['problem_id']}", flush=True)
            continue
        kpath = Path(meta["kernel_path"])
        if not kpath.exists():
            print(f"[SKIP] {kname}: kernel_path missing: {kpath}", flush=True)
            continue
        kernel_code = kpath.read_text(encoding="utf-8")

        stats["kernels_attempted"] += 1
        kept: List[Dict[str, Any]] = []

        print(f"\n=== {kname} ({meta['level']}/P{meta['problem_id']}, "
              f"{len(train_events)} train events) ===", flush=True)

        for idx, ev in enumerate(train_events, 1):
            stats["events_total"] += 1
            pol, seed = ev["policy"], ev["seed"]
            tag = f"({pol}, seed={seed})"

            if kname in existing and (pol, seed) in existing[kname]:
                kept.append(existing[kname][(pol, seed)])
                stats["events_skipped_resume"] += 1
                if existing[kname][(pol, seed)].get("ref_ok") and \
                   not existing[kname][(pol, seed)].get("original_ok"):
                    stats["events_buggy"] += 1
                print(f"  [{idx}/{len(train_events)}] {tag} -- resume "
                      f"(was buggy={not existing[kname][(pol,seed)].get('original_ok')})",
                      flush=True)
                continue

            cfg = {
                "mode": "training_stress",
                "device": "cuda",
                "atol": ATOL, "rtol": RTOL,
                "policy_name": pol, "seed": seed,
                "kernel_code": kernel_code,
                "mutated_code": kernel_code,
                "problem_file": str(problem_file),
                "sync_weights": True,
            }
            t0 = time.time()
            result = run_one(cfg, timeout=args.timeout)
            elapsed = time.time() - t0

            ref_ok = bool(result.get("ref_ok"))
            orig_ok = bool(result.get("original_ok"))
            diff = result.get("diff_summary", "")
            err = result.get("error", "")

            if ref_ok: stats["events_ref_ok_true"] += 1
            else: stats["events_ref_failed"] += 1
            if orig_ok: stats["events_orig_ok"] += 1
            if not ref_ok and not orig_ok: stats["events_errored"] += 1

            record = {
                "policy": pol, "seed": seed, "mode": "train_candidate_no_refok",
                "ref_ok": ref_ok, "original_ok": orig_ok,
                "diff_summary": diff, "error": err,
                "time_ms": result.get("time_ms"),
                "elapsed_sec": round(elapsed, 2),
            }
            kept.append(record)

            tag2 = ("★BUGGY" if (ref_ok and not orig_ok)
                    else ("ref_fail" if not ref_ok
                          else ("ok-no-bug" if orig_ok else "?")))
            if ref_ok and not orig_ok:
                stats["events_buggy"] += 1
            print(f"  [{idx}/{len(train_events)}] {tag} [{elapsed:.1f}s] "
                  f"ref_ok={ref_ok} orig_ok={orig_ok} {tag2}"
                  f"{' diff=' + diff[:80] if diff else ''}"
                  f"{' err=' + err[:80] if err else ''}", flush=True)

            try:
                OUT_FILE.write_text(json.dumps({
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "atol": ATOL, "rtol": RTOL,
                    "stats": stats,
                    "supplemented": {**supplemented, kname: kept},
                }, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

        supplemented[kname] = kept

    stats["elapsed_sec"] = round(time.time() - t_start, 2)
    OUT_FILE.write_text(json.dumps({
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "atol": ATOL, "rtol": RTOL,
        "stats": stats,
        "supplemented": supplemented,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[DONE] {stats['events_total']} events verified, "
          f"{stats['events_buggy']} confirmed buggy (ref_ok=True ∧ orig_ok=False)",
          flush=True)
    print(f"  Output: {OUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
