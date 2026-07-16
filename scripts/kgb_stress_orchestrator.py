"""Orchestrator for differential stress enhancement (runs ON the A800).

Spawns one isolated worker subprocess per mutant (so a CUDA illegal-memory
crash from an out-of-bounds mutant only kills that process). Crash / timeout
on a mutant whose original kernel is known-runnable is recorded as a
crash-kill (reported separately from differential kills). Resumable via a
JSONL checkpoint.

Layout (base dir = parent of this scripts/ dir):
  base/src/stress/policy_bank.py
  base/scripts/kgb_stress_worker.py
  base/stress/refmods/L0_P<id>.py
  base/stress/stress_work.json
  base/stress/worker_out/        (per-mutant tmp job+result)
  base/stress/results.jsonl      (checkpoint)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STRESS = os.path.join(BASE, "stress")
WORKER = os.path.join(BASE, "scripts", "kgb_stress_worker.py")
REFMODS = os.path.join(STRESS, "refmods")
WORK = os.path.join(STRESS, "stress_work.json")
JOBDIR = os.path.join(STRESS, "worker_out")
RESULTS = os.path.join(STRESS, "results.jsonl")
PY = sys.executable


def run_one(item: dict, timeout: int) -> dict:
    mid = item["id"]
    safe = mid.replace("/", "_")
    job_path = os.path.join(JOBDIR, f"{safe}.job.json")
    out_path = os.path.join(JOBDIR, f"{safe}.out.json")
    job = {
        "id": mid,
        "operator": item["operator"],
        "operator_name": item["operator_name"],
        "operator_category": item["operator_category"],
        "kernel_name": item["kernel_name"],
        "final_emd_status": item["final_emd_status"],
        "refmod": os.path.join(REFMODS, f"L0_P{item['problem_id']}.py"),
        "original_code": item["original_code"],
        "mutated_code": item["mutated_code"],
        "out": out_path,
    }
    json.dump(job, open(job_path, "w", encoding="utf-8"), ensure_ascii=False)
    if os.path.exists(out_path):
        os.remove(out_path)

    t0 = time.time()
    crashed = False
    timed_out = False
    try:
        p = subprocess.run([PY, WORKER, job_path], timeout=timeout,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if p.returncode != 0:
            crashed = True
    except subprocess.TimeoutExpired:
        timed_out = True

    if os.path.exists(out_path):
        try:
            res = json.load(open(out_path, encoding="utf-8"))
            res["_runtime_ms"] = round((time.time() - t0) * 1000, 1)
            return res
        except Exception:
            pass
    # no result file -> crash/timeout. original known-runnable (Phase1 survived),
    # so a mutant-induced crash/timeout counts as a kill, tracked separately.
    return {
        "mutant_id": mid, "operator_name": item["operator_name"],
        "operator_category": item["operator_category"], "kernel_name": item["kernel_name"],
        "final_emd_status": item["final_emd_status"],
        "any_killed": True, "first_kill_mode": "crash" if crashed else "timeout",
        "killed_dimensions": ["crash" if crashed else "timeout"],
        "main_track": {}, "config_track": {},
        "_crash_kill": True, "_runtime_ms": round((time.time() - t0) * 1000, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=150)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--status", default="", help="filter final_emd_status")
    args = ap.parse_args()

    os.makedirs(JOBDIR, exist_ok=True)
    work = json.load(open(WORK, encoding="utf-8"))
    if args.status:
        work = [w for w in work if w["final_emd_status"] == args.status]

    done = set()
    if os.path.exists(RESULTS):
        for line in open(RESULTS, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(json.loads(line)["mutant_id"])
                except Exception:
                    pass
    todo = [w for w in work if w["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"work={len(work)} done={len(done)} todo={len(todo)} "
          f"workers={args.workers} timeout={args.timeout}s", flush=True)

    n = killed = diff_killed = crash_killed = 0
    t_start = time.time()
    out_f = open(RESULTS, "a", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, w, args.timeout): w for w in todo}
        for fut in as_completed(futs):
            res = fut.result()
            out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
            out_f.flush()
            n += 1
            if res.get("any_killed"):
                killed += 1
                if res.get("_crash_kill"):
                    crash_killed += 1
                else:
                    diff_killed += 1
            if n % 10 == 0 or n == len(todo):
                rate = n / max(1e-9, time.time() - t_start)
                print(f"[{n}/{len(todo)}] killed={killed} (diff={diff_killed} "
                      f"crash={crash_killed}) {rate:.2f}/s "
                      f"eta={(len(todo)-n)/max(1e-9,rate):.0f}s", flush=True)
    out_f.close()
    print(f"DONE n={n} killed={killed} diff={diff_killed} crash={crash_killed} "
          f"elapsed={time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
