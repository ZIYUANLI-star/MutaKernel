#!/usr/bin/env python3
"""r-o differential probe — main driver.

For every kernel in best_kernels.json, spawn a fresh subprocess that:
  - loads ref + orig once
  - runs each (stress policy, seed) in (.eval(), .train()) modes
  - records ref_ok / orig_ok / nan_inf flags + diff_summary
  - writes per-kernel JSON to <out_dir>/details/<kernel_name>.json

Then aggregates into:
  <out_dir>/buggy_kernels.json    — kernels with at least 1 orig_ok=False event
  <out_dir>/buggy_kernels_report.md
  <out_dir>/run_manifest.json

This does NOT touch existing Phase II data.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"

# Map level → KernelBench problem directory (WSL paths)
KB_ROOT = Path("/home/kbuser/projects/KernelBench-0")
PROBLEM_DIRS = {
    "L1": KB_ROOT / "KernelBench" / "level1",
    "L2": KB_ROOT / "KernelBench" / "level2",
    "L3": KB_ROOT / "KernelBench" / "level3",
}

DEFAULT_OUT = (PROJECT_ROOT / "第二次实验汇总_补充" / "diff_probe_phase2_supp")


def find_problem_file(problem_dir: Path, problem_id) -> Path | None:
    pid = str(problem_id)
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def load_targets(only: set[str] | None) -> list[dict]:
    with open(BEST_KERNELS_FILE, encoding="utf-8") as f:
        best = json.load(f)
    targets = []
    for kname, meta in best.items():
        if only and kname not in only:
            continue
        level_key = meta.get("level", "")
        if not level_key.startswith("L"):
            level_key = f"L{level_key}"
        pdir = PROBLEM_DIRS.get(level_key)
        if pdir is None:
            print(f"[SKIP] {kname}: unknown level {meta.get('level')}",
                  flush=True)
            continue
        pfile = find_problem_file(pdir, meta["problem_id"])
        if pfile is None:
            print(f"[SKIP] {kname}: problem file not found", flush=True)
            continue
        kpath = Path(meta["kernel_path"])
        if not kpath.exists():
            print(f"[SKIP] {kname}: kernel file missing {kpath}", flush=True)
            continue
        targets.append({
            "kernel_name": kname,
            "level": level_key,
            "problem_id": meta["problem_id"],
            "turn": meta.get("turn"),
            "speedup": meta.get("speedup"),
            "problem_file": str(pfile),
            "kernel_path": str(kpath),
        })
    return targets


def run_one(target: dict, out_dir: Path, *,
            policies: list[str], seeds: list[int],
            include_training: bool, timeout_sec: int,
            device: str, atol: float, rtol: float) -> dict:
    kname = target["kernel_name"]
    detail_path = out_dir / "details" / f"{kname}.json"
    detail_path.parent.mkdir(parents=True, exist_ok=True)

    with open(target["kernel_path"], encoding="utf-8") as f:
        kernel_code = f.read()

    cfg = {
        "kernel_name": kname,
        "kernel_code": kernel_code,
        "problem_file": target["problem_file"],
        "policies": policies,
        "seeds": seeds,
        "include_training": include_training,
        "device": device,
        "atol": atol,
        "rtol": rtol,
    }
    cfg_path = out_dir / "_tmp" / f"cfg_{kname}.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)

    worker = SCRIPT_DIR / "_diff_probe_worker.py"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(worker), str(cfg_path), str(detail_path)],
            capture_output=True, text=True, timeout=timeout_sec,
        )
        ok = proc.returncode == 0
        stderr_tail = proc.stderr[-1000:] if proc.stderr else ""
        stdout_tail = proc.stdout[-800:] if proc.stdout else ""
    except subprocess.TimeoutExpired:
        return {"kernel_name": kname, "status": "subprocess_timeout",
                "elapsed_sec": round(time.time() - t0, 1)}

    elapsed = time.time() - t0
    if not ok:
        return {"kernel_name": kname, "status": "subprocess_failed",
                "returncode": proc.returncode,
                "stderr_tail": stderr_tail,
                "stdout_tail": stdout_tail,
                "elapsed_sec": round(elapsed, 1)}

    try:
        with open(detail_path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return {"kernel_name": kname, "status": "result_read_failed",
                "error": f"{e!r}",
                "stdout_tail": stdout_tail,
                "elapsed_sec": round(elapsed, 1)}

    return {
        "kernel_name": kname,
        "status": d.get("status", "?"),
        "events": d.get("total_events", 0),
        "bad_events": d.get("bad_events", 0),
        "elapsed_sec": round(elapsed, 1),
        "compile_sec": d.get("compile_sec"),
    }


def aggregate(out_dir: Path) -> dict:
    """Read every details/*.json, build the buggy-kernels report."""
    detail_dir = out_dir / "details"
    if not detail_dir.exists():
        return {"kernels": 0, "buggy": 0}

    buggy = []
    all_kernels = []
    for jp in sorted(detail_dir.glob("*.json")):
        with open(jp, encoding="utf-8") as f:
            d = json.load(f)
        kname = d.get("kernel_name", jp.stem)
        all_kernels.append(kname)
        if d.get("status") != "ok":
            continue
        events = d.get("events", [])
        bad_evals = []
        bad_trains = []
        for ev in events:
            if ev.get("orig_ok") is False and ev.get("ref_ok") is True:
                target = bad_trains if ev.get("train_mode") else bad_evals
                target.append(ev)
        if not bad_evals and not bad_trains:
            continue
        policies_eval = Counter(e["policy"] for e in bad_evals)
        policies_train = Counter(e["policy"] for e in bad_trains)
        # Sample a representative diff_summary
        sample = None
        for ev in bad_evals + bad_trains:
            if ev.get("diff_summary"):
                sample = {
                    "policy": ev["policy"], "seed": ev["seed"],
                    "train_mode": ev["train_mode"],
                    "verdict": ev.get("verdict"),
                    "diff_summary": ev["diff_summary"],
                }
                break
        buggy.append({
            "kernel_name": kname,
            "bad_eval_events": len(bad_evals),
            "bad_train_events": len(bad_trains),
            "total_events": len(events),
            "fail_policies_eval": dict(policies_eval.most_common()),
            "fail_policies_train": dict(policies_train.most_common()),
            "sample_failure": sample,
        })
    buggy.sort(key=lambda x: -(x["bad_eval_events"] + x["bad_train_events"]))
    return {
        "kernels_total": len(all_kernels),
        "kernels_buggy": len(buggy),
        "kernels": all_kernels,
        "buggy": buggy,
    }


def write_report(out_dir: Path, agg: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "buggy_kernels.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# r-o Differential Probe — Buggy Kernels Report")
    lines.append("")
    lines.append(f"- Kernels scanned: **{agg['kernels_total']}**")
    lines.append(f"- Kernels with at least 1 orig_ok=False event "
                 f"(ref_ok=True): **{agg['kernels_buggy']}**")
    lines.append("")
    if not agg["buggy"]:
        lines.append("> No buggy kernels found.")
    else:
        lines.append("## Per-kernel summary (sorted by bad-event count)")
        lines.append("")
        lines.append("| Kernel | Bad/Total (eval) | Bad/Total (train) | "
                     "Top failing eval-policies | Sample failure |")
        lines.append("|---|---|---|---|---|")
        for b in agg["buggy"]:
            total = b["total_events"]
            # Approximate eval-half size (half of total when training included)
            bad_e = b["bad_eval_events"]
            bad_t = b["bad_train_events"]
            top = ", ".join(
                f"{k}({v})"
                for k, v in list(b["fail_policies_eval"].items())[:3]
            ) or "-"
            sf = b.get("sample_failure")
            sample = (f"`{sf['policy']}` seed={sf['seed']} "
                      f"({'train' if sf['train_mode'] else 'eval'}): "
                      f"{sf['verdict']}") if sf else "-"
            lines.append(f"| `{b['kernel_name']}` | "
                         f"{bad_e}/{total} | {bad_t}/{total} | "
                         f"{top} | {sample} |")
        lines.append("")
        lines.append("## Full sample diff summaries")
        lines.append("")
        for b in agg["buggy"]:
            sf = b.get("sample_failure")
            if not sf:
                continue
            lines.append(f"### {b['kernel_name']}")
            lines.append("")
            lines.append(f"- policy: `{sf['policy']}`")
            lines.append(f"- seed: {sf['seed']}")
            lines.append(f"- mode: {'train' if sf['train_mode'] else 'eval'}")
            lines.append(f"- verdict: {sf['verdict']}")
            lines.append(f"- diff: `{sf.get('diff_summary','')}`")
            lines.append("")
    (out_dir / "buggy_kernels_report.md").write_text(
        "\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT))
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated kernel_names to run only.")
    parser.add_argument("--only-file", type=str, default="",
                        help="Path to text file with one kernel_name per line.")
    parser.add_argument("--seeds", type=str, default="42,1337",
                        help="Comma-separated seeds.")
    parser.add_argument("--no-training", action="store_true",
                        help="Skip .train() mode")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-run kernels even if a detail file exists.")
    parser.add_argument("--policies", type=str, default="",
                        help="Override policy list (comma-separated). "
                             "Default = all 21 stress policies + identity.")
    parser.add_argument("--max", type=int, default=0,
                        help="Cap the number of kernels processed (debug).")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip running; just re-aggregate existing details.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "details").mkdir(exist_ok=True)

    if args.aggregate_only:
        agg = aggregate(out_dir)
        write_report(out_dir, agg)
        print(f"[AGG] {agg['kernels_buggy']} / {agg['kernels_total']} "
              f"kernels show ORIG_FAIL events.", flush=True)
        return

    # Determine policies
    if args.policies.strip():
        policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    else:
        from src.stress.policy_bank import get_all_policy_names
        policies = ["__identity__"] + get_all_policy_names()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    # Resolve --only / --only-file
    only_set: set[str] | None = None
    if args.only.strip():
        only_set = {x.strip() for x in args.only.split(",") if x.strip()}
    if args.only_file.strip():
        only_set = set() if only_set is None else only_set
        with open(args.only_file, encoding="utf-8") as f:
            only_set.update(ln.strip() for ln in f if ln.strip())

    targets = load_targets(only_set)
    print(f"[INFO] Loaded {len(targets)} kernel targets from "
          f"best_kernels.json", flush=True)

    # Resume support
    detail_dir = out_dir / "details"
    if not args.no_resume:
        before = len(targets)
        targets = [t for t in targets
                   if not (detail_dir / f"{t['kernel_name']}.json").exists()]
        print(f"[INFO] Resume: skipped {before - len(targets)} already-done "
              f"kernels; {len(targets)} remaining.", flush=True)

    if args.max > 0:
        targets = targets[:args.max]
        print(f"[INFO] --max={args.max}, capped to {len(targets)}.", flush=True)

    print(f"[INFO] Policies ({len(policies)}): {policies}", flush=True)
    print(f"[INFO] Seeds: {seeds}", flush=True)
    print(f"[INFO] include_training: {not args.no_training}", flush=True)

    manifest = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policies": policies,
        "seeds": seeds,
        "include_training": not args.no_training,
        "device": args.device,
        "atol": args.atol,
        "rtol": args.rtol,
        "kernel_count": len(targets),
        "results": [],
    }

    t_run = time.time()
    for i, target in enumerate(targets, 1):
        kname = target["kernel_name"]
        print(f"\n[{i}/{len(targets)}] {kname} "
              f"(L{target['level'][1:]}_P{target['problem_id']}) ...",
              flush=True)
        res = run_one(target, out_dir,
                      policies=policies, seeds=seeds,
                      include_training=not args.no_training,
                      timeout_sec=args.timeout_sec,
                      device=args.device, atol=args.atol, rtol=args.rtol)
        manifest["results"].append(res)
        ok_marker = "OK" if res["status"] == "ok" else f"FAIL({res['status']})"
        print(f"  -> {ok_marker} | events={res.get('events',0)} | "
              f"bad={res.get('bad_events',0)} | "
              f"elapsed={res.get('elapsed_sec','?')}s", flush=True)
        with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    elapsed_total = time.time() - t_run
    manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["elapsed_sec_total"] = round(elapsed_total, 1)
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n[DONE] Probed {len(targets)} kernels in "
          f"{elapsed_total/60:.1f} min. Aggregating ...", flush=True)
    agg = aggregate(out_dir)
    write_report(out_dir, agg)
    print(f"[AGG] {agg['kernels_buggy']} / {agg['kernels_total']} "
          f"kernels show ORIG_FAIL events (ref ok but orig wrong/NaN).",
          flush=True)
    print(f"  buggy list: {out_dir / 'buggy_kernels.json'}", flush=True)
    print(f"  report:     {out_dir / 'buggy_kernels_report.md'}", flush=True)


if __name__ == "__main__":
    main()
