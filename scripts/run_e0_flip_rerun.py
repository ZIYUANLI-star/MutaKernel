#!/usr/bin/env python3
"""E0: harness-soundness flip-rate rerun (Table 3 of the V2 evaluation).

Re-executes historical Phase-I probes under the *corrected* execution
substrate (constructor-RNG replay + strict named state sync + isolated
inputs + strict oracle) with the *identical* baseline protocol the V1 study
used (5 random draws, atol=rtol=1e-2, eval mode, seed 42), and records how
many historical kill/survive verdicts flip.

Design properties:
  * every probe runs in its own subprocess (scripts/_mutant_worker.py) with
    a hard timeout and process-group kill -> no deadlock or GPU-state leak
    can propagate to the driver;
  * resumable: completed probe ids are checkpointed after every probe;
  * full evidence: one JSONL observation per probe (timestamps, timings,
    stdout/stderr of the worker preserved on failure), plus a run manifest
    with environment fingerprint and protocol constants.

Usage:
  python scripts/run_e0_flip_rerun.py --details-dir <dir with Phase-I *.json>
      --kernelbench-root <KernelBench repo root> --out-dir <run dir>
      [--mode pilot|full] [--per-cell 30] [--timeout 420] [--sample-seed 20260721]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

WORKER_CORRECTED = SCRIPT_DIR / "_mutant_worker.py"
WORKER_LEGACY = SCRIPT_DIR / "_mutant_worker_legacy.py"

# Identical to the V1 baseline protocol (config.py / full_block12.py).
PROTOCOL = {
    "atol": 1e-2,
    "rtol": 1e-2,
    "num_test_inputs": 5,
    "seed": 42,
    "device": "cuda",
    "mode_note": (
        "baseline 5-draw allclose, eval mode; paired same-environment design: "
        "legacy substrate (pre-fse-rework tag, no RNG replay / no state sync) "
        "vs corrected substrate, run back-to-back per probe on the same GPU. "
        "Primary metric: legacy-vs-corrected paired flips (isolates the "
        "substrate effect); secondary: corrected-vs-historical (adds "
        "environment drift)."
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_probes(details_dir: Path):
    probes = []
    for f in sorted(details_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        kernel = data["kernel"]
        for m in data["mutants"]:
            probes.append({
                "probe_id": m["id"],
                "level": int(kernel["level"]),
                "problem_id": int(kernel["problem_id"]),
                "problem_name": kernel["problem_name"],
                "language": kernel.get("language", "cuda"),
                "operator_name": m["operator_name"],
                "operator_category": m["operator_category"],
                "site": m["site"],
                "original_code": m["original_code"],
                "mutated_code": m["mutated_code"],
                "historical_status": m["status"],
                "historical_time_ms": m.get("execution_time_ms"),
                "source_detail_file": f.name,
            })
    return probes


def resolve_problem_file(kb_root: Path, level: int, problem_id: int) -> Path:
    level_dir = kb_root / "KernelBench" / f"level{level}"
    matches = sorted(level_dir.glob(f"{problem_id}_*.py"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one problem file for L{level} P{problem_id}, got {matches}")
    return matches[0]


def reference_is_stateful(problem_file: Path, cache: dict) -> bool:
    """CPU-only check: does the reference Model carry parameters/buffers?"""
    key = str(problem_file)
    if key in cache:
        return cache[key]
    code = (
        "import importlib.util, torch, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('ref_probe', r'''{problem_file}''')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "init = getattr(mod, 'get_init_inputs', lambda: [])()\n"
        "model = mod.Model(*init) if isinstance(init, (list, tuple)) else mod.Model()\n"
        "print(json.dumps(len(model.state_dict()) > 0))\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, timeout=120,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
        stateful = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        stateful = None  # recorded as unknown stratum
    cache[key] = stateful
    return stateful


def stratified_sample(probes, per_cell: int, sample_seed: int, stateful_of):
    """Cells: level x stateful x historical outcome-group (killed vs not-killed).

    Stillborn probes are excluded from the pilot (their rerun measures the
    compile environment, not state soundness) and strict_equivalent probes
    carry a source-level proof, so the flip pilot uses the dynamic verdicts.
    """
    eligible = [p for p in probes
                if p["historical_status"] in ("killed", "survived", "candidate_equivalent")]
    cells = defaultdict(list)
    for p in eligible:
        group = "killed" if p["historical_status"] == "killed" else "not_killed"
        cells[(p["level"], stateful_of(p), group)].append(p)
    rng = random.Random(sample_seed)
    sample = []
    plan = {}
    for key in sorted(cells, key=str):
        pool = sorted(cells[key], key=lambda p: p["probe_id"])
        take = min(per_cell, len(pool))
        sample.extend(rng.sample(pool, take))
        plan[str(key)] = {"pool": len(pool), "sampled": take}
    return sample, plan


def run_worker(worker: Path, cfg: dict, timeout: int, scratch: Path):
    cfg_path = scratch / f"cfg_{cfg['mutant_id'].replace('/', '_')}.json"
    res_path = scratch / f"res_{cfg['mutant_id'].replace('/', '_')}.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    if res_path.exists():
        res_path.unlink()
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(worker), str(cfg_path), str(res_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(PROJECT_ROOT), start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        stdout, stderr = proc.communicate()
        timed_out = True
    wall_ms = (time.time() - t0) * 1000
    result = None
    if res_path.exists() and res_path.stat().st_size > 2:
        try:
            result = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception:
            result = None
    for p in (cfg_path, res_path):
        try:
            p.unlink()
        except OSError:
            pass
    return result, timed_out, wall_ms, stdout[-4000:], stderr[-4000:]


def gpu_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def environment_fingerprint():
    info = {
        "captured_at": _now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR", ""),
    }
    try:
        import torch
        info.update({
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        })
    except Exception as exc:  # pragma: no cover
        info["torch_error"] = str(exc)
    info["nvidia_smi"] = gpu_snapshot()
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--details-dir", required=True, type=Path)
    ap.add_argument("--kernelbench-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--mode", choices=["pilot", "full", "smoke"], default="pilot")
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--sample-seed", type=int, default=20260721)
    ap.add_argument("--limit", type=int, default=0, help="hard cap for smoke runs")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "worker_logs").mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="e0_", dir=str(out)))
    obs_path = out / "observations.jsonl"
    done_path = out / "completed.json"
    completed = set(json.loads(done_path.read_text()) if done_path.exists() else [])

    probes = load_probes(args.details_dir)
    details_digest = hashlib.sha256(
        "".join(sorted(p["probe_id"] for p in probes)).encode()).hexdigest()

    # Per-kernel original-kernel control: does the *unmutated* kernel still
    # pass the baseline protocol on this GPU?  Probes of kernels whose
    # original fails here are environment-incompatible on this architecture
    # and are excluded from attributable flip statistics (kept in the log).
    controls_path = out / "original_controls.json"
    original_controls = (
        json.loads(controls_path.read_text()) if controls_path.exists() else {}
    )

    def ensure_original_control(p, problem_file):
        key = p["problem_name"]
        if key in original_controls:
            return original_controls[key]
        entry = {}
        base_cfg = {
            "mode": "run",
            "mutant_id": f"{key}__ORIGINAL_CONTROL",
            "problem_id": p["problem_id"], "level": p["level"],
            "problem_name": key, "language": p["language"],
            "problem_file": str(problem_file),
            "operator_name": "none", "operator_category": "A",
            "site": {"line_start": 1, "line_end": 1},
            "original_code": p["original_code"],
            "mutated_code": p["original_code"],
            **{k: PROTOCOL[k] for k in ("atol", "rtol", "num_test_inputs", "seed", "device")},
        }
        for arm, worker in (("legacy", WORKER_LEGACY), ("corrected", WORKER_CORRECTED)):
            result, timed_out, wall_ms, so, se = run_worker(worker, base_cfg, args.timeout, scratch)
            entry[arm] = {
                "status": "inconclusive_timeout" if timed_out else (
                    (result or {}).get("status", "inconclusive_worker_failure")),
                "wall_ms": round(wall_ms, 1),
                "error": (result or {}).get("error", "")[:300],
                "trials": (result or {}).get("trials"),
            }
        original_controls[key] = entry
        controls_path.write_text(json.dumps(original_controls, indent=2), encoding="utf-8")
        print(f"[{_now()}] control {key}: legacy={entry['legacy']['status']} "
              f"corrected={entry['corrected']['status']}", flush=True)
        return entry

    stateful_cache: dict = {}

    def stateful_of(p):
        pf = resolve_problem_file(args.kernelbench_root, p["level"], p["problem_id"])
        return reference_is_stateful(pf, stateful_cache)

    if args.mode == "full":
        todo = [p for p in probes if p["historical_status"] != "strict_equivalent"]
        plan = {"mode": "full", "count": len(todo)}
        for p in todo:
            stateful_of(p)
    elif args.mode == "smoke":
        by_status = defaultdict(list)
        for p in probes:
            by_status[p["historical_status"]].append(p)
        todo = [sorted(by_status[s], key=lambda x: x["probe_id"])[0]
                for s in ("killed", "survived", "candidate_equivalent") if by_status[s]]
        if args.limit:
            todo = todo[: args.limit]
        plan = {"mode": "smoke", "count": len(todo)}
        for p in todo:
            stateful_of(p)
    else:
        todo, plan = stratified_sample(probes, args.per_cell, args.sample_seed, stateful_of)

    if args.shard_count > 1:
        def shard_of(probe):
            digest = hashlib.sha256(probe["probe_id"].encode()).hexdigest()
            return int(digest[:8], 16) % args.shard_count
        todo = [p for p in todo if shard_of(p) == args.shard_index]
        plan = {"sharded_from": plan, "shard_index": args.shard_index,
                "shard_count": args.shard_count, "shard_probes": len(todo)}

    manifest = {
        "run_id": f"e0-{args.mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "started_at": _now(),
        "protocol": PROTOCOL,
        "mode": args.mode,
        "sampling": {"per_cell": args.per_cell, "sample_seed": args.sample_seed,
                      "plan": plan},
        "probe_population": Counter(p["historical_status"] for p in probes),
        "details_dir": str(args.details_dir),
        "details_probe_id_digest": details_digest,
        "worker_corrected": str(WORKER_CORRECTED),
        "worker_corrected_sha256": _sha256_file(WORKER_CORRECTED),
        "worker_legacy": str(WORKER_LEGACY),
        "worker_legacy_sha256": _sha256_file(WORKER_LEGACY),
        "timeout_s": args.timeout,
        "environment": environment_fingerprint(),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"[{_now()}] E0 {args.mode}: {len(todo)} probes planned, "
          f"{len(completed)} already completed", flush=True)

    flips = Counter()
    for i, p in enumerate(todo):
        if p["probe_id"] in completed:
            continue
        problem_file = resolve_problem_file(args.kernelbench_root, p["level"], p["problem_id"])
        cfg = {
            "mode": "run",
            "mutant_id": p["probe_id"],
            "problem_id": p["problem_id"],
            "level": p["level"],
            "problem_name": p["problem_name"],
            "language": p["language"],
            "problem_file": str(problem_file),
            "operator_name": p["operator_name"],
            "operator_category": p["operator_category"],
            "site": p["site"],
            "original_code": p["original_code"],
            "mutated_code": p["mutated_code"],
            **{k: PROTOCOL[k] for k in ("atol", "rtol", "num_test_inputs", "seed", "device")},
        }
        started = _now()
        control = ensure_original_control(p, problem_file)
        arms = {}
        for arm, worker in (("legacy", WORKER_LEGACY), ("corrected", WORKER_CORRECTED)):
            result, timed_out, wall_ms, so, se = run_worker(worker, cfg, args.timeout, scratch)
            status = "inconclusive_timeout" if timed_out else (
                (result or {}).get("status", "inconclusive_worker_failure"))
            arms[arm] = {
                "status": status,
                "wall_ms": round(wall_ms, 1),
                "worker_time_ms": (result or {}).get("time_ms"),
                "kill_seed": (result or {}).get("kill_seed"),
                "error": (result or {}).get("error", "")[:500],
                "timed_out": timed_out,
                "trials": (result or {}).get("trials"),
            }
            if timed_out or result is None or se.strip():
                log = out / "worker_logs" / f"{p['probe_id'].replace('/', '_')}_{arm}.log"
                log.write_text(f"STDOUT:\n{so}\n\nSTDERR:\n{se}\n", encoding="utf-8")

        def group(status):
            if status == "killed":
                return "killed"
            if status == "survived":
                return "not_killed"
            return None  # stillborn / inconclusive

        hist_group = "killed" if p["historical_status"] == "killed" else "not_killed"
        leg_g, cor_g = group(arms["legacy"]["status"]), group(arms["corrected"]["status"])
        original_ok = (control["corrected"]["status"] == "survived"
                       and control["legacy"]["status"] == "survived")
        record = {
            "probe_id": p["probe_id"],
            "level": p["level"],
            "problem_name": p["problem_name"],
            "operator_name": p["operator_name"],
            "operator_category": p["operator_category"],
            "reference_stateful": stateful_cache.get(str(problem_file)),
            "original_control_ok": original_ok,
            "historical_status": p["historical_status"],
            "legacy": arms["legacy"],
            "corrected": arms["corrected"],
            "flip_paired": (leg_g != cor_g) if (original_ok and leg_g and cor_g) else None,
            "flip_vs_historical": (hist_group != cor_g) if (original_ok and cor_g) else None,
            "legacy_reproduces_history": (leg_g == hist_group) if (original_ok and leg_g) else None,
            "started_at": started,
            "finished_at": _now(),
        }
        with open(obs_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        completed.add(p["probe_id"])
        done_path.write_text(json.dumps(sorted(completed)), encoding="utf-8")

        flips[("paired", record["flip_paired"])] += 1
        flips[("vs_hist", record["flip_vs_historical"])] += 1
        if (i + 1) % 5 == 0 or record["flip_paired"]:
            print(f"[{_now()}] {i+1}/{len(todo)} {p['probe_id']}: hist={p['historical_status']} "
                  f"legacy={arms['legacy']['status']} corrected={arms['corrected']['status']} "
                  f"paired_flip={record['flip_paired']} | GPU {gpu_snapshot()}",
                  flush=True)

    summary = {
        "finished_at": _now(),
        "completed": len(completed),
        "flip_counts": {f"{k[0]}={k[1]}": v for k, v in sorted(flips.items(), key=str)},
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{_now()}] DONE: {summary}", flush=True)


if __name__ == "__main__":
    main()
