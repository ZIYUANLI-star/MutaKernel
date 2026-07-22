#!/usr/bin/env python3
"""E1: controlled fault-probe study (RQ4; blueprint Tables 10-12 region).

Regenerates the first-order fault probes deterministically, replays the B1
baseline protocol on the corrected execution substrate, grades equivalence
evidence, and prepares the fault-to-stress map inputs.  Every E0 lesson is
built in:

  * per-kernel original-kernel control *gate*: the un-mutated kernel must
    pass the corrected substrate before any probe of that kernel is counted;
    gated-out kernels are skipped with a classified, auditable reason;
  * trial-level evidence: every worker trial reason lands in the observation;
  * the two legitimate INCONCLUSIVE families (state-sync non-bijection,
    CUDA invalid configuration) are classified into their own strata
    (src/audit/inconclusive.py), never merged into "unknown";
  * serial execution, MK_GPU_MEMORY_FRACTION-aware workers, completed.json
    checkpoint resume, run_manifest environment fingerprint.

Phases (run in order; ``generate`` is CPU-only and safe while E0 run5 owns
the GPU):

  generate  regenerate probes from the historical detail JSONs (same source,
            operators, and seed-42 downsampling as V1), reconcile probe ids
            against the historical population, and apply the machine-proven
            equivalence screen (byte-identical + versioned static rules).
  baseline  GPU: original-control gate + corrected-substrate B1 replay
            (5 draws, atol=rtol=1e-2, seed 42) for every non-machine-proven
            probe.
  equiv     GPU: sound paired-execution equivalence evidence for baseline
            survivors (validate_pair; directed stress policies).
  map       CPU: adapt equiv observations into mapbuild records, classify
            escape mechanisms, build the fault-to-stress map, and run the
            task-level k-fold cross-fitted closure evaluation.

Usage (remote A800):
  python scripts/run_e1_probe_study.py --phase generate \
      --details-dir phase1_details --out-dir /root/mk_v2_runs/e1
  python scripts/run_e1_probe_study.py --phase baseline \
      --kernelbench-root KernelBench --out-dir /root/mk_v2_runs/e1
  python scripts/run_e1_probe_study.py --phase equiv \
      --kernelbench-root KernelBench --out-dir /root/mk_v2_runs/e1
  python scripts/run_e1_probe_study.py --phase map --out-dir /root/mk_v2_runs/e1
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

from src.audit.inconclusive import (  # noqa: E402
    REASON_OTHER,
    classify_observation,
    summarize_reasons,
)

WORKER_CORRECTED = SCRIPT_DIR / "_mutant_worker.py"

# Identical to the V1 baseline protocol (B1 anchor; see run_e0_flip_rerun.py).
BASELINE_PROTOCOL = {
    "atol": 1e-2,
    "rtol": 1e-2,
    "num_test_inputs": 5,
    "seed": 42,
    "device": "cuda",
}

# V1 generation constants (scripts/full_block12.py) — must not drift.
SAMPLE_PER_OP = 3
GENERATION_SEED = 42

EQUIV_RUNS = 20
EQUIV_BASE_SEED = 10000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def environment_fingerprint():
    info = {
        "captured_at": _now(),
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_extensions_dir": os.environ.get("TORCH_EXTENSIONS_DIR", ""),
        "mk_gpu_memory_fraction": os.environ.get("MK_GPU_MEMORY_FRACTION", ""),
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
    return info


def run_worker(cfg: dict, timeout: int, scratch: Path):
    """Serial isolated worker execution with process-group kill (E0 pattern)."""
    tag = cfg["mutant_id"].replace("/", "_")
    cfg_path = scratch / f"cfg_{tag}.json"
    res_path = scratch / f"res_{tag}.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    if res_path.exists():
        res_path.unlink()
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, str(WORKER_CORRECTED), str(cfg_path), str(res_path)],
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
    for path in (cfg_path, res_path):
        try:
            path.unlink()
        except OSError:
            pass
    return result, timed_out, wall_ms, stdout[-4000:], stderr[-4000:]


def _write_worker_log(out: Path, name: str, stdout: str, stderr: str) -> None:
    log_dir = out / "worker_logs"
    log_dir.mkdir(exist_ok=True)
    (log_dir / f"{name.replace('/', '_')}.log").write_text(
        f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Phase: generate (CPU-only)
# ---------------------------------------------------------------------------

def load_historical_details(details_dir: Path):
    """Recover per-kernel source + historical sampled probe ids from V1 details."""
    kernels = []
    for detail_file in sorted(details_dir.glob("*.json")):
        data = json.loads(detail_file.read_text(encoding="utf-8"))
        mutants = data.get("mutants", [])
        if not mutants:
            continue
        sources = {m.get("original_code", "") for m in mutants if m.get("original_code")}
        if len(sources) != 1:
            raise ValueError(
                f"{detail_file.name}: expected one kernel source, got {len(sources)}")
        kernels.append({
            "kernel": data["kernel"],
            "kernel_source": next(iter(sources)),
            "historical_probe_ids": [m["id"] for m in mutants],
            "historical_status": {m["id"]: m["status"] for m in mutants},
            "detail_file": detail_file.name,
        })
    return kernels


def regenerate_probes(kernel_entry: dict):
    """Deterministically regenerate the V1 probe sample for one kernel.

    Mirrors scripts/full_block12.py exactly: full operator generation over
    the kernel source, then rng.Random(42 + problem_id).sample(...) with
    SAMPLE_PER_OP per operator, in per-operator insertion order.
    """
    from src.models import KernelInfo
    from src.mutengine.mutant_runner import MutantRunner

    kernel_meta = kernel_entry["kernel"]
    kernel = KernelInfo(
        problem_id=int(kernel_meta["problem_id"]),
        level=int(kernel_meta["level"]),
        problem_name=kernel_meta["problem_name"],
        source_path="",
        kernel_code=kernel_entry["kernel_source"],
        reference_module_path="",
        language=kernel_meta.get("language", "cuda"),
    )
    runner = MutantRunner(
        atol=BASELINE_PROTOCOL["atol"], rtol=BASELINE_PROTOCOL["rtol"],
        num_test_inputs=BASELINE_PROTOCOL["num_test_inputs"],
        device="cpu", seed=GENERATION_SEED,
        categories=["A", "B", "C", "D"],
    )
    all_mutants = runner.generate_mutants(kernel)

    by_op = defaultdict(list)
    for mutant in all_mutants:
        by_op[mutant.operator_name].append(mutant)

    rng = random.Random(GENERATION_SEED + kernel.problem_id)
    sampled = []
    for op_name, muts in by_op.items():
        sampled.extend(rng.sample(muts, min(SAMPLE_PER_OP, len(muts))))
    return kernel, sampled, len(all_mutants)


def phase_generate(args):
    from src.mutengine.fault_classes import OPERATOR_TO_FAULT_CLASS, TAXONOMY_VERSION
    from src.mutengine.static_equiv_rules import RULE_VERSIONS, machine_proof, rules_content_version

    out = args.out_dir
    probes_dir = out / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    kernels = load_historical_details(args.details_dir)
    print(f"[{_now()}] generate: {len(kernels)} kernels from {args.details_dir}", flush=True)

    total = Counter()
    proof_counter = Counter()
    reconciliation = []
    all_probe_ids = []
    for index, entry in enumerate(kernels):
        kernel, sampled, population = regenerate_probes(entry)
        regenerated_ids = [m.id for m in sampled]
        historical_ids = entry["historical_probe_ids"]
        id_match = set(regenerated_ids) == set(historical_ids)
        reconciliation.append({
            "kernel": kernel.problem_name,
            "historical": len(historical_ids),
            "regenerated": len(regenerated_ids),
            "exact_id_match": id_match,
            "missing_vs_history": sorted(set(historical_ids) - set(regenerated_ids)),
            "extra_vs_history": sorted(set(regenerated_ids) - set(historical_ids)),
        })

        probe_records = []
        for mutant in sampled:
            proof = machine_proof(mutant)
            if proof:
                proof_counter[proof["proof_kind"]] += 1
                if proof["rule"]:
                    proof_counter[f"rule:{proof['rule']}"] += 1
            total[mutant.operator_category] += 1
            all_probe_ids.append(mutant.id)
            probe_records.append({
                "probe_id": mutant.id,
                "operator_name": mutant.operator_name,
                "operator_category": mutant.operator_category,
                "fault_class": OPERATOR_TO_FAULT_CLASS.get(mutant.operator_name),
                "site": {
                    "line_start": mutant.site.line_start,
                    "line_end": mutant.site.line_end,
                    "original_code": mutant.site.original_code[:200],
                    "node_type": mutant.site.node_type,
                },
                "mutated_code_sha256": _sha256_text(mutant.mutated_code),
                "mutated_code": mutant.mutated_code,
                "machine_proof": proof,
                "historical_status": entry["historical_status"].get(mutant.id),
            })

        (probes_dir / f"{kernel.problem_name}.json").write_text(json.dumps({
            "kernel": {
                "problem_id": kernel.problem_id,
                "level": kernel.level,
                "problem_name": kernel.problem_name,
                "language": kernel.language,
            },
            "kernel_source_sha256": _sha256_text(entry["kernel_source"]),
            "kernel_source": entry["kernel_source"],
            "generated_population": population,
            "probes": probe_records,
        }, indent=1), encoding="utf-8")
        if (index + 1) % 10 == 0:
            print(f"[{_now()}] generate: {index + 1}/{len(kernels)} kernels", flush=True)

    mismatched = [r for r in reconciliation if not r["exact_id_match"]]
    manifest = {
        "phase": "generate",
        "created_at": _now(),
        "generation": {
            "sample_per_op": SAMPLE_PER_OP,
            "seed": GENERATION_SEED,
            "sampling": "rng.Random(seed + problem_id).sample per operator (V1 full_block12)",
        },
        "taxonomy_version": TAXONOMY_VERSION,
        "static_rules": {"versions": RULE_VERSIONS, "content_version": rules_content_version()},
        "counts": {
            "kernels": len(kernels),
            "probes_total": sum(total.values()),
            "by_category": dict(sorted(total.items())),
            "machine_proof": dict(sorted(proof_counter.items())),
        },
        "reconciliation": {
            "kernels_exact_match": len(reconciliation) - len(mismatched),
            "kernels_mismatched": len(mismatched),
            "mismatches": mismatched,
        },
        "probe_id_digest": hashlib.sha256(
            "".join(sorted(all_probe_ids)).encode()).hexdigest(),
        "environment": environment_fingerprint(),
    }
    (out / "probe_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"[{_now()}] generate DONE: probes={sum(total.values())} "
          f"by_cat={dict(total)} machine_proof={dict(proof_counter)} "
          f"id_mismatch_kernels={len(mismatched)}", flush=True)


# ---------------------------------------------------------------------------
# Phase: baseline (GPU)
# ---------------------------------------------------------------------------

def _resolve_problem_file(kb_root: Path, level: int, problem_id: int) -> Path:
    level_dir = kb_root / "KernelBench" / f"level{level}"
    matches = sorted(level_dir.glob(f"{problem_id}_*.py"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one problem file for L{level} P{problem_id}, got {matches}")
    return matches[0]


def _load_probe_files(out: Path):
    probes_dir = out / "probes"
    files = sorted(probes_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"no probe files under {probes_dir}; run --phase generate first")
    return [json.loads(f.read_text(encoding="utf-8")) for f in files]


def _classified_status(result, timed_out):
    status = "inconclusive_timeout" if timed_out else (
        (result or {}).get("status", "inconclusive_worker_failure"))
    reason_class = None
    if status not in ("killed", "survived"):
        reason_class = classify_observation(result or {}, timed_out=timed_out)
    return status, reason_class


def phase_baseline(args):
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="e1_", dir=str(out)))
    obs_path = out / "baseline_observations.jsonl"
    done_path = out / "baseline_completed.json"
    completed = set(json.loads(done_path.read_text()) if done_path.exists() else [])

    kernel_files = _load_probe_files(out)
    controls_path = out / "original_controls.json"
    controls = json.loads(controls_path.read_text()) if controls_path.exists() else {}

    manifest = {
        "phase": "baseline",
        "run_id": f"e1-baseline-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "started_at": _now(),
        "protocol": BASELINE_PROTOCOL,
        "worker": str(WORKER_CORRECTED),
        "worker_sha256": _sha256_text(WORKER_CORRECTED.read_text(encoding="utf-8")),
        "timeout_s": args.timeout,
        "environment": environment_fingerprint(),
    }
    (out / "baseline_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    status_counter = Counter()
    for kf in kernel_files:
        kernel = kf["kernel"]
        key = kernel["problem_name"]
        problem_file = _resolve_problem_file(
            args.kernelbench_root, kernel["level"], kernel["problem_id"])

        # E0 lesson 1: original-control gate, corrected substrate, before any
        # probe of this kernel is executed or counted.
        if key not in controls:
            cfg = {
                "mode": "run",
                "mutant_id": f"{key}__ORIGINAL_CONTROL",
                "problem_id": kernel["problem_id"], "level": kernel["level"],
                "problem_name": key, "language": kernel.get("language", "cuda"),
                "problem_file": str(problem_file),
                "operator_name": "none", "operator_category": "A",
                "site": {"line_start": 1, "line_end": 1},
                "original_code": kf["kernel_source"],
                "mutated_code": kf["kernel_source"],
                **BASELINE_PROTOCOL,
            }
            result, timed_out, wall_ms, so, se = run_worker(cfg, args.timeout, scratch)
            status, reason_class = _classified_status(result, timed_out)
            controls[key] = {
                "status": status,
                "reason_class": reason_class,
                "wall_ms": round(wall_ms, 1),
                "error": (result or {}).get("error", "")[:400],
                "trials": (result or {}).get("trials"),
                "checked_at": _now(),
            }
            if status != "survived":
                _write_worker_log(out, f"{key}__control", so, se)
            controls_path.write_text(json.dumps(controls, indent=2), encoding="utf-8")
            print(f"[{_now()}] control {key}: {status}"
                  + (f" ({reason_class})" if reason_class else ""), flush=True)

        control = controls[key]
        gate_open = control["status"] == "survived"

        for probe in kf["probes"]:
            probe_id = probe["probe_id"]
            if probe_id in completed:
                continue
            record = {
                "probe_id": probe_id,
                "kernel": key,
                "operator_name": probe["operator_name"],
                "operator_category": probe["operator_category"],
                "fault_class": probe["fault_class"],
                "machine_proof": probe["machine_proof"],
                "historical_status": probe["historical_status"],
                "original_control": {
                    "status": control["status"],
                    "reason_class": control.get("reason_class"),
                },
                "started_at": _now(),
            }
            if probe["machine_proof"]:
                record["status"] = "machine_proven_equivalent"
                record["skip_reason"] = "machine_proof"
            elif not gate_open:
                record["status"] = "excluded_control_failed"
                record["skip_reason"] = (
                    f"original control {control['status']}"
                    f" ({control.get('reason_class') or 'unclassified'})")
            else:
                cfg = {
                    "mode": "run",
                    "mutant_id": probe_id,
                    "problem_id": kernel["problem_id"], "level": kernel["level"],
                    "problem_name": key, "language": kernel.get("language", "cuda"),
                    "problem_file": str(problem_file),
                    "operator_name": probe["operator_name"],
                    "operator_category": probe["operator_category"],
                    "site": probe["site"],
                    "original_code": kf["kernel_source"],
                    "mutated_code": probe["mutated_code"],
                    **BASELINE_PROTOCOL,
                }
                result, timed_out, wall_ms, so, se = run_worker(cfg, args.timeout, scratch)
                status, reason_class = _classified_status(result, timed_out)
                record.update({
                    "status": status,
                    "inconclusive_class": reason_class,
                    "wall_ms": round(wall_ms, 1),
                    "worker_time_ms": (result or {}).get("time_ms"),
                    "kill_seed": (result or {}).get("kill_seed"),
                    "error": (result or {}).get("error", "")[:400],
                    "trials": (result or {}).get("trials"),
                    "timed_out": timed_out,
                })
                if timed_out or result is None or se.strip():
                    _write_worker_log(out, f"{probe_id}_baseline", so, se)
            record["finished_at"] = _now()
            with open(obs_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            completed.add(probe_id)
            done_path.write_text(json.dumps(sorted(completed)), encoding="utf-8")
            status_counter[record["status"]] += 1
            if len(completed) % 20 == 0:
                print(f"[{_now()}] baseline: {len(completed)} probes done "
                      f"{dict(status_counter)}", flush=True)

    inconclusive_classes = Counter()
    with open(obs_path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("inconclusive_class"):
                inconclusive_classes[row["inconclusive_class"]] += 1
    summary = {
        "finished_at": _now(),
        "completed": len(completed),
        "status_counts": dict(status_counter),
        "inconclusive_classes": dict(inconclusive_classes),
        "controls": {
            "total": len(controls),
            "passed": sum(1 for c in controls.values() if c["status"] == "survived"),
            "failed_reasons": summarize_reasons(
                c.get("reason_class") or REASON_OTHER
                for c in controls.values() if c["status"] != "survived"),
        },
    }
    (out / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[{_now()}] baseline DONE: {summary}", flush=True)


# ---------------------------------------------------------------------------
# Phase: equiv (GPU)
# ---------------------------------------------------------------------------

def phase_equiv(args):
    out = args.out_dir
    scratch = Path(tempfile.mkdtemp(prefix="e1eq_", dir=str(out)))
    obs_path = out / "equiv_observations.jsonl"
    done_path = out / "equiv_completed.json"
    completed = set(json.loads(done_path.read_text()) if done_path.exists() else [])

    baseline_obs = {}
    with open(out / "baseline_observations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            baseline_obs[row["probe_id"]] = row

    kernel_files = _load_probe_files(out)
    survivors = []
    for kf in kernel_files:
        for probe in kf["probes"]:
            row = baseline_obs.get(probe["probe_id"])
            if row and row.get("status") == "survived":
                survivors.append((kf, probe))
    print(f"[{_now()}] equiv: {len(survivors)} baseline survivors, "
          f"{len(completed)} already done", flush=True)

    manifest = {
        "phase": "equiv",
        "run_id": f"e1-equiv-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "started_at": _now(),
        "equiv_runs": EQUIV_RUNS,
        "base_seed": EQUIV_BASE_SEED,
        "timeout_s": args.timeout,
        "environment": environment_fingerprint(),
    }
    (out / "equiv_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    counter = Counter()
    for kf, probe in survivors:
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
            "equiv_runs": EQUIV_RUNS,
            "base_seed": EQUIV_BASE_SEED,
            "atol": BASELINE_PROTOCOL["atol"],
            "rtol": BASELINE_PROTOCOL["rtol"],
        }
        result, timed_out, wall_ms, so, se = run_worker(cfg, args.timeout, scratch)
        validation_status = "inconclusive" if timed_out else (
            (result or {}).get("validation_status", "inconclusive"))
        if validation_status == "fail":
            grade = "WITNESSED_NON_EQUIVALENT"
        elif validation_status == "pass":
            grade = "LIKELY_EQUIVALENT"
        else:
            grade = "INCONCLUSIVE"
        record = {
            "probe_id": probe_id,
            "kernel": kernel["problem_name"],
            "operator_name": probe["operator_name"],
            "fault_class": probe["fault_class"],
            "validation_status": validation_status,
            "evidence_grade": grade,
            "inconclusive_class": (
                classify_observation(result or {}, timed_out=timed_out)
                if grade == "INCONCLUSIVE" else None),
            "reason": (result or {}).get("reason", "")[:400],
            "divergence": (result or {}).get("divergence"),
            "trials": (result or {}).get("trials"),
            "valid_rounds": (result or {}).get("valid_rounds"),
            "wall_ms": round(wall_ms, 1),
            "timed_out": timed_out,
            "finished_at": _now(),
        }
        if timed_out or result is None or se.strip():
            _write_worker_log(out, f"{probe_id}_equiv", so, se)
        with open(obs_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        completed.add(probe_id)
        done_path.write_text(json.dumps(sorted(completed)), encoding="utf-8")
        counter[grade] += 1
        if len(completed) % 10 == 0:
            print(f"[{_now()}] equiv: {len(completed)}/{len(survivors)} "
                  f"{dict(counter)}", flush=True)

    (out / "equiv_summary.json").write_text(json.dumps({
        "finished_at": _now(), "completed": len(completed),
        "grades": dict(counter),
    }, indent=2), encoding="utf-8")
    print(f"[{_now()}] equiv DONE: {dict(counter)}", flush=True)


# ---------------------------------------------------------------------------
# Phase: map (CPU)
# ---------------------------------------------------------------------------

def _trials_to_map_records(row):
    """Adapt one equiv observation's trials into mapbuild record dicts.

    Limitation (documented): the equiv worker covers the value dimension
    (random + directed stress policies).  dtype/train/config/repeated cases
    come from the full audit stress phase and are appended by the same
    adapter once that phase runs.
    """
    from src.cse.verdict import (
        VERDICT_INCONCLUSIVE,
        VERDICT_INDISTINGUISHED,
        VERDICT_SPEC_VIOLATION,
    )

    records = []
    for order, trial in enumerate(row.get("trials") or []):
        status = str(trial.get("status", "")).lower()
        if status == "fail":
            verdict = VERDICT_SPEC_VIOLATION
        elif status == "pass":
            verdict = VERDICT_INDISTINGUISHED
        else:
            verdict = VERDICT_INCONCLUSIVE
        policy = trial.get("policy") or "iid"
        records.append({
            "probe_id": row["probe_id"],
            "operator": row["operator_name"],
            "task_id": row["probe_id"].split("__", 1)[0],
            "case": {"policy": policy, "mode": "eval", "parameters": {}},
            "verdict": verdict,
            "order": order,
        })
    return records


def phase_map(args):
    from src.audit.crossfit import crossfit_map_evaluation
    from src.audit.mapbuild import build_fault_to_stress_map
    from src.audit.ripr import classify_escape
    from src.cse.verdict import VERDICT_SPEC_VIOLATION

    out = args.out_dir
    records = []
    escape = Counter()
    with open(out / "equiv_observations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            row_records = _trials_to_map_records(row)
            records.extend(row_records)
            first_kill = next(
                (r for r in row_records if r["verdict"] == VERDICT_SPEC_VIOLATION), None)
            if first_kill:
                escape[classify_escape(first_kill["case"])["mechanism"]] += 1

    # Baseline kills are witnessed non-equivalences under the default IID
    # 5-draw case; include them so map closure rates see the easy kills too.
    with open(out / "baseline_observations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("status") == "killed":
                records.append({
                    "probe_id": row["probe_id"],
                    "operator": row["operator_name"],
                    "task_id": row["probe_id"].split("__", 1)[0],
                    "case": {"policy": "iid", "mode": "eval", "parameters": {}},
                    "verdict": VERDICT_SPEC_VIOLATION,
                    "order": 0,
                })

    fault_map = build_fault_to_stress_map(
        records,
        map_version=args.map_version,
        derived_from_run=str(out),
    )
    (out / "fault_to_stress_map.json").write_text(
        json.dumps(fault_map, indent=2), encoding="utf-8")

    crossfit = crossfit_map_evaluation(
        records, k=args.folds, planned_cases=args.planned_cases,
        map_version=args.map_version)
    (out / "map_crossfit_evaluation.json").write_text(
        json.dumps(crossfit, indent=2), encoding="utf-8")

    (out / "escape_mechanisms.json").write_text(json.dumps({
        "created_at": _now(),
        "note": "first-kill escape classification over equiv-phase witnesses",
        "mechanisms": dict(escape.most_common()),
    }, indent=2), encoding="utf-8")

    print(f"[{_now()}] map DONE: witnessed={fault_map['witnessed_probe_count']} "
          f"crossfit={crossfit['pooled']} escape={dict(escape)}", flush=True)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", required=True,
                    choices=["generate", "baseline", "equiv", "map"])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--details-dir", type=Path,
                    help="historical V1 detail JSONs (generate phase)")
    ap.add_argument("--kernelbench-root", type=Path,
                    help="KernelBench repo root (baseline/equiv phases)")
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--map-version", default="e1-map-v1")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--planned-cases", type=int, default=8)
    args = ap.parse_args()

    if args.phase == "generate":
        if not args.details_dir:
            ap.error("--details-dir is required for --phase generate")
        phase_generate(args)
    elif args.phase == "baseline":
        if not args.kernelbench_root:
            ap.error("--kernelbench-root is required for --phase baseline")
        phase_baseline(args)
    elif args.phase == "equiv":
        if not args.kernelbench_root:
            ap.error("--kernelbench-root is required for --phase equiv")
        phase_equiv(args)
    else:
        phase_map(args)


if __name__ == "__main__":
    main()
