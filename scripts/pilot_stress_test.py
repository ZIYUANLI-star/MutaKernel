#!/usr/bin/env python3
"""Pilot stress test: pick 5 diverse survived mutants and run 3-layer augmentation.

Self-contained script that adapts paths for local Windows execution.
"""
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# --- Local path mapping (Linux server → WSL local) ---
LINUX_KB_ROOT = "/home/kbuser/projects/KernelBench-0"
LOCAL_KB_ROOT = Path(r"\\wsl.localhost\Ubuntu-22.04\home\kbuser\projects\KernelBench-0")

BLOCK12_RESULT_DIR = PROJECT_ROOT / "full_block12_results"
BEST_KERNELS_FILE = PROJECT_ROOT / "best_kernels.json"
PILOT_RESULT_DIR = PROJECT_ROOT / "pilot_stress_results"
WORKER_SCRIPT = SCRIPT_DIR / "_stress_worker.py"

PROBLEM_DIRS = {
    "L1": LOCAL_KB_ROOT / "KernelBench" / "level1",
    "L2": LOCAL_KB_ROOT / "KernelBench" / "level2",
    "L3": LOCAL_KB_ROOT / "KernelBench" / "level3",
    "L4": LOCAL_KB_ROOT / "KernelBench" / "level4",
}

STRESS_TIMEOUT = 180
DEVICE = "cuda"
SEED = 42
ATOL = 1e-2
RTOL = 1e-2
N_SEEDS_PER_POLICY = 3
N_SEEDS_DTYPE = 3
N_SEEDS_REPEATED = 3
REPEATED_RUN_TRIALS = 10


def P(msg):
    print(msg, flush=True)


def localize_path(linux_path: str) -> Path:
    """Convert Linux server path to local Windows path."""
    if linux_path.startswith(LINUX_KB_ROOT):
        relative = linux_path[len(LINUX_KB_ROOT):]
        return LOCAL_KB_ROOT / relative.lstrip("/")
    return Path(linux_path)


def find_problem_file(problem_dir: Path, problem_id) -> Path | None:
    pid = str(problem_id)
    for f in problem_dir.iterdir():
        if f.name.startswith(f"{pid}_") and f.suffix == ".py":
            return f
    return None


def load_all_survived(result_dir: Path):
    details_dir = result_dir / "details"
    survived = []
    for jf in sorted(details_dir.glob("*.json")):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        kernel_name = data["kernel"]["problem_name"]
        for m in data.get("mutants", []):
            if m["status"] == "survived":
                survived.append((kernel_name, data["kernel"], m))
    return survived


def pick_pilot_mutants(survived_list, n=5):
    """Pick n mutants covering different operators."""
    seen_ops = set()
    picked = []

    priority_ops = [
        "stab_remove",      # C-class, expect Type 3 (value stress)
        "cast_remove",       # C-class, expect Type 2 (dtype stress)
        "sync_remove",       # B-class, expect Type 5 (repeated run)
        "arith_replace",     # A-class, may be Type 1 or Type 3
        "epsilon_modify",    # C-class, expect Type 3 (near_zero)
    ]

    for target_op in priority_ops:
        if len(picked) >= n:
            break
        for item in survived_list:
            _, _, mm = item
            if mm["operator_name"] == target_op and target_op not in seen_ops:
                picked.append(item)
                seen_ops.add(target_op)
                break

    if len(picked) < n:
        for item in survived_list:
            if len(picked) >= n:
                break
            _, _, mm = item
            op = mm["operator_name"]
            if op not in seen_ops:
                picked.append(item)
                seen_ops.add(op)

    return picked


def _run_worker(cfg: dict, timeout: int) -> dict | None:
    cfg_fd, cfg_path = tempfile.mkstemp(suffix=".json", prefix="pilotcfg_")
    res_fd, res_path = tempfile.mkstemp(suffix=".json", prefix="pilotres_")
    os.close(cfg_fd)
    os.close(res_fd)

    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    try:
        proc = subprocess.Popen(
            [sys.executable, str(WORKER_SCRIPT), cfg_path, res_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return None

        if os.path.exists(res_path) and os.path.getsize(res_path) > 2:
            with open(res_path) as f:
                return json.load(f)
        if stderr:
            err_text = stderr.decode("utf-8", errors="replace")[-500:]
            P(f"      [stderr] {err_text}")
        return None

    except Exception as e:
        P(f"      [exception] {e}")
        return None

    finally:
        for p in [cfg_path, res_path]:
            try:
                os.unlink(p)
            except OSError:
                pass


def gpu_cleanup():
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


# --- Stress Policy Bank names (inline to avoid torch import at top level) ---
ALL_POLICIES = [
    "large_magnitude", "near_overflow", "near_zero", "denormals",
    "all_negative", "all_positive", "mixed_extremes", "alternating_sign",
    "sparse", "uniform_constant", "structured_ramp", "boundary_last_element",
]

STRATEGY_MAP = {
    "epsilon_modify":    ["near_zero", "denormals"],
    "scale_modify":      ["uniform_constant", "structured_ramp"],
    "stab_remove":       ["large_magnitude", "near_overflow"],
    "cast_remove":       ["near_overflow", "large_magnitude"],
    "init_modify":       ["all_negative", "sparse"],
    "acc_downgrade":     ["mixed_extremes", "large_magnitude"],
    "reduction_reorder": ["mixed_extremes", "alternating_sign"],
    "broadcast_unsafe":  ["structured_ramp"],
    "layout_assume":     ["structured_ramp"],
    "index_replace":     ["structured_ramp", "large_magnitude"],
    "mask_boundary":     ["boundary_last_element", "sparse"],
    "sync_remove":       ["large_magnitude", "mixed_extremes"],
    "launch_config_mutate": ["structured_ramp", "large_magnitude"],
    "arith_replace":     ["large_magnitude", "mixed_extremes"],
    "relop_replace":     ["boundary_last_element", "structured_ramp"],
    "const_perturb":     ["near_zero", "large_magnitude"],
}


def _get_priority_policies(operator_name: str) -> list[str]:
    mapped = STRATEGY_MAP.get(operator_name, [])
    remaining = [p for p in ALL_POLICIES if p not in mapped]
    return mapped + remaining


def run_layer1(problem_file, kernel_code, mutated_code, operator_name):
    """Layer 1: 12 policies × 3 seeds. Returns (killed, killing_policy, killing_seed, orig_failures)."""
    priority = _get_priority_policies(operator_name)
    orig_failures = []

    for pi, policy_name in enumerate(priority):
        for si in range(N_SEEDS_PER_POLICY):
            seed = SEED + pi * N_SEEDS_PER_POLICY + si
            cfg = {
                "mode": "value_stress",
                "problem_file": str(problem_file),
                "kernel_code": kernel_code,
                "mutated_code": mutated_code,
                "policy_name": policy_name,
                "seed": seed,
                "atol": ATOL, "rtol": RTOL, "device": DEVICE,
            }
            data = _run_worker(cfg, timeout=STRESS_TIMEOUT)
            if data is None:
                P(f"      [L1] {policy_name} seed={seed} -> TIMEOUT")
                continue

            ref_ok = data.get("ref_ok", False)
            orig_ok = data.get("original_ok", False)
            mut_ok = data.get("mutant_ok", False)

            if not ref_ok:
                P(f"      [L1] {policy_name} seed={seed} -> REF_FAIL")
                continue
            if orig_ok and not mut_ok:
                P(f"      [L1] {policy_name} seed={seed} -> KILLED!")
                return True, policy_name, seed, orig_failures
            if not orig_ok and not mut_ok:
                P(f"      [L1] {policy_name} seed={seed} -> ORIG_ALSO_FAILS")
                if policy_name not in orig_failures:
                    orig_failures.append(policy_name)
            elif not orig_ok:
                P(f"      [L1] {policy_name} seed={seed} -> ORIG_FAIL_ONLY")
            else:
                P(f"      [L1] {policy_name} seed={seed} -> both OK")

    return False, None, None, orig_failures


def run_layer2(problem_file, kernel_code, mutated_code):
    """Layer 2: 3 seeds × 2 dtypes. Returns (killed, killing_dtype, killing_seed)."""
    for si in range(N_SEEDS_DTYPE):
        seed = SEED + 100 + si
        cfg = {
            "mode": "dtype_stress",
            "problem_file": str(problem_file),
            "kernel_code": kernel_code,
            "mutated_code": mutated_code,
            "seed": seed,
            "atol": ATOL, "rtol": RTOL, "device": DEVICE,
            "target_dtypes": ["float16", "bfloat16"],
        }
        data = _run_worker(cfg, timeout=STRESS_TIMEOUT * 2)
        if data is None:
            P(f"      [L2] seed={seed} -> TIMEOUT")
            continue
        if data.get("killed"):
            dtype = data.get("killing_dtype", "?")
            P(f"      [L2] seed={seed} -> KILLED by dtype={dtype}")
            return True, dtype, seed
        P(f"      [L2] seed={seed} -> not killed")

    return False, None, None


def run_layer3(problem_file, kernel_code, mutated_code):
    """Layer 3: 3 seeds × 10 trials. Returns (killed, divergent_trial, self_inconsistent, killing_seed)."""
    for si in range(N_SEEDS_REPEATED):
        seed = SEED + 200 + si
        cfg = {
            "mode": "repeated_run",
            "problem_file": str(problem_file),
            "kernel_code": kernel_code,
            "mutated_code": mutated_code,
            "seed": seed,
            "atol": ATOL, "rtol": RTOL, "device": DEVICE,
            "n_trials": REPEATED_RUN_TRIALS,
        }
        data = _run_worker(cfg, timeout=STRESS_TIMEOUT * 3)
        if data is None:
            P(f"      [L3] seed={seed} -> TIMEOUT")
            continue
        if data.get("killed"):
            trial = data.get("divergent_trial")
            si_flag = data.get("self_inconsistent", False)
            reason = "self-inconsistency" if si_flag else f"trial {trial}"
            P(f"      [L3] seed={seed} -> KILLED ({reason})")
            return True, trial, si_flag, seed
        P(f"      [L3] seed={seed} -> not killed")

    return False, None, False, None


def attribute_result(killed_layer, orig_failures, operator_name):
    """Simple attribution based on which layer killed (or not)."""
    if killed_layer == 1:
        return "Type 3 (Input-Distribution-Blind)"
    if killed_layer == 2:
        return "Type 2 (Configuration-Blind)"
    if killed_layer == 3:
        return "Type 5 (Non-deterministic)"
    if orig_failures:
        return "Type 4 (Code Defect)"

    KNOWN_EQUIV = {"mask_boundary", "launch_config_mutate"}
    if operator_name in KNOWN_EQUIV:
        return "Type 1 (Algorithmically Absorbed)"

    return "Unresolved"


def main():
    t0 = time.time()
    PILOT_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    with open(BEST_KERNELS_FILE) as f:
        best_kernels = json.load(f)

    P(f"\n{'='*70}")
    P(f"  PILOT Stress Test (5 mutants, 3-layer augmentation)")
    P(f"{'='*70}")

    survived_list = load_all_survived(BLOCK12_RESULT_DIR)
    P(f"  Total survived mutants available: {len(survived_list)}")

    pilots = pick_pilot_mutants(survived_list, n=5)
    P(f"  Selected {len(pilots)} pilot mutants:")
    for i, (kn, _, mm) in enumerate(pilots):
        P(f"    {i+1}. {mm['id']} ({mm['operator_name']}, cat={mm['operator_category']})")
    P("")

    results = []

    for idx, (kernel_name, kernel_meta, mutant_meta) in enumerate(pilots):
        mutant_id = mutant_meta["id"]
        operator = mutant_meta["operator_name"]
        P(f"\n{'_'*70}")
        P(f"  [{idx+1}/5] {mutant_id}")
        P(f"  Operator: {operator} (Category {mutant_meta['operator_category']})")
        P(f"{'_'*70}")

        # Resolve paths
        level_key = f"L{kernel_meta['level']}"
        problem_dir = PROBLEM_DIRS.get(level_key)
        if not problem_dir:
            P(f"    ERROR: unknown level {level_key}, skip")
            continue

        bk = best_kernels.get(kernel_name)
        if not bk:
            P(f"    ERROR: {kernel_name} not in best_kernels.json, skip")
            continue

        kernel_path = localize_path(bk["kernel_path"])
        if not kernel_path.exists():
            P(f"    ERROR: kernel not found: {kernel_path}, skip")
            continue

        problem_file = find_problem_file(problem_dir, kernel_meta["problem_id"])
        if not problem_file:
            P(f"    ERROR: problem file not found for P{kernel_meta['problem_id']}, skip")
            continue

        kernel_code = kernel_path.read_text(encoding="utf-8")
        mutated_code = mutant_meta.get("mutated_code", "")
        if not mutated_code:
            P(f"    ERROR: no mutated_code in JSON, skip")
            continue

        P(f"    kernel_path: {kernel_path}")
        P(f"    problem_file: {problem_file}")
        P(f"    mutated_code length: {len(mutated_code)} chars")

        # --- Layer 1: Value-Distribution Stress ---
        P(f"\n    === Layer 1: Value-Distribution Stress (12 policies × 3 seeds) ===")
        t1 = time.time()
        l1_killed, l1_policy, l1_seed, orig_failures = run_layer1(
            problem_file, kernel_code, mutated_code, operator)
        t1_elapsed = time.time() - t1

        if l1_killed:
            P(f"    -> KILLED in Layer 1 by {l1_policy} (seed={l1_seed}) [{t1_elapsed:.1f}s]")
            attribution = attribute_result(1, orig_failures, operator)
        else:
            P(f"    -> Survived Layer 1 [{t1_elapsed:.1f}s]")

            # --- Layer 2: Configuration Augmentation ---
            P(f"\n    === Layer 2: Configuration Augmentation (3 seeds × 2 dtypes) ===")
            t2 = time.time()
            l2_killed, l2_dtype, l2_seed = run_layer2(
                problem_file, kernel_code, mutated_code)
            t2_elapsed = time.time() - t2

            if l2_killed:
                P(f"    -> KILLED in Layer 2 by dtype={l2_dtype} (seed={l2_seed}) [{t2_elapsed:.1f}s]")
                attribution = attribute_result(2, orig_failures, operator)
            else:
                P(f"    -> Survived Layer 2 [{t2_elapsed:.1f}s]")

                # --- Layer 3: Execution Augmentation ---
                P(f"\n    === Layer 3: Execution Augmentation (3 seeds × 10 trials) ===")
                t3 = time.time()
                l3_killed, l3_trial, l3_si, l3_seed = run_layer3(
                    problem_file, kernel_code, mutated_code)
                t3_elapsed = time.time() - t3

                if l3_killed:
                    reason = "self-inconsistency" if l3_si else f"trial {l3_trial}"
                    P(f"    -> KILLED in Layer 3 ({reason}, seed={l3_seed}) [{t3_elapsed:.1f}s]")
                    attribution = attribute_result(3, orig_failures, operator)
                else:
                    P(f"    -> Survived all 3 layers [{t3_elapsed:.1f}s]")
                    attribution = attribute_result(0, orig_failures, operator)

        P(f"\n    *** Attribution: {attribution} ***")

        result = {
            "mutant_id": mutant_id,
            "operator": operator,
            "category": mutant_meta["operator_category"],
            "kernel": kernel_name,
            "l1_killed": l1_killed,
            "l1_policy": l1_policy,
            "l1_seed": l1_seed,
            "l2_killed": l2_killed if not l1_killed else None,
            "l2_dtype": l2_dtype if not l1_killed else None,
            "l3_killed": l3_killed if not l1_killed and not (l2_killed if not l1_killed else False) else None,
            "original_failures": orig_failures,
            "attribution": attribution,
        }
        results.append(result)
        gpu_cleanup()

    # Save results
    out_path = PILOT_RESULT_DIR / "pilot_5_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    P(f"\n{'='*70}")
    P(f"  PILOT COMPLETE ({elapsed/60:.1f} min)")
    P(f"{'='*70}")
    for r in results:
        status = "KILLED" if any([r.get("l1_killed"), r.get("l2_killed"), r.get("l3_killed")]) else "SURVIVED"
        P(f"  {r['mutant_id']}: {status} -> {r['attribution']}")
    P(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
