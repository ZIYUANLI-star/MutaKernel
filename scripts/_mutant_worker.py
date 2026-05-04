#!/usr/bin/env python3
"""Isolated subprocess worker for mutant execution / equivalence checking.

Usage:
    python _mutant_worker.py <config.json> <result.json>

Modes:
    "run"   – compile and execute a single mutant (killed/survived/stillborn)
    "equiv" – check if a survived mutant is statistically equivalent

Complete process isolation ensures CUDA crashes, compilation hangs,
or illegal memory accesses cannot affect the parent orchestrator.
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _bitwise_identical(a, b):
    """NaN-aware bitwise comparison."""
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.is_floating_point():
            nan_a = torch.isnan(a)
            nan_b = torch.isnan(b)
            if not torch.equal(nan_a, nan_b):
                return False
            finite = ~nan_a
            if finite.any():
                return torch.equal(a[finite], b[finite])
            return True
        return torch.equal(a, b)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_bitwise_identical(x, y) for x, y in zip(a, b))
    return a == b


def _run_mode(cfg):
    """Compile and run a single mutant → killed / survived / stillborn."""
    import torch  # noqa: F401
    from src.models import KernelInfo, Mutant, MutationSite
    from src.mutengine.mutant_runner import MutantRunner
    from src.bridge.eval_bridge import _load_module_from_path

    t0 = time.time()

    kernel = KernelInfo(
        problem_id=cfg["problem_id"],
        level=cfg["level"],
        problem_name=cfg["problem_name"],
        source_path=cfg.get("source_path", ""),
        kernel_code=cfg.get("kernel_code", ""),
        reference_module_path=cfg["problem_file"],
        language=cfg.get("language", "cuda"),
    )
    site = MutationSite(
        line_start=cfg["site"]["line_start"],
        line_end=cfg["site"]["line_end"],
        col_start=cfg["site"].get("col_start", 0),
        col_end=cfg["site"].get("col_end", 0),
        original_code=cfg["site"].get("original_code", ""),
        node_type=cfg["site"].get("node_type", ""),
    )
    mutant = Mutant(
        id=cfg["mutant_id"],
        operator_name=cfg["operator_name"],
        operator_category=cfg["operator_category"],
        site=site,
        original_code=cfg.get("original_code", ""),
        mutated_code=cfg["mutated_code"],
        description=cfg.get("description", ""),
    )
    runner = MutantRunner(
        atol=cfg["atol"], rtol=cfg["rtol"],
        num_test_inputs=cfg["num_test_inputs"],
        device=cfg["device"], seed=cfg["seed"],
    )

    safe_id = cfg["mutant_id"].replace("-", "_").replace(".", "_")
    ref_mod = _load_module_from_path(cfg["problem_file"], f"ref_w_{safe_id}")
    get_inputs = ref_mod.get_inputs
    get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])

    runner.run_mutant(kernel, mutant, ref_mod, get_inputs, get_init_inputs)

    return {
        "status": mutant.status.value,
        "time_ms": mutant.execution_time_ms,
        "error": mutant.error_message or "",
        "kill_seed": mutant.kill_input_seed,
    }


EQUIV_STRESS_POLICIES = [
    "large_magnitude", "near_zero", "structured_ramp",
    "all_negative", "sparse", "boundary_last_element",
]

OPERATOR_DIRECTED_POLICIES = {
    "relop_replace": ["relop_boundary_hit", "boundary_last_element", "structured_ramp",
                       "near_zero", "sparse", "large_magnitude"],
    "arith_replace": ["extreme_magnitude", "large_magnitude", "near_zero",
                       "all_negative", "sparse", "boundary_last_element"],
    "epsilon_modify": ["near_epsilon", "near_zero", "denormals",
                        "large_magnitude", "sparse", "boundary_last_element"],
    "mask_boundary": ["boundary_last_element", "structured_ramp", "head_heavy",
                       "tail_heavy", "sparse", "large_magnitude"],
    "index_replace": ["head_heavy", "tail_heavy", "structured_ramp",
                       "large_magnitude", "sparse", "boundary_last_element"],
    "sync_remove": ["structured_ramp", "head_heavy", "tail_heavy",
                     "large_magnitude", "sparse", "boundary_last_element"],
    "const_perturb": ["near_zero", "boundary_last_element", "sparse",
                       "large_magnitude", "structured_ramp", "all_negative"],
    "launch_config_mutate": ["structured_ramp", "head_heavy", "tail_heavy",
                              "large_magnitude", "sparse", "boundary_last_element"],
}


def _tensor_summary(t):
    """Compact summary of a tensor for reproducibility logs."""
    import torch
    if isinstance(t, torch.Tensor):
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "min": float(t.min()) if t.numel() > 0 else None,
            "max": float(t.max()) if t.numel() > 0 else None,
            "mean": float(t.float().mean()) if t.numel() > 0 else None,
            "has_nan": bool(t.isnan().any()) if t.is_floating_point() else False,
            "has_inf": bool(t.isinf().any()) if t.is_floating_point() else False,
        }
    return {"type": type(t).__name__, "value": str(t)[:100]}


def _equiv_mode(cfg):
    """Check if a survived mutant is statistically equivalent.

    Returns a rich result dict containing full reproducibility information:
    - All seeds and policies tested
    - If divergence found: exact round, seed, policy, input summary
    - If equivalent: complete list of passed rounds
    """
    import torch
    from src.mutengine.mutant_runner import _load_module_from_source, CompilationError
    from src.bridge.eval_bridge import _load_module_from_path
    from src.stress.policy_bank import STRESS_POLICIES

    t0 = time.time()
    device = cfg["device"]
    equiv_runs = cfg.get("equiv_runs", 20)
    base_seed = cfg.get("base_seed", 10000)
    kernel_code = cfg.get("kernel_code", "")
    operator_name = cfg.get("operator_name", "")
    stress_policies = OPERATOR_DIRECTED_POLICIES.get(operator_name, EQUIV_STRESS_POLICIES)

    safe_id = cfg["mutant_id"].replace("-", "_").replace(".", "_")
    tmp_dir = tempfile.mkdtemp(prefix="equiv_iso_")

    ref_mod = _load_module_from_path(cfg["problem_file"], f"ref_eq_{safe_id}")
    get_inputs = ref_mod.get_inputs
    get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
    init_args = get_init_inputs()

    if kernel_code:
        try:
            import hashlib
            orig_hash = hashlib.md5(kernel_code.encode()).hexdigest()[:10]
            orig_mod = _load_module_from_source(
                kernel_code, f"eqo_{orig_hash}", tmp_dir,
            )
        except (CompilationError, Exception) as e:
            return {
                "is_equivalent": False,
                "error": f"OrigCompile: {str(e)[:200]}",
                "time_ms": (time.time() - t0) * 1000,
            }
        orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")
        orig_model = (orig_cls(*init_args) if isinstance(init_args, (list, tuple))
                      else orig_cls())
        orig_model = orig_model.to(device).eval()
    else:
        ref_cls = ref_mod.Model
        orig_model = (ref_cls(*init_args) if isinstance(init_args, (list, tuple))
                      else ref_cls())
        orig_model = orig_model.to(device).eval()

    try:
        mut_mod = _load_module_from_source(
            cfg["mutated_code"], f"eqm_{safe_id}", tmp_dir,
        )
    except CompilationError as e:
        return {
            "is_equivalent": False,
            "error": f"Compilation: {str(e)[:200]}",
            "time_ms": (time.time() - t0) * 1000,
        }

    mut_cls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
    mut_model = (mut_cls(*init_args) if isinstance(init_args, (list, tuple))
                 else mut_cls())
    mut_model = mut_model.to(device).eval()

    def _run_pair(inputs_on_device):
        with torch.no_grad():
            orig_out = orig_model(*inputs_on_device)
            mut_out = mut_model(*inputs_on_device)
        return _bitwise_identical(orig_out, mut_out)

    tested_random_seeds = []
    tested_policies = []
    first_input_summary = None
    last_input_summary = None

    # --- Random seed rounds ---
    for i in range(equiv_runs):
        seed = base_seed + i
        torch.manual_seed(seed)
        inputs = get_inputs()
        moved = [x.to(device) if isinstance(x, torch.Tensor) else x
                 for x in inputs]
        if not _run_pair(moved):
            return {
                "is_equivalent": False,
                "time_ms": (time.time() - t0) * 1000,
                "divergence": {
                    "round_type": "random",
                    "round_index": i,
                    "seed": seed,
                    "policy": None,
                    "input_summary": [_tensor_summary(x) for x in inputs],
                },
                "tested_random_seeds": tested_random_seeds + [seed],
                "tested_policies": tested_policies,
            }
        if i == 0:
            first_input_summary = {
                "round": "random_0", "seed": seed,
                "tensors": [_tensor_summary(x) for x in inputs],
            }
        last_input_summary = {
            "round": f"random_{i}", "seed": seed,
            "tensors": [_tensor_summary(x) for x in inputs],
        }
        tested_random_seeds.append(seed)

    # --- Stress policy rounds ---
    for policy_name in stress_policies:
        policy_fn = STRESS_POLICIES.get(policy_name)
        if policy_fn is None:
            continue
        for si in range(2):
            seed = base_seed + equiv_runs + si
            torch.manual_seed(seed)
            try:
                template = get_inputs()
                stress_inputs = policy_fn(template, seed)
            except Exception:
                tested_policies.append({
                    "name": policy_name, "sub_index": si,
                    "seed": seed, "status": "generation_failed"})
                continue
            moved = [x.to(device) if isinstance(x, torch.Tensor) else x
                     for x in stress_inputs]

            orig_exc = mut_exc = None
            orig_out = mut_out = None
            try:
                with torch.no_grad():
                    orig_out = orig_model(*moved)
            except Exception as e:
                orig_exc = e
            try:
                with torch.no_grad():
                    mut_out = mut_model(*moved)
            except Exception as e:
                mut_exc = e

            if orig_exc is not None and mut_exc is not None:
                if type(orig_exc) is type(mut_exc):
                    tested_policies.append({
                        "name": policy_name, "sub_index": si,
                        "seed": seed, "status": "both_exception_same_type"})
                    continue
                return {
                    "is_equivalent": False,
                    "time_ms": (time.time() - t0) * 1000,
                    "divergence": {
                        "round_type": "stress",
                        "policy": policy_name, "sub_index": si,
                        "seed": seed,
                        "detail": "diff_exception",
                        "input_summary": [_tensor_summary(x)
                                          for x in stress_inputs],
                    },
                    "tested_random_seeds": tested_random_seeds,
                    "tested_policies": tested_policies,
                }
            if orig_exc is not None or mut_exc is not None:
                oom_msg = str(orig_exc or mut_exc).lower()
                if "out of memory" in oom_msg:
                    tested_policies.append({
                        "name": policy_name, "sub_index": si,
                        "seed": seed, "status": "oom_skipped"})
                    continue
                return {
                    "is_equivalent": False,
                    "time_ms": (time.time() - t0) * 1000,
                    "divergence": {
                        "round_type": "stress",
                        "policy": policy_name, "sub_index": si,
                        "seed": seed,
                        "detail": "one_side_exception",
                        "orig_exc": str(orig_exc)[:200] if orig_exc else None,
                        "mut_exc": str(mut_exc)[:200] if mut_exc else None,
                        "input_summary": [_tensor_summary(x)
                                          for x in stress_inputs],
                    },
                    "tested_random_seeds": tested_random_seeds,
                    "tested_policies": tested_policies,
                }
            if not _bitwise_identical(orig_out, mut_out):
                return {
                    "is_equivalent": False,
                    "time_ms": (time.time() - t0) * 1000,
                    "divergence": {
                        "round_type": "stress",
                        "policy": policy_name, "sub_index": si,
                        "seed": seed,
                        "detail": "output_diverged",
                        "input_summary": [_tensor_summary(x)
                                          for x in stress_inputs],
                    },
                    "tested_random_seeds": tested_random_seeds,
                    "tested_policies": tested_policies,
                }
            tested_policies.append({
                "name": policy_name, "sub_index": si,
                "seed": seed, "status": "passed"})
            last_input_summary = {
                "round": f"stress_{policy_name}_{si}", "seed": seed,
                "tensors": [_tensor_summary(x) for x in stress_inputs],
            }

    return {
        "is_equivalent": True,
        "time_ms": (time.time() - t0) * 1000,
        "tested_random_seeds": tested_random_seeds,
        "tested_policies": tested_policies,
        "total_rounds": len(tested_random_seeds) + len(tested_policies),
        "first_input_summary": first_input_summary,
        "last_input_summary": last_input_summary,
    }


def main():
    cfg_path, res_path = sys.argv[1], sys.argv[2]
    with open(cfg_path) as f:
        cfg = json.load(f)

    t0 = time.time()
    mode = cfg.get("mode", "run")

    try:
        if mode == "run":
            result = _run_mode(cfg)
        elif mode == "equiv":
            result = _equiv_mode(cfg)
        else:
            result = {"status": "stillborn", "error": f"Unknown mode: {mode}"}
    except Exception as e:
        if mode == "run":
            result = {
                "status": "stillborn",
                "error": f"WorkerCrash: {str(e)[:300]}",
                "time_ms": (time.time() - t0) * 1000,
                "kill_seed": None,
            }
        else:
            result = {
                "is_equivalent": False,
                "error": f"EquivCrash: {str(e)[:300]}",
                "time_ms": (time.time() - t0) * 1000,
            }

    with open(res_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
