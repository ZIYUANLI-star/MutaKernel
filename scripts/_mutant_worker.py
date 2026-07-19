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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


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
        mean = (
            float(t.mean().item())
            if t.numel() > 0 and t.is_floating_point()
            else None
        )
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "min": float(t.min()) if t.numel() > 0 else None,
            "max": float(t.max()) if t.numel() > 0 else None,
            "mean": mean,
            "has_nan": bool(t.isnan().any()) if t.is_floating_point() else False,
            "has_inf": bool(t.isinf().any()) if t.is_floating_point() else False,
        }
    return {"type": type(t).__name__, "value": str(t)[:100]}


def _equiv_mode(cfg):
    """Dynamically challenge a survived mutant using sound paired execution.

    ``is_equivalent`` is retained for legacy consumers, but it is deliberately
    three-valued: ``True`` means no divergence was observed, ``False`` means a
    concrete non-equivalence witness was observed, and ``None`` means that the
    comparison was inconclusive.  Passing tests are evidence, not a proof of
    semantic equivalence.

    Final labels are derived only from :mod:`src.validation`.  An LLM may be
    used by an outer workflow to suggest inputs, but no LLM verdict is accepted
    by this worker as an equivalence label.
    """
    import hashlib
    import random

    import torch

    from src.bridge.eval_bridge import _load_module_from_path
    from src.mutengine.mutant_runner import (
        CompilationError,
        _load_module_from_source,
        _move_tree_to_device,
    )
    from src.stress.policy_bank import STRESS_POLICIES
    from src.validation import (
        ExecutionConfig,
        OracleConfig,
        RNGSnapshot,
        Tolerance,
        ValidationStatus,
        clone_tree,
        validate_pair,
    )

    t0 = time.time()
    device = cfg["device"]
    equiv_runs = int(cfg.get("equiv_runs", 20))
    base_seed = int(cfg.get("base_seed", 10000))
    kernel_code = cfg.get("kernel_code", "")
    operator_name = cfg.get("operator_name", "")
    stress_policies = cfg.get(
        "stress_policies",
        OPERATOR_DIRECTED_POLICIES.get(operator_name, EQUIV_STRESS_POLICIES),
    )
    stress_repeats = int(cfg.get("stress_repeats", 2))

    safe_id = cfg["mutant_id"].replace("-", "_").replace(".", "_")
    tested_random_seeds = []
    tested_policies = []
    trials = []
    errors = []
    first_input_summary = None
    last_input_summary = None
    saw_inconclusive = False
    valid_rounds = 0

    def _seed_all(seed):
        random.seed(seed)
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - NumPy is optional.
            np = None
        if np is not None:
            np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _summaries(values):
        summaries = []
        for value in values:
            try:
                summaries.append(_tensor_summary(value))
            except Exception as exc:  # Logging must never decide correctness.
                summaries.append({
                    "type": type(value).__name__,
                    "summary_error": f"{type(exc).__name__}: {str(exc)[:160]}",
                })
        return summaries

    def _normalise_args(generated):
        if isinstance(generated, (list, tuple)):
            args = tuple(generated)
        else:
            args = (generated,)
        return tuple(_move_tree_to_device(args, device))

    def _error(phase, exc, **context):
        record = {
            "phase": phase,
            "exception_type": type(exc).__name__,
            "message": str(exc)[:300],
        }
        record.update(context)
        errors.append(record)
        return record

    def _base_result(status, reason, is_equivalent, *, divergence=None):
        result = {
            # Legacy fields.
            "is_equivalent": is_equivalent,
            "time_ms": (time.time() - t0) * 1000,
            "tested_random_seeds": tested_random_seeds,
            "tested_policies": tested_policies,
            "total_rounds": len(trials),
            "first_input_summary": first_input_summary,
            "last_input_summary": last_input_summary,
            # Sound, three-valued fields.
            "validation_status": status.value,
            "reason": reason,
            "errors": errors,
            "valid_rounds": valid_rounds,
            "trials": trials,
        }
        if status is ValidationStatus.INCONCLUSIVE:
            result["error"] = reason[:300]
        if divergence is not None:
            result["divergence"] = divergence
        return result

    def _setup_inconclusive(phase, exc):
        _error(phase, exc)
        return _base_result(
            ValidationStatus.INCONCLUSIVE,
            f"{phase} failed; equivalence is unknown: "
            f"{type(exc).__name__}: {str(exc)[:200]}",
            None,
        )

    def _validation_errors(verdict, **context):
        serialised = verdict.to_dict()
        for entry in serialised["errors"]:
            enriched = dict(entry)
            enriched.update(context)
            errors.append(enriched)
        return serialised

    def _evaluate(args, metadata, input_summary):
        nonlocal saw_inconclusive, valid_rounds
        with torch.no_grad():
            verdict = validate_pair(
                reference=orig_model,
                candidate=mut_model,
                args=args,
                oracle_config=oracle_config,
                execution_config=execution_config,
            )
        serialised = _validation_errors(verdict, **metadata)
        trial = {**metadata, **serialised}
        trials.append(trial)

        if verdict.status is ValidationStatus.PASS:
            valid_rounds += 1
            return None
        if verdict.status is ValidationStatus.INCONCLUSIVE:
            saw_inconclusive = True
            return None

        # FAIL is emitted by ValidationExecutor only after the reference ran
        # successfully and either the candidate concretely diverged or failed.
        candidate_crash = any(
            item.get("phase") == "candidate" for item in serialised["errors"]
        )
        divergence = {
            **metadata,
            "detail": "candidate_crash" if candidate_crash else "output_diverged",
            "input_summary": input_summary,
            "oracle": serialised["oracle"],
            "errors": serialised["errors"],
        }
        return _base_result(
            ValidationStatus.FAIL,
            verdict.reason,
            False,
            divergence=divergence,
        )

    caller_rng = RNGSnapshot.capture(include_cuda=True)
    try:
        with tempfile.TemporaryDirectory(prefix="equiv_iso_") as tmp_dir:
            try:
                ref_mod = _load_module_from_path(
                    cfg["problem_file"], f"ref_eq_{safe_id}",
                )
                get_inputs = ref_mod.get_inputs
                get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
            except Exception as exc:
                return _setup_inconclusive("reference_load", exc)

            try:
                _seed_all(base_seed)
                init_args = get_init_inputs()
            except Exception as exc:
                return _setup_inconclusive("initial_input_generation", exc)

            if kernel_code:
                try:
                    orig_hash = hashlib.md5(kernel_code.encode()).hexdigest()[:10]
                    orig_mod = _load_module_from_source(
                        kernel_code, f"eqo_{orig_hash}", tmp_dir,
                    )
                    orig_cls = (
                        getattr(orig_mod, "ModelNew", None)
                        or getattr(orig_mod, "Model")
                    )
                except (CompilationError, Exception) as exc:
                    return _setup_inconclusive("reference_compile", exc)
            else:
                try:
                    orig_cls = ref_mod.Model
                except Exception as exc:
                    return _setup_inconclusive("reference_class", exc)

            try:
                mut_mod = _load_module_from_source(
                    cfg["mutated_code"], f"eqm_{safe_id}", tmp_dir,
                )
                mut_cls = (
                    getattr(mut_mod, "ModelNew", None)
                    or getattr(mut_mod, "Model")
                )
            except (CompilationError, Exception) as exc:
                return _setup_inconclusive("candidate_compile", exc)

            try:
                _seed_all(base_seed)
                constructor_rng = RNGSnapshot.capture(include_cuda=True)
                constructor_rng.restore()
                orig_model = (
                    orig_cls(*clone_tree(init_args))
                    if isinstance(init_args, (list, tuple))
                    else orig_cls()
                )
                constructor_rng.restore()
                mut_model = (
                    mut_cls(*clone_tree(init_args))
                    if isinstance(init_args, (list, tuple))
                    else mut_cls()
                )
                orig_model = orig_model.to(device).eval()
                mut_model = mut_model.to(device).eval()
            except Exception as exc:
                return _setup_inconclusive("model_initialisation", exc)

            tolerance = Tolerance(
                rtol=float(cfg.get("rtol", 0.0)),
                atol=float(cfg.get("atol", 0.0)),
            )
            oracle_config = OracleConfig(
                default_tolerance=tolerance,
                dtype_tolerances={
                    dtype: tolerance
                    for dtype in (
                        torch.float16,
                        torch.bfloat16,
                        torch.float32,
                        torch.float64,
                        torch.complex64,
                        torch.complex128,
                    )
                },
                require_dtype=True,
                require_device=True,
                require_layout=True,
            )
            execution_config = ExecutionConfig(
                synchronize_state=True,
                preserve_module_state=True,
                preserve_caller_rng=True,
                include_cuda_rng=True,
                synchronize_cuda_timing=True,
                retain_outputs=False,
            )

            # --- Random seed rounds ---
            for i in range(equiv_runs):
                seed = base_seed + i
                tested_random_seeds.append(seed)
                metadata = {
                    "round_type": "random",
                    "round_index": i,
                    "seed": seed,
                    "policy": None,
                }
                try:
                    _seed_all(seed)
                    generated = get_inputs()
                    input_values = (
                        list(generated)
                        if isinstance(generated, (list, tuple))
                        else [generated]
                    )
                    input_summary = _summaries(input_values)
                    args = _normalise_args(generated)
                except Exception as exc:
                    saw_inconclusive = True
                    _error("input_generation", exc, **metadata)
                    trials.append({
                        **metadata,
                        "status": ValidationStatus.INCONCLUSIVE.value,
                        "reason": (
                            "input generation failed; equivalence is unknown: "
                            f"{type(exc).__name__}: {str(exc)[:200]}"
                        ),
                        "errors": [errors[-1]],
                    })
                    continue

                if first_input_summary is None:
                    first_input_summary = {
                        "round": f"random_{i}",
                        "seed": seed,
                        "tensors": input_summary,
                    }
                last_input_summary = {
                    "round": f"random_{i}",
                    "seed": seed,
                    "tensors": input_summary,
                }
                failure = _evaluate(args, metadata, input_summary)
                if failure is not None:
                    return failure

            # --- Stress policy rounds ---
            for policy_name in stress_policies:
                policy_fn = STRESS_POLICIES.get(policy_name)
                if policy_fn is None:
                    saw_inconclusive = True
                    missing = KeyError(f"unknown stress policy: {policy_name}")
                    _error("stress_policy_lookup", missing, policy=policy_name)
                    tested_policies.append({
                        "name": policy_name,
                        "status": "policy_missing",
                    })
                    continue

                for si in range(stress_repeats):
                    seed = base_seed + equiv_runs + si
                    metadata = {
                        "round_type": "stress",
                        "policy": policy_name,
                        "sub_index": si,
                        "seed": seed,
                    }
                    policy_record = {
                        "name": policy_name,
                        "sub_index": si,
                        "seed": seed,
                    }
                    tested_policies.append(policy_record)
                    try:
                        _seed_all(seed)
                        template = get_inputs()
                        stress_inputs = policy_fn(clone_tree(template), seed)
                        input_values = (
                            list(stress_inputs)
                            if isinstance(stress_inputs, (list, tuple))
                            else [stress_inputs]
                        )
                        input_summary = _summaries(input_values)
                        args = _normalise_args(stress_inputs)
                    except Exception as exc:
                        saw_inconclusive = True
                        policy_record["status"] = "generation_failed"
                        _error("stress_input_generation", exc, **metadata)
                        trials.append({
                            **metadata,
                            "status": ValidationStatus.INCONCLUSIVE.value,
                            "reason": (
                                "stress input generation failed; equivalence is unknown: "
                                f"{type(exc).__name__}: {str(exc)[:200]}"
                            ),
                            "errors": [errors[-1]],
                        })
                        continue

                    failure = _evaluate(args, metadata, input_summary)
                    policy_record["status"] = (
                        trials[-1]["status"] if failure is None else "non_equivalent"
                    )
                    last_input_summary = {
                        "round": f"stress_{policy_name}_{si}",
                        "seed": seed,
                        "tensors": input_summary,
                    }
                    if first_input_summary is None:
                        first_input_summary = last_input_summary
                    if failure is not None:
                        return failure

            if saw_inconclusive or valid_rounds == 0:
                reason = (
                    "one or more rounds were inconclusive; no sound "
                    "non-equivalence witness was observed"
                    if saw_inconclusive
                    else "no valid validation round completed; equivalence is unknown"
                )
                return _base_result(
                    ValidationStatus.INCONCLUSIVE,
                    reason,
                    None,
                )
            return _base_result(
                ValidationStatus.PASS,
                "no divergence was observed in the completed validation rounds; "
                "this is not a proof of semantic equivalence",
                True,
            )
    finally:
        caller_rng.restore()


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
                "is_equivalent": None,
                "error": f"EquivCrash: {str(e)[:300]}",
                "validation_status": "inconclusive",
                "reason": (
                    "equivalence worker crashed; equivalence is unknown: "
                    f"{type(e).__name__}: {str(e)[:300]}"
                ),
                "errors": [{
                    "phase": "worker",
                    "exception_type": type(e).__name__,
                    "message": str(e)[:300],
                }],
                "time_ms": (time.time() - t0) * 1000,
            }

    with open(res_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
