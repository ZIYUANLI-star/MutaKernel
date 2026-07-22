#!/usr/bin/env python3
from __future__ import annotations

"""Isolated subprocess worker for stress differential testing.

Usage:
    python _stress_worker.py <config.json> <result.json>

For a given (original_kernel, mutant_kernel, stress_policy):
  1. Load reference implementation and generate stress input
  2. Run ref → check for NaN/Inf (if so, input is invalid)
  3. Run original kernel with stress input → original_ok
  4. Run mutant kernel with stress input → mutant_ok
  5. Write results to JSON
"""
import hashlib
import functools
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def _code_hash(code: str) -> str:
    """Short hash of kernel code for stable CUDA extension naming."""
    return hashlib.md5(code.encode()).hexdigest()[:10]


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))


_INFRASTRUCTURE_TERMS = (
    "out of memory",
    "cuda oom",
    "timed out",
    "timeout",
    "gpu_preflight_fail",
    "workercrash",
)

_SETUP_TERMS = (
    "unknown policy",
    "build:",
    "orig compile",
    "get_inputs_error",
    "rebatch_error",
    "dtype_unsupported",
    "unsupported dtype",
    "llm_code_exec",
    "llm_generate",
    "missing generate_inputs",
    "state_dict keys differ",
    "state synchronization",
)


def _collect_error_messages(value, path="result", limit=20):
    """Collect JSON-safe diagnostics without treating outputs as exceptions."""
    found = []

    def visit(item, item_path):
        if len(found) >= limit:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{item_path}.{key}"
                if key in {"error", "mut_error"} and child:
                    found.append({
                        "phase": child_path,
                        "type": "ExecutionError",
                        "message": str(child)[:500],
                    })
                elif key in {"status", "reason"} and isinstance(child, str) and any(
                    token in child.lower()
                    for token in ("crash", "error", "diverge", "unsupported", "nan_inf")
                ):
                    found.append({
                        "phase": child_path,
                        "type": "ValidationDiagnostic",
                        "message": child[:500],
                    })
                else:
                    visit(child, child_path)
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{item_path}[{index}]")

    visit(value, path)
    return found


def _normalize_validation_result(result):
    """Add three-valued semantics while preserving legacy JSON fields.

    Legacy consumers infer a kill from ``original_ok and not mutant_ok``.  For
    inconclusive runs we conservatively clear those booleans as well as
    ``killed`` so an infrastructure or oracle failure cannot become a kill.
    """
    if not isinstance(result, dict):
        result = {"error": f"worker returned {type(result).__name__}, expected dict"}
    result = dict(result)
    if "bitwise_orig_mut_eq" in result:
        # Bitwise determinism is not part of the default numerical correctness
        # contract.  Older orchestrators treat False as an unconditional kill,
        # so retain the observation separately and neutralize that legacy path.
        result.setdefault(
            "observed_bitwise_orig_mut_eq",
            bool(result["bitwise_orig_mut_eq"]),
        )
        result["bitwise_orig_mut_eq"] = True
    errors = list(result.get("errors") or [])
    errors.extend(_collect_error_messages(result))

    def contains_invalid_baseline(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {"ref_ok", "orig_ok", "original_ok"} and child is False:
                    return True
                if contains_invalid_baseline(child):
                    return True
        elif isinstance(value, (list, tuple)):
            return any(contains_invalid_baseline(child) for child in value)
        return False

    text = " ".join(str(error.get("message", "")) for error in errors).lower()
    explicit_status = result.get("validation_status")
    # Infrastructure and invalid-baseline evidence always override a stale or
    # optimistic status supplied by a caller.
    if any(term in text for term in _INFRASTRUCTURE_TERMS):
        status = "inconclusive"
    elif contains_invalid_baseline(result):
        status = "inconclusive"
    elif any(term in text for term in _SETUP_TERMS):
        status = "inconclusive"
    elif explicit_status in {"pass", "fail", "inconclusive"}:
        status = explicit_status
    elif result.get("killed") is True:
        status = "fail"
    elif (
        result.get("ref_ok") is True
        and result.get("original_ok") is True
        and result.get("mutant_ok") is False
    ):
        status = "fail"
    elif errors:
        status = "inconclusive"
    else:
        status = "pass"

    existing_reason = result.get("reason")
    if status == "pass":
        reason = existing_reason or "all valid reference/original/candidate comparisons passed"
    elif status == "fail":
        reason = existing_reason or "candidate crashed or diverged under a valid reference/original comparison"
        result["killed"] = True
    else:
        reason = existing_reason or (
            errors[0]["message"] if errors else "validation could not make a sound comparison"
        )
        # Preserve any observed values separately, then encode the conservative
        # legacy state so older aggregation scripts cannot count a false kill.
        result.setdefault("observed_ref_ok", result.get("ref_ok"))
        result.setdefault("observed_original_ok", result.get("original_ok"))
        result.setdefault("observed_mutant_ok", result.get("mutant_ok"))
        result["ref_ok"] = False
        result["original_ok"] = False
        result["mutant_ok"] = False
        result["killed"] = False

    result["validation_status"] = status
    result["reason"] = reason
    result["errors"] = errors
    return result


def _sound_mode_result(function):
    """Make every mode return the shared legacy-compatible three-value schema."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        try:
            return _normalize_validation_result(function(*args, **kwargs))
        except Exception as exc:
            return _normalize_validation_result({
                "ref_ok": False,
                "original_ok": False,
                "mutant_ok": False,
                "killed": False,
                "error": f"validator setup failed: {type(exc).__name__}: {str(exc)[:500]}",
                "validation_status": "inconclusive",
                "time_ms": 0.0,
            })

    return wrapped


def _has_nan_inf(out):
    import torch
    if isinstance(out, torch.Tensor):
        return torch.isnan(out).any().item() or torch.isinf(out).any().item()
    if isinstance(out, (tuple, list)):
        return any(_has_nan_inf(x) for x in out)
    return False


def _allclose(a, b, atol, rtol):
    import torch
    from src.validation import OracleConfig, Tolerance, ValidationStatus, compare_outputs

    tolerance = Tolerance(rtol=float(rtol), atol=float(atol))
    configured_dtypes = {
        torch.float16: tolerance,
        torch.bfloat16: tolerance,
        torch.float32: tolerance,
        torch.float64: tolerance,
        torch.complex64: tolerance,
        torch.complex128: tolerance,
    }
    comparison = compare_outputs(
        a,
        b,
        OracleConfig(
            default_tolerance=tolerance,
            dtype_tolerances=configured_dtypes,
            equal_nan=True,
            require_dtype=True,
            require_device=True,
            require_layout=True,
        ),
    )
    return comparison.status is ValidationStatus.PASS


def _call_isolated(model, inputs):
    """Invoke a model with a recursively cloned input tree."""
    from src.validation import clone_call_inputs

    args, kwargs = clone_call_inputs(tuple(inputs), {})
    return model(*args, **kwargs)


def _compute_diff_summary(ref_out, orig_out):
    """Compute error magnitude between reference and original outputs."""
    import torch
    if isinstance(ref_out, torch.Tensor) and isinstance(orig_out, torch.Tensor):
        if ref_out.shape != orig_out.shape:
            return f"shape_mismatch: ref={list(ref_out.shape)}, orig={list(orig_out.shape)}"
        diff = (ref_out.float().cpu() - orig_out.float().cpu()).abs()
        return (
            f"max_diff={diff.max().item():.6e}, "
            f"mean_diff={diff.mean().item():.6e}, "
            f"ref_range=[{ref_out.float().min().item():.4e},{ref_out.float().max().item():.4e}], "
            f"orig_range=[{orig_out.float().min().item():.4e},{orig_out.float().max().item():.4e}]"
        )
    if isinstance(ref_out, (tuple, list)) and isinstance(orig_out, (tuple, list)):
        for i, (ro, oo) in enumerate(zip(ref_out, orig_out)):
            if isinstance(ro, torch.Tensor) and isinstance(oo, torch.Tensor):
                if ro.shape != oo.shape:
                    return f"shape_mismatch[{i}]: ref={list(ro.shape)}, orig={list(oo.shape)}"
                diff = (ro.float().cpu() - oo.float().cpu()).abs()
                return (
                    f"output[{i}]: max_diff={diff.max().item():.6e}, "
                    f"mean_diff={diff.mean().item():.6e}, "
                    f"ref_range=[{ro.float().min().item():.4e},{ro.float().max().item():.4e}], "
                    f"orig_range=[{oo.float().min().item():.4e},{oo.float().max().item():.4e}]"
                )
    return ""


def _sync_weights(src_model, dst_model):
    """Synchronize state by exact keys; failures are never guessed or hidden."""
    from src.validation import strict_sync_state_dict

    return strict_sync_state_dict(src_model, dst_model)


def _capture_init_rng():
    """Snapshot RNG state before the reference model is constructed."""
    from src.validation import RNGSnapshot

    return RNGSnapshot.capture()


def _instantiate_model(cls, init_args, snapshot=None):
    """Construct a model, optionally replaying the pre-reference RNG state.

    Replaying the snapshot makes paired constructions draw identical
    initialization entropy (方法V2_01 §3.1 double insurance).  Strict named
    state synchronization remains the second layer and still overwrites
    parameters afterwards; the replay additionally covers construction-time
    randomness that never reaches the state dict.
    """
    from src.validation import replay_rng

    def _make():
        return cls(*init_args) if isinstance(init_args, (list, tuple)) else cls()

    if snapshot is None:
        return _make()
    with replay_rng(snapshot):
        return _make()


@_sound_mode_result
def run_stress(cfg):
    import torch
    from src.mutengine.mutant_runner import _load_module_from_source, CompilationError
    from src.bridge.eval_bridge import _load_module_from_path
    from src.stress.policy_bank import STRESS_POLICIES

    t0 = time.time()
    device = cfg["device"]
    atol = cfg["atol"]
    rtol = cfg["rtol"]
    policy_name = cfg["policy_name"]
    seed = cfg["seed"]

    orig_hash = _code_hash(cfg["kernel_code"])
    mut_hash = _code_hash(cfg["mutated_code"])
    ref_hash = _code_hash(cfg["problem_file"])
    is_same_code = (cfg["kernel_code"] == cfg["mutated_code"])

    ref_mod = _load_module_from_path(cfg["problem_file"], f"ext_ref_{ref_hash}")
    get_inputs = ref_mod.get_inputs
    get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])

    torch.manual_seed(seed)
    template_inputs = get_inputs()

    if policy_name == "__identity__":
        stress_inputs = template_inputs
    else:
        policy_fn = STRESS_POLICIES.get(policy_name)
        if policy_fn is None:
            return {"error": f"Unknown policy: {policy_name}", "ref_ok": False,
                    "original_ok": False, "mutant_ok": False}
        stress_inputs = policy_fn(template_inputs, seed)

    stress_on_device = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in stress_inputs
    ]

    init_args = get_init_inputs()
    ref_cls = ref_mod.Model
    init_rng = _capture_init_rng()
    ref_model = _instantiate_model(ref_cls, init_args)
    ref_model = ref_model.to(device).eval()
    with torch.no_grad():
        try:
            ref_out = _call_isolated(ref_model, stress_on_device)
        except Exception as e:
            return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                    "error": f"ref crash: {str(e)[:200]}",
                    "time_ms": (time.time() - t0) * 1000}

    ref_nan = _has_nan_inf(ref_out)

    tmp_dir = tempfile.mkdtemp(prefix="stress_")

    try:
        orig_mod = _load_module_from_source(
            cfg["kernel_code"], f"ext_orig_{orig_hash}", tmp_dir,
        )
    except Exception as e:
        return {"ref_ok": not ref_nan, "original_ok": False, "mutant_ok": False,
                "error": f"orig compile: {str(e)[:200]}",
                "time_ms": (time.time() - t0) * 1000}

    orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")
    orig_model = _instantiate_model(orig_cls, init_args, init_rng)
    orig_model = orig_model.to(device).eval()
    _sync_weights(ref_model, orig_model)

    original_ok = False
    orig_out = None
    with torch.no_grad():
        try:
            orig_out = _call_isolated(orig_model, stress_on_device)
            original_ok = _allclose(ref_out, orig_out, atol, rtol)
        except Exception:
            original_ok = False

    if ref_nan and not original_ok:
        return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                "ref_nan_fallback": True,
                "error": "ref NaN/Inf and original also invalid",
                "time_ms": (time.time() - t0) * 1000}

    compare_target = ref_out

    if is_same_code:
        mutant_ok = original_ok
        bitwise_orig_mut_eq = True
        mut_out = orig_out
        mut_error = ""
    else:
        try:
            mut_mod = _load_module_from_source(
                cfg["mutated_code"], f"ext_mut_{mut_hash}", tmp_dir,
            )
        except Exception as e:
            return {"ref_ok": True, "original_ok": original_ok, "mutant_ok": False,
                    "ref_nan_fallback": ref_nan,
                    "error": f"mut compile: {str(e)[:200]}",
                    "time_ms": (time.time() - t0) * 1000}

        mut_cls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
        mut_model = _instantiate_model(mut_cls, init_args, init_rng)
        mut_model = mut_model.to(device).eval()
        _sync_weights(ref_model, mut_model)

        mutant_ok = False
        mut_error = ""
        with torch.no_grad():
            try:
                mut_out = _call_isolated(mut_model, stress_on_device)
                mutant_ok = _allclose(compare_target, mut_out, atol, rtol)
            except Exception as e:
                mutant_ok = False
                mut_error = f"mutant crash: {str(e)[:200]}"

        bitwise_orig_mut_eq = False
        if orig_out is not None and mutant_ok is not None:
            try:
                bitwise_orig_mut_eq = _bitwise_eq(orig_out, mut_out)
            except Exception:
                pass

    diff_summary = ""
    if not original_ok and orig_out is not None and not ref_nan:
        try:
            diff_summary = _compute_diff_summary(ref_out, orig_out)
        except Exception:
            pass

    return {
        "ref_ok": True,
        "ref_nan_fallback": ref_nan,
        "original_ok": original_ok,
        "mutant_ok": mutant_ok,
        "bitwise_orig_mut_eq": bitwise_orig_mut_eq,
        "diff_summary": diff_summary,
        "error": mut_error,
        "time_ms": (time.time() - t0) * 1000,
    }


def _build_models(cfg, seed_suffix, device):
    """Shared helper: load ref, orig, mut models and return (ref_model, orig_model, mut_model, get_inputs, init_args, tmp_dir)."""
    import torch
    from src.mutengine.mutant_runner import _load_module_from_source
    from src.bridge.eval_bridge import _load_module_from_path

    orig_hash = _code_hash(cfg["kernel_code"])
    mut_hash = _code_hash(cfg["mutated_code"])
    ref_hash = _code_hash(cfg["problem_file"])

    is_same_code = (cfg["kernel_code"] == cfg["mutated_code"])

    ref_mod = _load_module_from_path(cfg["problem_file"], f"ext_ref_{ref_hash}")
    get_inputs = ref_mod.get_inputs
    get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
    init_args = get_init_inputs()
    ref_cls = ref_mod.Model
    init_rng = _capture_init_rng()
    ref_model = _instantiate_model(ref_cls, init_args)
    ref_model = ref_model.to(device).eval()

    tmp_dir = tempfile.mkdtemp(prefix="stress_")

    orig_mod = _load_module_from_source(
        cfg["kernel_code"], f"ext_orig_{orig_hash}", tmp_dir)
    orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")
    orig_model = _instantiate_model(orig_cls, init_args, init_rng)
    orig_model = orig_model.to(device).eval()
    _sync_weights(ref_model, orig_model)

    if is_same_code:
        mut_model = orig_model
    else:
        mut_mod = _load_module_from_source(
            cfg["mutated_code"], f"ext_mut_{mut_hash}", tmp_dir)
        mut_cls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
        mut_model = _instantiate_model(mut_cls, init_args, init_rng)
        mut_model = mut_model.to(device).eval()
        _sync_weights(ref_model, mut_model)

    return ref_model, orig_model, mut_model, get_inputs, init_args, tmp_dir


@_sound_mode_result
def run_training_stress(cfg):
    """Training-Mode Augmentation: re-run stress policies with models in .train() mode.

    In eval mode, BatchNorm/LayerNorm use fixed running_var (typically 1.0),
    masking eps/scale/init mutations.  In training mode, the actual batch
    statistics are computed from the stress input, exposing differences like
    rsqrt(tiny_var + 0) vs rsqrt(tiny_var + 1e-5).

    Returns the same schema as run_stress() but with models in .train().
    """
    import torch
    from src.mutengine.mutant_runner import _load_module_from_source, CompilationError
    from src.bridge.eval_bridge import _load_module_from_path
    from src.stress.policy_bank import STRESS_POLICIES

    t0 = time.time()
    device = cfg["device"]
    atol = cfg.get("atol", 1e-2)
    rtol = cfg.get("rtol", 1e-2)
    policy_name = cfg["policy_name"]
    seed = cfg["seed"]

    orig_hash = _code_hash(cfg["kernel_code"])
    mut_hash = _code_hash(cfg["mutated_code"])
    ref_hash = _code_hash(cfg["problem_file"])
    is_same_code = (cfg["kernel_code"] == cfg["mutated_code"])

    ref_mod = _load_module_from_path(cfg["problem_file"], f"ext_ref_{ref_hash}")
    get_inputs = ref_mod.get_inputs
    get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])

    torch.manual_seed(seed)
    template_inputs = get_inputs()

    if policy_name == "__identity__":
        stress_inputs = template_inputs
    else:
        policy_fn = STRESS_POLICIES.get(policy_name)
        if policy_fn is None:
            return {"error": f"Unknown policy: {policy_name}", "ref_ok": False,
                    "original_ok": False, "mutant_ok": False}
        stress_inputs = policy_fn(template_inputs, seed)

    stress_on_device = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in stress_inputs
    ]

    init_args = get_init_inputs()

    ref_cls = ref_mod.Model
    init_rng = _capture_init_rng()
    ref_model = _instantiate_model(ref_cls, init_args)
    ref_model = ref_model.to(device).train()
    with torch.no_grad():
        try:
            ref_out = _call_isolated(ref_model, stress_on_device)
        except Exception as e:
            return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                    "error": f"ref crash (train): {str(e)[:200]}",
                    "time_ms": (time.time() - t0) * 1000}

    ref_nan = _has_nan_inf(ref_out)

    tmp_dir = tempfile.mkdtemp(prefix="train_stress_")

    try:
        orig_mod = _load_module_from_source(
            cfg["kernel_code"], f"ext_orig_{orig_hash}", tmp_dir)
    except Exception as e:
        return {"ref_ok": not ref_nan, "original_ok": False, "mutant_ok": False,
                "error": f"orig compile: {str(e)[:200]}",
                "time_ms": (time.time() - t0) * 1000}

    orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")
    orig_model = _instantiate_model(orig_cls, init_args, init_rng)
    orig_model = orig_model.to(device).train()
    _sync_weights(ref_model, orig_model)

    original_ok = False
    orig_out = None
    with torch.no_grad():
        try:
            orig_out = _call_isolated(orig_model, stress_on_device)
            original_ok = _allclose(ref_out, orig_out, atol, rtol)
        except Exception:
            original_ok = False

    if ref_nan and not original_ok:
        return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                "ref_nan_fallback": True,
                "error": "ref NaN/Inf and original also invalid (train)",
                "time_ms": (time.time() - t0) * 1000}

    compare_target = ref_out

    if is_same_code:
        mutant_ok = original_ok
        mut_error = ""
    else:
        try:
            mut_mod = _load_module_from_source(
                cfg["mutated_code"], f"ext_mut_{mut_hash}", tmp_dir)
        except Exception as e:
            return {"ref_ok": True, "original_ok": original_ok, "mutant_ok": False,
                    "ref_nan_fallback": ref_nan,
                    "error": f"mut compile: {str(e)[:200]}",
                    "time_ms": (time.time() - t0) * 1000}

        mut_cls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
        mut_model = _instantiate_model(mut_cls, init_args, init_rng)
        mut_model = mut_model.to(device).train()
        _sync_weights(ref_model, mut_model)

        mutant_ok = False
        mut_error = ""
        with torch.no_grad():
            try:
                mut_out = _call_isolated(mut_model, stress_on_device)
                mutant_ok = _allclose(compare_target, mut_out, atol, rtol)
            except Exception as e:
                mutant_ok = False
                mut_error = f"mutant crash: {str(e)[:200]}"

    diff_summary = ""
    if not original_ok and orig_out is not None and not ref_nan:
        try:
            diff_summary = _compute_diff_summary(ref_out, orig_out)
        except Exception:
            pass

    return {
        "ref_ok": True,
        "ref_nan_fallback": ref_nan,
        "original_ok": original_ok,
        "mutant_ok": mutant_ok,
        "diff_summary": diff_summary,
        "error": mut_error,
        "time_ms": (time.time() - t0) * 1000,
    }


@_sound_mode_result
def run_dtype_stress(cfg):
    """Configuration Augmentation: re-run with float16 / bfloat16 inputs.

    For each target dtype, cast inputs and all three models, then compare.
    Returns killed=True if mutant diverges from ref under any dtype.
    """
    import torch

    t0 = time.time()
    device = cfg["device"]
    atol = cfg.get("atol", 1e-2)
    rtol = cfg.get("rtol", 1e-2)
    seed = cfg.get("seed", 50000)
    target_dtypes = cfg.get("target_dtypes", ["float16", "bfloat16"])

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}

    try:
        ref_model, orig_model, mut_model, get_inputs, _, _ = _build_models(
            cfg, f"dtype_{seed}", device)
    except Exception as e:
        return {"killed": False, "error": f"build: {str(e)[:200]}",
                "time_ms": (time.time() - t0) * 1000}

    torch.manual_seed(seed)
    base_inputs = get_inputs()
    results_per_dtype = {}

    for dname in target_dtypes:
        dt = dtype_map.get(dname)
        if dt is None:
            results_per_dtype[dname] = {"error": f"unsupported dtype: {dname}"}
            continue

        try:
            cast_inputs = [
                x.to(dtype=dt, device=device) if (isinstance(x, torch.Tensor) and x.dtype.is_floating_point) else
                (x.to(device) if isinstance(x, torch.Tensor) else x)
                for x in base_inputs
            ]

            ref_m = ref_model.to(dt)
            orig_m = orig_model.to(dt)
            mut_m = mut_model.to(dt)

            with torch.no_grad():
                try:
                    ref_out = _call_isolated(ref_m, cast_inputs)
                except Exception as e:
                    results_per_dtype[dname] = {
                        "ref_ok": False,
                        "error": f"reference crash: {str(e)[:300]}",
                    }
                    continue
                ref_nan = _has_nan_inf(ref_out)

                try:
                    orig_out = _call_isolated(orig_m, cast_inputs)
                except Exception as e:
                    results_per_dtype[dname] = {
                        "ref_ok": True,
                        "orig_ok": False,
                        "error": f"original crash: {str(e)[:300]}",
                    }
                    continue
                orig_ok = _allclose(ref_out, orig_out, atol, rtol)
                if not orig_ok:
                    results_per_dtype[dname] = {
                        "ref_ok": True,
                        "orig_ok": False,
                        "ref_nan_fallback": ref_nan,
                        "reason": "original diverges from reference",
                    }
                    continue

                try:
                    mut_out = _call_isolated(mut_m, cast_inputs)
                except Exception as e:
                    results_per_dtype[dname] = {
                        "ref_ok": True,
                        "orig_ok": True,
                        "mut_ok": False,
                        "error": f"candidate crash: {str(e)[:300]}",
                    }
                    return {
                        "killed": True,
                        "killing_dtype": dname,
                        "kill_type": "candidate_crash",
                        "results_per_dtype": results_per_dtype,
                        "error": f"candidate crash: {str(e)[:300]}",
                        "time_ms": (time.time() - t0) * 1000,
                    }
                mut_ok = _allclose(ref_out, mut_out, atol, rtol)

            results_per_dtype[dname] = {
                "orig_ok": orig_ok, "mut_ok": mut_ok,
                "ref_nan_fallback": ref_nan,
            }

            if orig_ok and not mut_ok:
                return {
                    "killed": True,
                    "killing_dtype": dname,
                    "results_per_dtype": results_per_dtype,
                    "time_ms": (time.time() - t0) * 1000,
                }

        except RuntimeError as e:
            err_msg = str(e)[:300]
            is_unsupported = any(kw in err_msg.lower() for kw in [
                "not implemented for 'half'", "not implemented for 'bfloat16'",
                "expected scalar type", "dtype", "doesn't match",
            ])
            results_per_dtype[dname] = {
                "error": err_msg,
                "dtype_unsupported": is_unsupported,
            }
            continue
        except Exception as e:
            results_per_dtype[dname] = {"error": str(e)[:200]}
            continue

    return {
        "killed": False,
        "killing_dtype": None,
        "results_per_dtype": results_per_dtype,
        "time_ms": (time.time() - t0) * 1000,
    }


@_sound_mode_result
def run_repeated(cfg):
    """Execution Augmentation: run the same input N times, any-divergence detection.

    Detects non-deterministic mutants (Type 5) by checking:
    1. Any trial where mutant != ref  → killed
    2. Mutant outputs inconsistent across trials → killed (self-inconsistency)
    """
    import torch

    t0 = time.time()
    device = cfg["device"]
    atol = cfg.get("atol", 1e-2)
    rtol = cfg.get("rtol", 1e-2)
    seed = cfg.get("seed", 60000)
    n_trials = cfg.get("n_trials", 10)

    try:
        ref_model, orig_model, mut_model, get_inputs, _, _ = _build_models(
            cfg, f"rep_{seed}", device)
    except Exception as e:
        return {"killed": False, "error": f"build: {str(e)[:200]}",
                "time_ms": (time.time() - t0) * 1000}

    torch.manual_seed(seed)
    base_inputs = get_inputs()
    inputs_on_device = [
        x.to(device) if isinstance(x, torch.Tensor) else x
        for x in base_inputs
    ]

    with torch.no_grad():
        try:
            ref_out = _call_isolated(ref_model, inputs_on_device)
        except Exception as e:
            return {
                "killed": False,
                "ref_ok": False,
                "original_ok": False,
                "mutant_ok": False,
                "error": f"ref crash: {str(e)[:200]}",
                "time_ms": (time.time() - t0) * 1000,
            }
        try:
            orig_out = _call_isolated(orig_model, inputs_on_device)
        except Exception as e:
            return {
                "killed": False,
                "ref_ok": True,
                "original_ok": False,
                "mutant_ok": False,
                "error": f"original crash: {str(e)[:200]}",
                "time_ms": (time.time() - t0) * 1000,
            }
        if not _allclose(ref_out, orig_out, atol, rtol):
            return {
                "killed": False,
                "ref_ok": True,
                "original_ok": False,
                "mutant_ok": False,
                "error": "original diverges from reference",
                "time_ms": (time.time() - t0) * 1000,
            }

    mut_outputs = []
    divergent_trial = None
    self_inconsistent = False

    for trial_i in range(n_trials):
        with torch.no_grad():
            try:
                mut_out = _call_isolated(mut_model, inputs_on_device)
            except Exception as e:
                return {
                    "killed": True,
                    "ref_ok": True,
                    "original_ok": True,
                    "mutant_ok": False,
                    "divergent_trial": trial_i,
                    "self_inconsistent": False,
                    "reason": "mutant_crash",
                    "error": f"candidate crash: {str(e)[:200]}",
                    "time_ms": (time.time() - t0) * 1000,
                }

        if not _allclose(ref_out, mut_out, atol, rtol):
            if divergent_trial is None:
                divergent_trial = trial_i

        if isinstance(mut_out, torch.Tensor):
            mut_outputs.append(mut_out.detach().cpu().clone())

    if divergent_trial is not None:
        return {
            "killed": True,
            "divergent_trial": divergent_trial,
            "self_inconsistent": False,
            "time_ms": (time.time() - t0) * 1000,
        }

    if len(mut_outputs) >= 2:
        first = mut_outputs[0]
        for i, out in enumerate(mut_outputs[1:], start=1):
            if not _allclose(first, out, atol=1e-6, rtol=1e-6):
                self_inconsistent = True
                divergent_trial = i
                break

    if self_inconsistent:
        return {
            "killed": True,
            "divergent_trial": divergent_trial,
            "self_inconsistent": True,
            "time_ms": (time.time() - t0) * 1000,
        }

    return {
        "killed": False,
        "divergent_trial": None,
        "self_inconsistent": False,
        "time_ms": (time.time() - t0) * 1000,
    }


@_sound_mode_result
def run_llm_verify(cfg):
    """Verify an LLM-suggested input against original and mutant kernels.

    cfg must contain:
      - problem_file, kernel_code, mutated_code, device, atol, rtol
      - test_inputs_code: Python source defining generate_inputs(device)
    """
    import torch
    import math
    from src.mutengine.mutant_runner import _load_module_from_source
    from src.bridge.eval_bridge import _load_module_from_path

    t0 = time.time()
    device = cfg["device"]
    atol = cfg["atol"]
    rtol = cfg["rtol"]
    test_code = cfg["test_inputs_code"]

    # --- safely execute LLM-suggested input generator ---
    namespace = {"torch": torch, "math": math}
    try:
        exec(test_code, namespace)
    except Exception as e:
        return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                "error": f"llm_code_exec: {str(e)[:300]}",
                "time_ms": (time.time() - t0) * 1000}

    gen_fn = namespace.get("generate_inputs")
    if gen_fn is None:
        return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                "error": "llm code missing generate_inputs function",
                "time_ms": (time.time() - t0) * 1000}

    try:
        llm_inputs = gen_fn(device)
        if not isinstance(llm_inputs, (list, tuple)):
            llm_inputs = [llm_inputs]
        llm_inputs = list(llm_inputs)
    except Exception as e:
        return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                "error": f"llm_generate: {str(e)[:300]}",
                "time_ms": (time.time() - t0) * 1000}

    orig_hash = _code_hash(cfg["kernel_code"])
    mut_hash = _code_hash(cfg["mutated_code"])
    is_same_code = (cfg["kernel_code"] == cfg["mutated_code"])

    ref_mod = _load_module_from_path(cfg["problem_file"], "llm_ref")
    get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
    init_args = get_init_inputs()
    ref_cls = ref_mod.Model
    init_rng = _capture_init_rng()
    ref_model = _instantiate_model(ref_cls, init_args)
    ref_model = ref_model.to(device).eval()

    try:
        ref_get_inputs = getattr(ref_mod, "get_inputs", None)
        if ref_get_inputs:
            ref_inputs = ref_get_inputs()
            if isinstance(ref_inputs, (list, tuple)):
                expected_n = len(ref_inputs)
                if len(llm_inputs) != expected_n and expected_n > 0:
                    if len(llm_inputs) > expected_n and len(llm_inputs) % expected_n == 0:
                        llm_inputs = llm_inputs[:expected_n]
                    elif len(llm_inputs) > expected_n:
                        llm_inputs = llm_inputs[:expected_n]
    except Exception:
        pass

    with torch.no_grad():
        try:
            ref_out = _call_isolated(ref_model, llm_inputs)
        except Exception as e:
            return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                    "error": f"ref crash: {str(e)[:200]}",
                    "time_ms": (time.time() - t0) * 1000}

    ref_nan = _has_nan_inf(ref_out)

    tmp_dir = tempfile.mkdtemp(prefix="llm_verify_")

    try:
        orig_mod = _load_module_from_source(
            cfg["kernel_code"], f"ext_orig_{orig_hash}", tmp_dir)
    except Exception as e:
        return {"ref_ok": not ref_nan, "original_ok": False, "mutant_ok": False,
                "error": f"orig compile: {str(e)[:200]}",
                "time_ms": (time.time() - t0) * 1000}

    orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")
    orig_model = _instantiate_model(orig_cls, init_args, init_rng)
    orig_model = orig_model.to(device).eval()
    _sync_weights(ref_model, orig_model)

    original_ok = False
    orig_out = None
    orig_error = ""
    with torch.no_grad():
        try:
            orig_out = _call_isolated(orig_model, llm_inputs)
            original_ok = _allclose(ref_out, orig_out, atol, rtol)
        except Exception as e:
            original_ok = False
            orig_error = f"original crash: {str(e)[:200]}"

    if ref_nan and not original_ok:
        return {"ref_ok": False, "original_ok": False, "mutant_ok": False,
                "killed": False, "ref_nan_fallback": True,
                "diff_summary": "", "mut_error": "",
                "error": "ref NaN/Inf and original also invalid on LLM input",
                "time_ms": (time.time() - t0) * 1000}

    compare_target = ref_out

    if is_same_code:
        mutant_ok = original_ok
        mut_out = orig_out
        mut_error = ""
    else:
        try:
            mut_mod = _load_module_from_source(
                cfg["mutated_code"], f"ext_mut_{mut_hash}", tmp_dir)
        except Exception as e:
            return {"ref_ok": True, "original_ok": original_ok, "mutant_ok": False,
                    "killed": original_ok, "ref_nan_fallback": ref_nan,
                    "diff_summary": "", "mut_error": f"mut compile: {str(e)[:200]}",
                    "error": f"mut compile: {str(e)[:200]}",
                    "time_ms": (time.time() - t0) * 1000}

        mut_cls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
        mut_model = _instantiate_model(mut_cls, init_args, init_rng)
        mut_model = mut_model.to(device).eval()
        _sync_weights(ref_model, mut_model)

        mutant_ok = False
        mut_out = None
        mut_error = ""
        with torch.no_grad():
            try:
                mut_out = _call_isolated(mut_model, llm_inputs)
                mutant_ok = _allclose(compare_target, mut_out, atol, rtol)
            except Exception as e:
                mutant_ok = False
                mut_error = str(e)[:200]

    diff_summary = ""
    if mut_out is not None and not mutant_ok:
        try:
            if isinstance(compare_target, torch.Tensor) and isinstance(mut_out, torch.Tensor):
                diff = (compare_target.float() - mut_out.float()).abs()
                ct_label = "orig" if ref_nan else "ref"
                diff_summary = (
                    f"max_diff={diff.max().item():.6e}, "
                    f"mean_diff={diff.mean().item():.6e}, "
                    f"{ct_label}_range=[{compare_target.min().item():.4e},"
                    f"{compare_target.max().item():.4e}], "
                    f"mut_range=[{mut_out.min().item():.4e},{mut_out.max().item():.4e}]"
                )
        except Exception:
            pass

    killed = original_ok and not mutant_ok

    return {
        "ref_ok": True,
        "ref_nan_fallback": ref_nan,
        "original_ok": original_ok,
        "mutant_ok": mutant_ok,
        "killed": killed,
        "diff_summary": diff_summary,
        "mut_error": mut_error,
        "error": orig_error or mut_error,
        "time_ms": (time.time() - t0) * 1000,
    }


def _rebatch_inputs(inputs, target_batch_size):
    """Resize the batch dimension (dim 0) of each tensor in inputs."""
    import torch
    rebatched = []
    for x in inputs:
        if not isinstance(x, torch.Tensor) or x.dim() == 0:
            rebatched.append(x)
            continue
        orig_bs = x.shape[0]
        if orig_bs == target_batch_size:
            rebatched.append(x)
            continue
        if target_batch_size < orig_bs:
            rebatched.append(x[:target_batch_size].contiguous())
        else:
            repeats = (target_batch_size + orig_bs - 1) // orig_bs
            expanded = x.repeat(repeats, *([1] * (x.dim() - 1)))
            rebatched.append(expanded[:target_batch_size].contiguous())
    return rebatched


def _bitwise_eq(a, b):
    """NaN-aware bitwise equality check (same as _mutant_worker._bitwise_identical)."""
    import torch
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.is_floating_point():
            nan_a = torch.isnan(a)
            nan_b = torch.isnan(b)
            if not torch.equal(nan_a, nan_b):
                return False
            mask = ~nan_a
            if mask.any():
                a_bytes = a[mask].contiguous().view(torch.uint8)
                b_bytes = b[mask].contiguous().view(torch.uint8)
                if not torch.equal(a_bytes, b_bytes):
                    return False
            return True
        return torch.equal(a, b)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False
        return all(_bitwise_eq(x, y) for x, y in zip(a, b))
    return a == b


@_sound_mode_result
def run_config_stress(cfg):
    """Configuration-Stress Track: vary batch_size while keeping all other dims fixed.

    cfg must contain:
      - problem_file, kernel_code, mutated_code, device
      - batch_sizes: list of int (e.g. [1, 2, 4, 8, 16, 32, 64])
      - seeds: list of int (random seeds per batch_size, default [42, 123, 7777])

    Returns:
      - killed: bool
      - killing_batch_size: int or null
      - results_per_batch: {bs: {status, ...}}
    """
    import torch
    from src.mutengine.mutant_runner import _load_module_from_source
    from src.bridge.eval_bridge import _load_module_from_path

    t0 = time.time()
    device = cfg["device"]
    batch_sizes = cfg.get("batch_sizes", [1, 2, 4, 8, 16, 32, 64])
    seeds = cfg.get("seeds", [42, 123, 7777])

    orig_hash = _code_hash(cfg["kernel_code"])
    mut_hash = _code_hash(cfg["mutated_code"])
    ref_hash = _code_hash(cfg["problem_file"])
    is_same_code = (cfg["kernel_code"] == cfg["mutated_code"])

    ref_mod = _load_module_from_path(cfg["problem_file"], f"ext_ref_{ref_hash}")
    get_inputs = ref_mod.get_inputs
    get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
    init_args = get_init_inputs()

    ref_cls = ref_mod.Model
    init_rng = _capture_init_rng()
    ref_model = _instantiate_model(ref_cls, init_args)
    ref_model = ref_model.to(device).eval()

    tmp_dir = tempfile.mkdtemp(prefix="cfg_stress_")

    try:
        orig_mod = _load_module_from_source(
            cfg["kernel_code"], f"ext_orig_{orig_hash}", tmp_dir)
    except Exception as e:
        return {"killed": False, "error": f"orig compile: {str(e)[:200]}",
                "time_ms": (time.time() - t0) * 1000}
    orig_cls = getattr(orig_mod, "ModelNew", None) or getattr(orig_mod, "Model")
    orig_model = _instantiate_model(orig_cls, init_args, init_rng)
    orig_model = orig_model.to(device).eval()
    _sync_weights(ref_model, orig_model)

    if is_same_code:
        mut_model = orig_model
    else:
        try:
            mut_mod = _load_module_from_source(
                cfg["mutated_code"], f"ext_mut_{mut_hash}", tmp_dir)
        except Exception as e:
            return {"killed": False, "error": f"mut compile: {str(e)[:200]}",
                    "time_ms": (time.time() - t0) * 1000}
        mut_cls = getattr(mut_mod, "ModelNew", None) or getattr(mut_mod, "Model")
        mut_model = _instantiate_model(mut_cls, init_args, init_rng)
        mut_model = mut_model.to(device).eval()
        _sync_weights(ref_model, mut_model)

    results_per_batch = {}

    for bs in batch_sizes:
        bs_result = {"seeds_tested": [], "status": "passed"}
        for seed in seeds:
            torch.manual_seed(seed)
            try:
                base_inputs = get_inputs()
            except Exception as e:
                bs_result = {"status": "get_inputs_error", "error": str(e)[:200]}
                break

            try:
                rebatched = _rebatch_inputs(base_inputs, bs)
                on_device = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in rebatched
                ]
            except Exception as e:
                bs_result = {"status": "rebatch_error", "error": str(e)[:200]}
                break

            with torch.no_grad():
                try:
                    ref_out = _call_isolated(ref_model, on_device)
                except Exception as e:
                    bs_result["seeds_tested"].append(
                        {"seed": seed, "status": "ref_crash", "error": str(e)[:100]})
                    continue

                ref_nan = _has_nan_inf(ref_out)

                try:
                    orig_out = _call_isolated(orig_model, on_device)
                except Exception:
                    bs_result["seeds_tested"].append(
                        {"seed": seed, "status": "orig_crash"})
                    continue

                cfg_atol = cfg.get("atol", 1e-2)
                cfg_rtol = cfg.get("rtol", 1e-2)

                orig_ok = _allclose(ref_out, orig_out, cfg_atol, cfg_rtol)
                if not orig_ok:
                    bs_result["seeds_tested"].append(
                        {"seed": seed, "status": "orig_diverges_from_ref"})
                    continue

                try:
                    mut_out = _call_isolated(mut_model, on_device)
                except Exception as e:
                    results_per_batch[str(bs)] = {
                        "status": "killed_by_crash",
                        "seed": seed,
                        "ref_nan_fallback": ref_nan,
                        "error": str(e)[:200],
                    }
                    return {
                        "killed": True,
                        "killing_batch_size": bs,
                        "killing_seed": seed,
                        "kill_type": "mutant_crash",
                        "results_per_batch": results_per_batch,
                        "time_ms": (time.time() - t0) * 1000,
                    }

                mut_ok = _allclose(ref_out, mut_out, cfg_atol, cfg_rtol)
                if not mut_ok:
                    results_per_batch[str(bs)] = {
                        "status": "killed_by_divergence",
                        "seed": seed,
                        "ref_nan_fallback": ref_nan,
                    }
                    return {
                        "killed": True,
                        "killing_batch_size": bs,
                        "killing_seed": seed,
                        "kill_type": "output_divergence",
                        "results_per_batch": results_per_batch,
                        "time_ms": (time.time() - t0) * 1000,
                    }

                status = "passed_ref_nan_fallback" if ref_nan else "passed"
                bs_result["seeds_tested"].append(
                    {"seed": seed, "status": status})

        results_per_batch[str(bs)] = bs_result

    return {
        "killed": False,
        "killing_batch_size": None,
        "results_per_batch": results_per_batch,
        "time_ms": (time.time() - t0) * 1000,
    }


def _gpu_preflight() -> str | None:
    """Check GPU health and set memory limit. Returns error string or None if OK."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "CUDA not available"
        free, total = torch.cuda.mem_get_info()
        free_mb = free / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        if free_mb < 512:
            return f"GPU memory too low: {free_mb:.0f}/{total_mb:.0f} MB free"
        # Cap GPU memory at 90% to prevent driver-level OOM/hang.
        # This makes overuse throw a clean CUDA OOM error instead of
        # freezing the GPU driver.
        try:
            torch.cuda.set_per_process_memory_fraction(0.90)
        except Exception:
            pass
        return None
    except Exception as e:
        return f"GPU check failed: {e}"


def main():
    cfg_path, res_path = sys.argv[1], sys.argv[2]
    with open(cfg_path) as f:
        cfg = json.load(f)

    mode = cfg.get("mode", "value_stress")

    gpu_err = _gpu_preflight()
    if gpu_err:
        result = _normalize_validation_result({
            "ref_ok": False,
            "original_ok": False,
            "mutant_ok": False,
            "killed": False,
            "error": f"GPU_PREFLIGHT_FAIL: {gpu_err}",
            "validation_status": "inconclusive",
        })
        with open(res_path, "w") as f:
            json.dump(result, f)
        return

    try:
        if mode == "dtype_stress":
            result = run_dtype_stress(cfg)
        elif mode == "repeated_run":
            result = run_repeated(cfg)
        elif mode == "training_stress":
            result = run_training_stress(cfg)
        elif mode == "llm_verify":
            result = run_llm_verify(cfg)
        elif mode == "config_stress":
            result = run_config_stress(cfg)
        else:
            result = run_stress(cfg)
    except Exception as e:
        result = _normalize_validation_result({
            "ref_ok": False,
            "original_ok": False,
            "mutant_ok": False,
            "killed": False,
            "error": f"WorkerCrash: {str(e)[:300]}",
            "validation_status": "inconclusive",
        })

    with open(res_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
