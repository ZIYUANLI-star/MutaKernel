#!/usr/bin/env python3
"""Execute one direct candidate-versus-reference FSE test case.

This worker is intentionally independent of mutation generation and EMD.  It
loads a frozen reference task and one candidate in an isolated subprocess,
constructs both from replayed RNG state and identical init arguments, prepares
one explicitly scoped test input, and delegates the verdict to
``src.validation``.

Usage: ``python scripts/_candidate_worker.py CONFIG.json RESULT.json``
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stress.policy_bank import STRESS_POLICIES
from src.experiments.contract import (
    assert_case_in_contract,
    validate_call_inputs,
    validate_case_parameters,
    validate_contract,
)
from src.validation import (
    ExecutionConfig,
    OracleConfig,
    RNGSnapshot,
    Tolerance,
    ValidationStatus,
    clone_tree,
    describe_input_tree,
    validate_pair,
)


WORKER_SCHEMA_VERSION = "1.0"
ALLOWED_SCOPES = {"in_contract", "extended_contract"}
DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
ORACLE_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    )
}
RUNTIME_ADAPTER_FIELDS = frozenset(
    {
        "policy_arg_indices",
        "batch_arg_indices",
        "batch_dimension",
        "layout_arg_indices",
        "dtype_arg_indices",
    }
)


def _validate_materialized_parameters(
    raw: Any,
    mode: str,
) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("case parameters must be an object")
    semantic_fields = {
        "dtype",
        "requires_backward",
        "batch_size",
        "layout",
        "repeat_count",
    }
    unknown = set(raw) - semantic_fields - RUNTIME_ADAPTER_FIELDS
    if unknown:
        raise ValueError(f"case parameters contain unknown fields: {sorted(unknown)}")
    semantic = validate_case_parameters(
        {key: raw[key] for key in semantic_fields if key in raw},
        mode,
    )
    normalized = dict(semantic)
    for name in (
        "policy_arg_indices",
        "batch_arg_indices",
        "layout_arg_indices",
        "dtype_arg_indices",
    ):
        if name not in raw:
            continue
        indices = raw[name]
        if not isinstance(indices, list) or not indices or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        ):
            raise ValueError(f"{name} must be a non-empty list of non-negative integers")
        if len(indices) != len(set(indices)):
            raise ValueError(f"{name} contains duplicate indices")
        normalized[name] = list(indices)
    if "batch_dimension" in raw:
        dimension = raw["batch_dimension"]
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise ValueError("batch_dimension must be a non-negative integer")
        normalized["batch_dimension"] = dimension
    return normalized


def _short_error(exc: BaseException, limit: int = 500) -> Dict[str, str]:
    return {
        "exception_type": type(exc).__name__,
        "message": str(exc).replace("\r", " ").replace("\n", " ")[:limit],
    }


def _result(
    status: ValidationStatus,
    reason: str,
    *,
    phase: str,
    scope: str,
    errors: Optional[Sequence[Mapping[str, Any]]] = None,
    timings_ms: Optional[Mapping[str, Any]] = None,
    candidate_runs: int = 0,
    reference_runs: int = 0,
    verdict: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": WORKER_SCHEMA_VERSION,
        "validation_status": status.value,
        "reason": reason,
        "phase": phase,
        "scope": scope,
        "errors": list(errors or ()),
        "timings_ms": dict(timings_ms or {}),
        "candidate_runs": candidate_runs,
        "reference_runs": reference_runs,
        # Legacy-friendly fields.  They are observations, not a binary ground
        # truth label; callers must use validation_status for aggregation.
        "ref_ok": reference_runs > 0 and status is not ValidationStatus.INCONCLUSIVE,
        "candidate_ok": status is ValidationStatus.PASS,
        "killed": status is ValidationStatus.FAIL,
    }
    if verdict is not None:
        payload["verdict"] = dict(verdict)
    return payload


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_class(module: Any, names: Sequence[str]) -> type:
    for name in names:
        model_class = getattr(module, name, None)
        if isinstance(model_class, type):
            return model_class
    raise AttributeError(f"module has none of the model classes {list(names)!r}")


def _construct(model_class: type, init_inputs: Any) -> Any:
    if isinstance(init_inputs, (tuple, list)):
        return model_class(*init_inputs)
    # KernelBench convention is list/tuple unpacking.  For a non-sequence
    # return, preserve the value as one positional constructor argument rather
    # than silently dropping it and constructing a different model.
    return model_class(init_inputs)


def _map_tree(
    value: Any,
    transform: Any,
    memo: Optional[Dict[int, Any]] = None,
) -> Any:
    """Map tensor leaves while preserving common Python containers."""

    if memo is None:
        memo = {}
    object_id = id(value)
    if object_id in memo:
        return memo[object_id]
    if isinstance(value, torch.Tensor):
        transformed = transform(value)
        memo[object_id] = transformed
        return transformed
    if isinstance(value, list):
        transformed_list = []
        memo[object_id] = transformed_list
        transformed_list.extend(_map_tree(item, transform, memo) for item in value)
        return transformed_list
    if isinstance(value, tuple):
        values = tuple(_map_tree(item, transform, memo) for item in value)
        transformed_tuple = type(value)(*values) if hasattr(value, "_fields") else values
        memo[object_id] = transformed_tuple
        return transformed_tuple
    if isinstance(value, Mapping):
        transformed_mapping = type(value)(
            (key, _map_tree(item, transform, memo)) for key, item in value.items()
        )
        memo[object_id] = transformed_mapping
        return transformed_mapping
    return value


def _cast_floating_tree(value: Any, dtype: torch.dtype) -> Any:
    return _map_tree(
        value,
        lambda tensor: tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor,
    )


def _move_tree(value: Any, device: str) -> Any:
    return _map_tree(value, lambda tensor: tensor.to(device))


def _noncontiguous_same_shape(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a tensor into a same-shape view with a non-unit final stride."""

    if tensor.dim() == 0:
        raise ValueError("cannot create a non-contiguous scalar input")
    if tensor.shape[-1] == 0:
        return tensor.detach().clone(memory_format=torch.preserve_format)
    storage_shape = list(tensor.shape)
    storage_shape[-1] *= 2
    backing = torch.empty(storage_shape, dtype=tensor.dtype, device=tensor.device)
    view = backing[..., ::2]
    view.copy_(tensor)
    view.requires_grad_(tensor.requires_grad)
    if view.is_contiguous():
        raise RuntimeError("failed to construct a non-contiguous input view")
    return view


def _detach_tree(value: Any) -> Any:
    return _map_tree(value, lambda tensor: tensor.detach().clone())


def _differentiable_tensors(value: Any) -> list[torch.Tensor]:
    leaves: list[torch.Tensor] = []
    seen: set[int] = set()

    def collect(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            if (
                id(item) not in seen
                and item.requires_grad
                and (item.is_floating_point() or item.is_complex())
            ):
                seen.add(id(item))
                leaves.append(item)
        elif isinstance(item, Mapping):
            for child in item.values():
                collect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)

    collect(value)
    return leaves


def _upstream_gradients(
    outputs: Sequence[torch.Tensor],
    *,
    vjp_index: int,
    seed: int,
) -> list[torch.Tensor]:
    if vjp_index == 0:
        return [torch.ones_like(output) for output in outputs]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 104729 * vjp_index)
    gradients = []
    for output in outputs:
        generation_dtype = torch.float64 if output.dtype in {
            torch.float64,
            torch.complex128,
        } else torch.float32
        real = torch.randn(output.shape, generator=generator, dtype=generation_dtype)
        if output.is_complex():
            imaginary = torch.randn(
                output.shape,
                generator=generator,
                dtype=generation_dtype,
            )
            value = torch.complex(real, imaginary)
        else:
            value = real
        gradients.append(value.to(device=output.device, dtype=output.dtype))
    return gradients


class _ForwardBackwardModule(nn.Module):
    """Expose forward output plus several deterministic VJPs to the oracle."""

    def __init__(self, model: nn.Module, *, seed: int, vjp_count: int) -> None:
        super().__init__()
        self.model = model
        self.seed = seed
        self.vjp_count = vjp_count

    def forward(self, *args: Any) -> Mapping[str, Any]:
        prepared = _map_tree(
            args,
            lambda tensor: tensor.detach().clone().requires_grad_(
                tensor.is_floating_point() or tensor.is_complex()
            ),
        )
        output = self.model(*prepared)
        differentiable_outputs = _differentiable_tensors(output)
        if not differentiable_outputs:
            raise RuntimeError("forward output has no differentiable floating/complex leaf")
        input_targets = _differentiable_tensors(prepared)
        parameters = list(self.model.named_parameters())
        targets = input_targets + [parameter for _, parameter in parameters]
        vjp_gradients = []
        for vjp_index in range(self.vjp_count):
            gradients = torch.autograd.grad(
                differentiable_outputs,
                targets,
                grad_outputs=_upstream_gradients(
                    differentiable_outputs,
                    vjp_index=vjp_index,
                    seed=self.seed,
                ),
                retain_graph=vjp_index + 1 < self.vjp_count,
                allow_unused=True,
            )
            input_values = gradients[: len(input_targets)]
            parameter_values = gradients[len(input_targets) :]
            input_by_id = {
                id(tensor): gradient
                for tensor, gradient in zip(input_targets, input_values)
            }

            def input_gradient(tensor: torch.Tensor) -> Any:
                if not (tensor.is_floating_point() or tensor.is_complex()):
                    return None
                gradient = input_by_id.get(id(tensor))
                return None if gradient is None else gradient.detach().clone()

            vjp_gradients.append(
                {
                    "vjp_index": vjp_index,
                    "input_gradients": _map_tree(prepared, input_gradient),
                    "parameter_gradients": {
                        name: None if gradient is None else gradient.detach().clone()
                        for (name, _), gradient in zip(parameters, parameter_values)
                    },
                }
            )
        return {
            "output": _detach_tree(output),
            "vjp_gradients": vjp_gradients,
        }


def _rebatch_tensor(tensor: torch.Tensor, *, size: int, dimension: int) -> torch.Tensor:
    if tensor.dim() <= dimension:
        raise ValueError(
            f"cannot rebatch shape {tuple(tensor.shape)} along dimension {dimension}"
        )
    original = tensor.shape[dimension]
    if original == size:
        return tensor.clone(memory_format=torch.preserve_format)
    if original <= 0:
        raise ValueError("cannot expand an empty batch dimension")
    if size < original:
        slices = [slice(None)] * tensor.dim()
        slices[dimension] = slice(0, size)
        return tensor[tuple(slices)].contiguous()
    repeats = [1] * tensor.dim()
    repeats[dimension] = (size + original - 1) // original
    expanded = tensor.repeat(*repeats)
    slices = [slice(None)] * tensor.dim()
    slices[dimension] = slice(0, size)
    return expanded[tuple(slices)].contiguous()


def _normalize_args(raw: Any) -> Tuple[Any, ...]:
    if isinstance(raw, (tuple, list)):
        return tuple(raw)
    return (raw,)


def _prepare_inputs(
    reference_module: Any,
    case: Mapping[str, Any],
    contract: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, ...]:
    get_inputs = getattr(reference_module, "get_inputs", None)
    if not callable(get_inputs):
        raise AttributeError("reference module has no callable get_inputs")
    seed = case["seed"]
    _seed_all(seed)
    args = _normalize_args(get_inputs())
    if contract is not None:
        validate_call_inputs(args, contract)

    policy = case["policy"]
    if policy not in {"iid", "identity"}:
        policy_function = STRESS_POLICIES.get(policy)
        if policy_function is None:
            raise ValueError(f"unknown input policy: {policy}")
        selected = case.get("parameters", {}).get("policy_arg_indices")
        if not isinstance(selected, list) or not selected:
            raise ValueError("stress policy requires explicit policy_arg_indices")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(args)
            for index in selected
        ):
            raise ValueError("policy_arg_indices contains an invalid argument index")
        selected_values = [args[index] for index in selected]
        stressed_values = policy_function(selected_values, seed)
        mutable_args = list(args)
        for index, stressed in zip(selected, stressed_values):
            mutable_args[index] = stressed
        args = tuple(mutable_args)

    parameters = case.get("parameters", {})
    target_dtype = parameters.get("dtype")
    if target_dtype is not None:
        if target_dtype not in DTYPES:
            raise ValueError(f"unsupported dtype parameter: {target_dtype}")
        selected = parameters.get("dtype_arg_indices")
        if not isinstance(selected, list) or not selected:
            raise ValueError("dtype case requires explicit dtype_arg_indices")
        mutable_args = list(args)
        for index in selected:
            if isinstance(index, bool) or not isinstance(index, int) or not (0 <= index < len(args)):
                raise ValueError("dtype_arg_indices contains an invalid argument index")
            mutable_args[index] = _cast_floating_tree(
                mutable_args[index], DTYPES[target_dtype]
            )
        args = tuple(mutable_args)

    if "batch_size" in parameters:
        selected = parameters.get("batch_arg_indices")
        if not isinstance(selected, list) or not selected:
            raise ValueError(
                "batch_size requires an explicit non-empty batch_arg_indices list"
            )
        size = parameters["batch_size"]
        dimension = parameters.get("batch_dimension", 0)
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
            raise ValueError("batch_dimension must be a non-negative integer")
        mutable = list(args)
        for index in selected:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("batch_arg_indices must contain integers")
            if index < 0 or index >= len(mutable):
                raise ValueError(f"batch argument index out of range: {index}")
            if not isinstance(mutable[index], torch.Tensor):
                raise ValueError(f"batch argument {index} is not a tensor")
            mutable[index] = _rebatch_tensor(
                mutable[index], size=size, dimension=dimension
            )
        args = tuple(mutable)

    if parameters.get("layout") == "noncontiguous":
        selected = parameters.get("layout_arg_indices")
        if not isinstance(selected, list) or not selected:
            raise ValueError(
                "noncontiguous layout requires a non-empty layout_arg_indices list"
            )
        mutable = list(args)
        for index in selected:
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("layout_arg_indices must contain integers")
            if index < 0 or index >= len(mutable):
                raise ValueError(f"layout argument index out of range: {index}")
            if not isinstance(mutable[index], torch.Tensor):
                raise ValueError(f"layout argument {index} is not a tensor")
            mutable[index] = _noncontiguous_same_shape(mutable[index])
        args = tuple(mutable)

    prepared = tuple(_move_tree(args, case["device"]))
    if contract is not None:
        validate_call_inputs(prepared, contract)
    return prepared


def _instantiate_models(
    reference_module: Any,
    candidate_module: Any,
    case: Mapping[str, Any],
) -> Tuple[Any, Any]:
    get_init_inputs = getattr(reference_module, "get_init_inputs", lambda: [])
    if not callable(get_init_inputs):
        raise ModelSetupError(
            "reference_model_setup",
            TypeError("reference get_init_inputs is not callable"),
        )

    _seed_all(case["seed"])
    try:
        init_inputs = get_init_inputs()
    except BaseException as exc:
        raise ModelSetupError("reference_model_setup", exc) from exc
    constructor_rng = RNGSnapshot.capture(include_cuda=True)
    try:
        reference_class = _model_class(reference_module, ("Model",))
    except BaseException as exc:
        raise ModelSetupError("reference_model_setup", exc) from exc
    candidate_names = case.get("candidate_classes", ["ModelNew", "Model"])
    if not isinstance(candidate_names, list) or not candidate_names:
        raise ModelSetupError(
            "candidate_model_setup",
            ValueError("candidate_classes must be a non-empty list"),
        )
    try:
        candidate_class = _model_class(candidate_module, tuple(candidate_names))
    except BaseException as exc:
        raise ModelSetupError("candidate_model_setup", exc) from exc

    constructor_rng.restore()
    try:
        reference = _construct(reference_class, clone_tree(init_inputs))
    except BaseException as exc:
        raise ModelSetupError("reference_model_setup", exc) from exc
    constructor_rng.restore()
    try:
        candidate = _construct(candidate_class, clone_tree(init_inputs))
    except BaseException as exc:
        raise ModelSetupError("candidate_model_setup", exc) from exc

    dtype_name = case.get("parameters", {}).get("dtype")
    target_dtype = None if dtype_name is None else DTYPES[dtype_name]
    try:
        reference = (
            reference.to(device=case["device"])
            if target_dtype is None
            else reference.to(device=case["device"], dtype=target_dtype)
        )
    except BaseException as exc:
        raise ModelSetupError("reference_model_setup", exc) from exc
    try:
        candidate = (
            candidate.to(device=case["device"])
            if target_dtype is None
            else candidate.to(device=case["device"], dtype=target_dtype)
        )
    except BaseException as exc:
        raise ModelSetupError("candidate_model_setup", exc) from exc

    mode = case.get("mode", "eval")
    if mode in {"eval", "repeated", "config"}:
        try:
            reference.eval()
        except BaseException as exc:
            raise ModelSetupError("reference_model_setup", exc) from exc
        try:
            candidate.eval()
        except BaseException as exc:
            raise ModelSetupError("candidate_model_setup", exc) from exc
    elif mode == "train":
        try:
            reference.train()
        except BaseException as exc:
            raise ModelSetupError("reference_model_setup", exc) from exc
        try:
            candidate.train()
        except BaseException as exc:
            raise ModelSetupError("candidate_model_setup", exc) from exc
    else:
        raise ModelSetupError(
            "reference_model_setup",
            ValueError(f"unsupported execution mode: {mode}"),
        )
    return reference, candidate


class ModelSetupError(RuntimeError):
    """Attribute a construction/device/mode failure to one side of the pair."""

    def __init__(self, phase: str, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.phase = phase
        self.cause = cause


def _validate_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    required = {"subject_id", "reference_path", "candidate_path", "case", "device"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"worker config missing fields: {missing}")
    case = config["case"]
    if not isinstance(case, Mapping):
        raise ValueError("case must be an object")
    for key in ("test_id", "policy", "seed", "scope"):
        if key not in case:
            raise ValueError(f"case missing field: {key}")
    if case["scope"] not in ALLOWED_SCOPES:
        raise ValueError(f"invalid scope: {case['scope']}")
    if isinstance(case["seed"], bool) or not isinstance(case["seed"], int):
        raise ValueError("case seed must be an integer")
    parameters = case.get("parameters", {})
    mode = case.get("mode", "eval")
    parameters = _validate_materialized_parameters(parameters, mode)
    normalized = dict(case)
    normalized["parameters"] = parameters
    normalized["device"] = config["device"]
    contract = config.get("contract")
    if contract is not None:
        normalized_contract = validate_contract(contract)
        normalized["contract"] = normalized_contract
        semantic_parameters = {
            key: parameters[key]
            for key in (
                "dtype",
                "requires_backward",
                "batch_size",
                "layout",
                "repeat_count",
            )
            if key in parameters
        }
        if normalized["scope"] == "in_contract":
            assert_case_in_contract(
                {
                    "policy": normalized["policy"],
                    "mode": mode,
                    "parameters": semantic_parameters,
                },
                normalized_contract,
            )
        expected_adapters: Dict[str, Any] = {}
        if normalized["policy"] not in {"iid", "identity"}:
            expected_adapters["policy_arg_indices"] = normalized_contract[
                "policy_bindings"
            ].get(normalized["policy"])
        adapter_map = normalized_contract.get("input_adapters", {})
        for parameter_name, adapter_name, runtime_name in (
            ("dtype", "dtype", "dtype_arg_indices"),
            ("batch_size", "batch", "batch_arg_indices"),
            ("layout", "layout", "layout_arg_indices"),
        ):
            if parameter_name in semantic_parameters:
                adapter = adapter_map.get(adapter_name)
                expected_adapters[runtime_name] = (
                    None if adapter is None else adapter.get("arg_indices")
                )
        if "batch_size" in semantic_parameters:
            batch_adapter = adapter_map.get("batch")
            expected_adapters["batch_dimension"] = (
                None if batch_adapter is None else batch_adapter.get("dimension")
            )
        for runtime_name, expected in expected_adapters.items():
            if expected is None or parameters.get(runtime_name) != expected:
                raise ValueError(
                    f"materialized {runtime_name} does not match the frozen contract"
                )
    return normalized


def execute(config: Mapping[str, Any]) -> Dict[str, Any]:
    total_started = time.perf_counter_ns()
    try:
        case = _validate_config(config)
    except BaseException as exc:
        return _result(
            ValidationStatus.INCONCLUSIVE,
            "invalid worker configuration",
            phase="configuration",
            scope="in_contract",
            errors=[{"phase": "configuration", **_short_error(exc)}],
        )
    scope = case["scope"]
    timings: Dict[str, Any] = {}

    if str(config["device"]).startswith("cuda") and not torch.cuda.is_available():
        return _result(
            ValidationStatus.INCONCLUSIVE,
            "requested CUDA device is unavailable",
            phase="preflight",
            scope=scope,
        )

    reference_path = Path(config["reference_path"])
    candidate_path = Path(config["candidate_path"])
    import_caller_rng = RNGSnapshot.capture(include_cuda=True)
    started = time.perf_counter_ns()
    _seed_all(int(case["seed"]))
    try:
        reference_module = _load_module(
            reference_path, f"fse_ref_{case['test_id'][:16]}"
        )
    except BaseException as exc:
        import_caller_rng.restore()
        timings["reference_import_ms"] = (time.perf_counter_ns() - started) / 1e6
        return _result(
            ValidationStatus.INCONCLUSIVE,
            "reference import failed",
            phase="reference_import",
            scope=scope,
            errors=[{"phase": "reference_import", **_short_error(exc)}],
            timings_ms=timings,
        )
    timings["reference_import_ms"] = (time.perf_counter_ns() - started) / 1e6

    started = time.perf_counter_ns()
    _seed_all(int(case["seed"]))
    try:
        candidate_module = _load_module(
            candidate_path, f"fse_candidate_{case['test_id'][:16]}"
        )
    except BaseException as exc:
        import_caller_rng.restore()
        timings["candidate_import_ms"] = (time.perf_counter_ns() - started) / 1e6
        return _result(
            ValidationStatus.FAIL,
            "candidate import/compilation failed while reference import succeeded",
            phase="candidate_import",
            scope=scope,
            errors=[{"phase": "candidate_import", **_short_error(exc)}],
            timings_ms=timings,
        )
    timings["candidate_import_ms"] = (time.perf_counter_ns() - started) / 1e6

    caller_rng = import_caller_rng
    started = time.perf_counter_ns()
    try:
        try:
            reference, candidate = _instantiate_models(
                reference_module, candidate_module, case
            )
        except ModelSetupError as exc:
            timings["model_setup_ms"] = (time.perf_counter_ns() - started) / 1e6
            status = (
                ValidationStatus.FAIL
                if exc.phase == "candidate_model_setup"
                else ValidationStatus.INCONCLUSIVE
            )
            return _result(
                status,
                (
                    "candidate model setup failed while reference setup succeeded"
                    if status is ValidationStatus.FAIL
                    else "reference model setup failed"
                ),
                phase=exc.phase,
                scope=scope,
                errors=[{"phase": exc.phase, **_short_error(exc.cause)}],
                timings_ms=timings,
            )
        timings["model_setup_ms"] = (time.perf_counter_ns() - started) / 1e6

        started = time.perf_counter_ns()
        try:
            args = _prepare_inputs(
                reference_module,
                case,
                contract=case.get("contract"),
            )
        except Exception as exc:
            timings["input_generation_ms"] = (time.perf_counter_ns() - started) / 1e6
            return _result(
                ValidationStatus.INCONCLUSIVE,
                "input generation or policy application failed",
                phase="input_generation",
                scope=scope,
                errors=[{"phase": "input_generation", **_short_error(exc)}],
                timings_ms=timings,
            )
        timings["input_generation_ms"] = (time.perf_counter_ns() - started) / 1e6

        oracle_contract = config.get("oracle")
        if oracle_contract is None:
            oracle_contract = {
                "atol": float(config.get("atol", 1e-2)),
                "rtol": float(config.get("rtol", 1e-2)),
                "dtype_tolerances": {},
                "equal_nan": True,
                "require_dtype": True,
                "require_device": True,
                "require_layout": True,
                "require_stride": False,
                "require_aliasing": False,
            }
        if not isinstance(oracle_contract, Mapping):
            raise ValueError("oracle contract must be an object")
        atol = float(oracle_contract["atol"])
        rtol = float(oracle_contract["rtol"])
        tolerance = Tolerance(rtol=rtol, atol=atol)
        dtype_tolerances = {}
        for name, values in oracle_contract.get("dtype_tolerances", {}).items():
            if name not in ORACLE_DTYPES or not isinstance(values, Mapping):
                raise ValueError(f"invalid oracle dtype tolerance: {name!r}")
            dtype_tolerances[ORACLE_DTYPES[name]] = Tolerance(
                rtol=float(values["rtol"]),
                atol=float(values["atol"]),
            )
        for dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ):
            dtype_tolerances.setdefault(dtype, tolerance)
        oracle = OracleConfig(
            default_tolerance=tolerance,
            dtype_tolerances=dtype_tolerances,
            equal_nan=bool(oracle_contract.get("equal_nan", True)),
            require_dtype=bool(oracle_contract.get("require_dtype", True)),
            require_device=bool(oracle_contract.get("require_device", True)),
            require_layout=bool(oracle_contract.get("require_layout", True)),
            require_stride=bool(oracle_contract.get("require_stride", False)),
            require_aliasing=bool(oracle_contract.get("require_aliasing", False)),
        )
        parameters = case.get("parameters", {})
        requires_backward = bool(parameters.get("requires_backward", False))
        if requires_backward:
            vjp_count = int(
                case.get("contract", {})
                .get("execution", {})
                .get("backward_vjp_count", 3)
            )
            reference = _ForwardBackwardModule(
                reference,
                seed=int(case["seed"]),
                vjp_count=vjp_count,
            )
            candidate = _ForwardBackwardModule(
                candidate,
                seed=int(case["seed"]),
                vjp_count=vjp_count,
            )
        repeat_count = int(parameters.get("repeat_count", 1))
        execution_contract = case.get("contract", {}).get("execution", {})
        compare_input_side_effects = bool(
            execution_contract.get("compare_input_side_effects", True)
        )
        compare_module_state = bool(
            execution_contract.get("compare_module_state", True)
        )
        execution_context = torch.enable_grad() if requires_backward else torch.no_grad()
        verdicts = []
        with execution_context:
            for attempt in range(repeat_count):
                verdict = validate_pair(
                    reference,
                    candidate,
                    args=args,
                    oracle_config=oracle,
                    execution_config=ExecutionConfig(
                        # A repeated case is a true call sequence on the same
                        # instances.  Synchronize only the initial state; each
                        # later call starts from the two implementations' own
                        # post-state, which was compared by the preceding call.
                        synchronize_state=attempt == 0,
                        preserve_module_state=repeat_count == 1,
                        preserve_caller_rng=repeat_count == 1,
                        retain_outputs=False,
                        compare_input_side_effects=compare_input_side_effects,
                        compare_module_state=compare_module_state,
                    ),
                )
                verdicts.append(verdict)
                if verdict.status is not ValidationStatus.PASS:
                    break

        final_status = verdicts[-1].status
        final_reason = verdicts[-1].reason

        timings["validation_attempts"] = [
            verdict.timings.to_dict() for verdict in verdicts
        ]
        timings["total_ms"] = (time.perf_counter_ns() - total_started) / 1e6
        if repeat_count == 1:
            verdict_payload: Mapping[str, Any] = verdicts[0].to_dict()
        else:
            verdict_payload = {
                "attempts": [verdict.to_dict() for verdict in verdicts],
                "sequence_semantics": "same_instances_without_state_reset",
            }
        result = _result(
            final_status,
            final_reason,
            phase="validation",
            scope=scope,
            errors=[
                {
                    "phase": error.phase,
                    "exception_type": error.exception_type,
                    "message": error.message,
                }
                for verdict in verdicts
                for error in verdict.errors
            ],
            timings_ms=timings,
            candidate_runs=sum(verdict.candidate_invocations for verdict in verdicts),
            reference_runs=sum(verdict.reference_invocations for verdict in verdicts),
            verdict=verdict_payload,
        )
        # Metadata is retained for every trial.  Exact logical-value bytes are
        # hashed for non-passing cases so a replay can verify that it rebuilt
        # the same counterexample without embedding large tensors in JSONL.
        try:
            fingerprint_started = time.perf_counter_ns()
            result["input_evidence"] = describe_input_tree(
                args,
                hash_contents=final_status is not ValidationStatus.PASS,
            )
            result["timings_ms"]["input_evidence_ms"] = (
                time.perf_counter_ns() - fingerprint_started
            ) / 1e6
        except BaseException as exc:
            result["input_evidence"] = {
                "schema_version": "1.0",
                "error": _short_error(exc),
            }
        return result
    finally:
        caller_rng.restore()


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 2:
        print("usage: _candidate_worker.py CONFIG.json RESULT.json", file=sys.stderr)
        return 2
    config_path, result_path = map(Path, arguments)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = execute(config)
    except BaseException as exc:
        result = _result(
            ValidationStatus.INCONCLUSIVE,
            "worker crashed outside the validation protocol",
            phase="worker",
            scope="in_contract",
            errors=[{"phase": "worker", **_short_error(exc)}],
        )
    result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
