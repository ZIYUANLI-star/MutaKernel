"""Versioned correctness contracts and planned-case applicability checks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict

import torch


CONTRACT_SCHEMA_VERSION = "1.0"
ALLOWED_DTYPES = {
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
}
ALLOWED_LAYOUTS = {"contiguous", "noncontiguous", "strided", "sparse_coo", "any"}
ALLOWED_MODES = {"eval", "train", "config", "repeated"}
ALLOWED_VALUE_DOMAINS = {
    "unrestricted_finite",
    "bounded",
    "positive",
    "nonnegative",
    "probability",
    "integer_index",
    "task_specific",
}
REAL_FLOAT_DTYPES = {"float16", "bfloat16", "float32", "float64"}
EXECUTABLE_DTYPE_CASES = frozenset(REAL_FLOAT_DTYPES)
EXECUTABLE_LAYOUT_CASES = frozenset({"noncontiguous"})
CASE_PARAMETER_FIELDS = frozenset(
    {"dtype", "requires_backward", "batch_size", "layout", "repeat_count"}
)

_TOP_REQUIRED = {
    "schema_version",
    "contract_id",
    "tensor_inputs",
    "execution",
    "oracle",
    "policy_bindings",
}
_TOP_OPTIONAL = {"input_adapters", "candidate_classes", "notes"}
_TENSOR_FIELDS = {
    "arg_index",
    "dtypes",
    "shape",
    "value_domain",
    "layouts",
    "requires_grad",
    "aliases",
}
_EXECUTION_FIELDS = {
    "modes",
    "backward",
    "deterministic",
    "repeat_count_max",
    "streams",
    "compare_input_side_effects",
    "compare_module_state",
    "backward_vjp_count",
}
_ORACLE_FIELDS = {
    "atol",
    "rtol",
    "equal_nan",
    "require_dtype",
    "require_device",
    "require_layout",
    "require_stride",
    "require_aliasing",
    "dtype_tolerances",
}


class ContractError(ValueError):
    """A correctness contract is incomplete or internally inconsistent."""


def validate_case_parameters(parameters: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    """Validate the exact case vocabulary implemented by schema v1 workers."""

    if not isinstance(parameters, Mapping):
        raise ContractError("case parameters must be an object")
    unknown = set(parameters) - CASE_PARAMETER_FIELDS
    if unknown:
        raise ContractError(f"case parameters contain unknown fields: {sorted(unknown)}")
    allowed_by_mode = {
        "eval": {"dtype", "layout"},
        "train": {"requires_backward"},
        "config": {"batch_size"},
        "repeated": {"repeat_count"},
    }
    if mode not in allowed_by_mode:
        raise ContractError(f"unsupported execution mode: {mode!r}")
    invalid_for_mode = set(parameters) - allowed_by_mode[mode]
    if invalid_for_mode:
        raise ContractError(
            f"case parameters {sorted(invalid_for_mode)} are invalid in mode {mode!r}"
        )

    normalized = dict(parameters)
    if "dtype" in parameters:
        dtype = parameters["dtype"]
        if dtype not in EXECUTABLE_DTYPE_CASES:
            raise ContractError(f"dtype case {dtype!r} is not executable in schema v1")
    if "layout" in parameters:
        layout = parameters["layout"]
        if layout not in EXECUTABLE_LAYOUT_CASES:
            raise ContractError(f"layout case {layout!r} is not executable in schema v1")
    if "requires_backward" in parameters and parameters["requires_backward"] is not True:
        raise ContractError("requires_backward must be true when present")
    if mode == "config":
        batch_size = parameters.get("batch_size")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ContractError("config mode requires a positive integer batch_size")
    if mode == "repeated":
        repeat_count = parameters.get("repeat_count")
        if isinstance(repeat_count, bool) or not isinstance(repeat_count, int):
            raise ContractError("repeated mode requires an integer repeat_count")
        if repeat_count < 2:
            raise ContractError("repeat_count must be at least two")
    return normalized


def _exact_fields(value: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ContractError(f"{context} has unknown fields: {sorted(unknown)}")


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{context} must be a non-empty string")
    return value


def _string_list(value: Any, context: str, allowed: set[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{context} must be a non-empty list")
    if any(not isinstance(item, str) or item not in allowed for item in value):
        raise ContractError(f"{context} contains an unsupported value")
    if len(value) != len(set(value)):
        raise ContractError(f"{context} contains duplicates")
    return list(value)


def _nonnegative_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ContractError(f"{context} must be finite and non-negative")
    return result


def _validate_dimension(value: Any, context: str) -> Any:
    if isinstance(value, bool):
        raise ContractError(f"{context} is not a valid dimension")
    if isinstance(value, int):
        if value < 0:
            raise ContractError(f"{context} must be non-negative")
        return value
    if not isinstance(value, Mapping):
        raise ContractError(f"{context} must be an integer or a dimension object")
    _exact_fields(value, {"name", "min", "max"}, context)
    name = _nonempty_string(value.get("name"), f"{context}.name")
    lower = value.get("min")
    upper = value.get("max")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in (lower, upper)):
        raise ContractError(f"{context}.min/max must be integers")
    if lower < 0 or upper < lower:
        raise ContractError(f"{context}.min/max are invalid")
    return {"name": name, "min": lower, "max": upper}


def _validate_value_domain(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{context} must be an object")
    _exact_fields(value, {"kind", "min", "max", "description"}, context)
    kind = _nonempty_string(value.get("kind"), f"{context}.kind")
    if kind not in ALLOWED_VALUE_DOMAINS:
        raise ContractError(f"{context}.kind is unsupported: {kind!r}")
    result: Dict[str, Any] = {"kind": kind}
    if kind == "bounded":
        lower = value.get("min")
        upper = value.get("max")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in (lower, upper)):
            raise ContractError(f"{context} bounded domain requires numeric min/max")
        if not math.isfinite(float(lower)) or not math.isfinite(float(upper)) or lower > upper:
            raise ContractError(f"{context} bounded min/max are invalid")
        result.update({"min": float(lower), "max": float(upper)})
    if kind == "task_specific":
        result["description"] = _nonempty_string(
            value.get("description"), f"{context}.description"
        )
    return result


def _validate_index_list(value: Any, context: str, valid_indices: set[int]) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{context} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ContractError(f"{context} must contain integer argument indices")
    if len(value) != len(set(value)):
        raise ContractError(f"{context} contains duplicates")
    missing = set(value) - valid_indices
    if missing:
        raise ContractError(f"{context} references undeclared tensor arguments: {sorted(missing)}")
    return list(value)


def _validate_tolerance_map(value: Any) -> Dict[str, Dict[str, float]]:
    if not isinstance(value, Mapping):
        raise ContractError("contract.oracle.dtype_tolerances must be an object")
    result = {}
    for dtype, tolerance in value.items():
        if dtype not in ALLOWED_DTYPES or not isinstance(tolerance, Mapping):
            raise ContractError("contract.oracle.dtype_tolerances is invalid")
        _exact_fields(tolerance, {"atol", "rtol"}, f"dtype tolerance {dtype}")
        result[dtype] = {
            "atol": _nonnegative_float(tolerance.get("atol"), f"{dtype}.atol"),
            "rtol": _nonnegative_float(tolerance.get("rtol"), f"{dtype}.rtol"),
        }
    return result


def validate_contract(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the complete contract stored in a subject manifest."""

    if not isinstance(raw, Mapping):
        raise ContractError("contract must be an object")
    missing = _TOP_REQUIRED - set(raw)
    if missing:
        raise ContractError(f"contract is missing required fields: {sorted(missing)}")
    _exact_fields(raw, _TOP_REQUIRED | _TOP_OPTIONAL, "contract")
    if raw.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ContractError("unsupported contract schema_version")

    tensor_inputs = raw.get("tensor_inputs")
    if not isinstance(tensor_inputs, list) or not tensor_inputs:
        raise ContractError("contract.tensor_inputs must be a non-empty list")
    normalized_inputs = []
    indices = []
    for position, item in enumerate(tensor_inputs):
        context = f"contract.tensor_inputs[{position}]"
        if not isinstance(item, Mapping):
            raise ContractError(f"{context} must be an object")
        _exact_fields(item, _TENSOR_FIELDS, context)
        missing_item = _TENSOR_FIELDS - set(item)
        if missing_item:
            raise ContractError(f"{context} is missing fields: {sorted(missing_item)}")
        arg_index = item.get("arg_index")
        if isinstance(arg_index, bool) or not isinstance(arg_index, int) or arg_index < 0:
            raise ContractError(f"{context}.arg_index must be non-negative")
        shape = item.get("shape")
        if not isinstance(shape, list):
            raise ContractError(f"{context}.shape must be a list")
        requires_grad = item.get("requires_grad")
        if not isinstance(requires_grad, bool):
            raise ContractError(f"{context}.requires_grad must be boolean")
        aliases = item.get("aliases")
        if not isinstance(aliases, list) or any(
            isinstance(alias, bool) or not isinstance(alias, int) or alias < 0
            for alias in aliases
        ):
            raise ContractError(f"{context}.aliases must contain non-negative indices")
        normalized_inputs.append(
            {
                "arg_index": arg_index,
                "dtypes": _string_list(item.get("dtypes"), f"{context}.dtypes", ALLOWED_DTYPES),
                "shape": [
                    _validate_dimension(dimension, f"{context}.shape[{index}]")
                    for index, dimension in enumerate(shape)
                ],
                "value_domain": _validate_value_domain(
                    item.get("value_domain"), f"{context}.value_domain"
                ),
                "layouts": _string_list(
                    item.get("layouts"), f"{context}.layouts", ALLOWED_LAYOUTS
                ),
                "requires_grad": requires_grad,
                "aliases": list(aliases),
            }
        )
        indices.append(arg_index)
    if len(indices) != len(set(indices)):
        raise ContractError("contract tensor arg_index values must be unique")
    valid_indices = set(indices)
    for item in normalized_inputs:
        undeclared_aliases = set(item["aliases"]) - valid_indices
        if undeclared_aliases:
            raise ContractError(
                f"tensor argument {item['arg_index']} aliases undeclared arguments"
            )
        if item["arg_index"] in item["aliases"]:
            raise ContractError("a tensor argument cannot alias itself in the contract")
    aliases_by_index = {
        item["arg_index"]: set(item["aliases"]) for item in normalized_inputs
    }
    for left in valid_indices:
        for right in valid_indices:
            if (right in aliases_by_index[left]) != (left in aliases_by_index[right]):
                raise ContractError("contract alias relationships must be symmetric")

    execution = raw.get("execution")
    if not isinstance(execution, Mapping):
        raise ContractError("contract.execution must be an object")
    _exact_fields(execution, _EXECUTION_FIELDS, "contract.execution")
    if set(execution) != _EXECUTION_FIELDS:
        raise ContractError("contract.execution must declare every execution field")
    backward = execution.get("backward")
    deterministic = execution.get("deterministic")
    compare_input_side_effects = execution.get("compare_input_side_effects")
    compare_module_state = execution.get("compare_module_state")
    repeat_count_max = execution.get("repeat_count_max")
    backward_vjp_count = execution.get("backward_vjp_count")
    if any(
        not isinstance(value, bool)
        for value in (
            backward,
            deterministic,
            compare_input_side_effects,
            compare_module_state,
        )
    ):
        raise ContractError(
            "contract.execution backward/deterministic/side-effect fields must be boolean"
        )
    if (
        isinstance(repeat_count_max, bool)
        or not isinstance(repeat_count_max, int)
        or repeat_count_max < 1
    ):
        raise ContractError("contract.execution.repeat_count_max must be positive")
    if (
        isinstance(backward_vjp_count, bool)
        or not isinstance(backward_vjp_count, int)
        or backward_vjp_count < (3 if backward else 1)
    ):
        raise ContractError(
            "contract.execution.backward_vjp_count must be at least three when "
            "backward is in scope"
        )
    streams = execution.get("streams")
    if not isinstance(streams, list) or not streams or any(
        stream not in {"default", "nondefault", "concurrent"} for stream in streams
    ):
        raise ContractError("contract.execution.streams is invalid")
    if streams != ["default"]:
        raise ContractError(
            "contract schema 1.0 executes only the default stream; "
            "nondefault/concurrent stream testing is explicitly unsupported"
        )
    normalized_execution = {
        "modes": _string_list(execution.get("modes"), "contract.execution.modes", ALLOWED_MODES),
        "backward": backward,
        "deterministic": deterministic,
        "repeat_count_max": repeat_count_max,
        "streams": list(streams),
        "compare_input_side_effects": compare_input_side_effects,
        "compare_module_state": compare_module_state,
        "backward_vjp_count": backward_vjp_count,
    }

    oracle = raw.get("oracle")
    if not isinstance(oracle, Mapping):
        raise ContractError("contract.oracle must be an object")
    _exact_fields(oracle, _ORACLE_FIELDS, "contract.oracle")
    if set(oracle) != _ORACLE_FIELDS:
        raise ContractError("contract.oracle must declare every oracle field")
    normalized_oracle: Dict[str, Any] = {
        "atol": _nonnegative_float(oracle.get("atol"), "contract.oracle.atol"),
        "rtol": _nonnegative_float(oracle.get("rtol"), "contract.oracle.rtol"),
        "dtype_tolerances": _validate_tolerance_map(oracle.get("dtype_tolerances")),
    }
    for field in (
        "equal_nan",
        "require_dtype",
        "require_device",
        "require_layout",
        "require_stride",
        "require_aliasing",
    ):
        value = oracle.get(field)
        if not isinstance(value, bool):
            raise ContractError(f"contract.oracle.{field} must be boolean")
        normalized_oracle[field] = value

    bindings = raw.get("policy_bindings")
    if not isinstance(bindings, Mapping):
        raise ContractError("contract.policy_bindings must be an object")
    normalized_bindings = {
        _nonempty_string(policy, "policy binding name"): _validate_index_list(
            bound, f"contract.policy_bindings.{policy}", valid_indices
        )
        for policy, bound in bindings.items()
    }
    from src.stress.policy_bank import STRESS_POLICIES

    unknown_policies = set(normalized_bindings) - set(STRESS_POLICIES)
    if unknown_policies:
        raise ContractError(
            f"contract binds unknown stress policies: {sorted(unknown_policies)}"
        )
    task_specific_indices = {
        item["arg_index"]
        for item in normalized_inputs
        if item["value_domain"]["kind"] == "task_specific"
    }
    for policy, bound in normalized_bindings.items():
        if task_specific_indices.intersection(bound):
            raise ContractError(
                f"generic policy {policy!r} cannot target a task_specific domain; "
                "register an executable domain checker first"
            )
        for index in bound:
            declared_dtypes = set(
                next(
                    item["dtypes"]
                    for item in normalized_inputs
                    if item["arg_index"] == index
                )
            )
            if not declared_dtypes or not declared_dtypes <= REAL_FLOAT_DTYPES:
                raise ContractError(
                    f"generic value policy {policy!r} targets argument {index} with "
                    "a non-real-floating dtype; schema v1 forbids silent no-op stress"
                )

    adapters = raw.get("input_adapters", {})
    if not isinstance(adapters, Mapping):
        raise ContractError("contract.input_adapters must be an object")
    _exact_fields(adapters, {"batch", "layout", "dtype"}, "contract.input_adapters")
    normalized_adapters: Dict[str, Any] = {}
    for name, adapter in adapters.items():
        if not isinstance(adapter, Mapping):
            raise ContractError(f"contract.input_adapters.{name} must be an object")
        allowed = {"arg_indices"}
        if name == "batch":
            allowed |= {"dimension", "allowed_values"}
        elif name == "layout":
            allowed |= {"allowed_values"}
        _exact_fields(adapter, allowed, f"contract.input_adapters.{name}")
        normalized = {
            "arg_indices": _validate_index_list(
                adapter.get("arg_indices"),
                f"contract.input_adapters.{name}.arg_indices",
                valid_indices,
            )
        }
        if name == "batch":
            dimension = adapter.get("dimension")
            allowed_values = adapter.get("allowed_values")
            if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0:
                raise ContractError("batch adapter dimension must be non-negative")
            if not isinstance(allowed_values, list) or not allowed_values or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in allowed_values
            ):
                raise ContractError("batch adapter allowed_values must be positive integers")
            normalized.update({"dimension": dimension, "allowed_values": list(allowed_values)})
        elif name == "layout":
            normalized["allowed_values"] = _string_list(
                adapter.get("allowed_values"),
                "layout adapter allowed_values",
                ALLOWED_LAYOUTS,
            )
        normalized_adapters[name] = normalized

    candidate_classes = raw.get("candidate_classes", ["ModelNew", "Model"])
    if not isinstance(candidate_classes, list) or not candidate_classes or any(
        not isinstance(item, str) or not item for item in candidate_classes
    ):
        raise ContractError("contract.candidate_classes must be a non-empty string list")

    result: Dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": _nonempty_string(raw.get("contract_id"), "contract.contract_id"),
        "tensor_inputs": normalized_inputs,
        "execution": normalized_execution,
        "oracle": normalized_oracle,
        "policy_bindings": normalized_bindings,
        "input_adapters": normalized_adapters,
        "candidate_classes": list(candidate_classes),
    }
    if "notes" in raw:
        result["notes"] = _nonempty_string(raw.get("notes"), "contract.notes")
    return result


def assert_case_in_contract(case: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    """Reject a planned test that is not explicitly authorized by a contract."""

    policy = str(case.get("policy"))
    parameters = validate_case_parameters(case.get("parameters", {}), str(case.get("mode", "eval")))
    mode = str(case.get("mode", "eval"))
    execution = contract["execution"]
    if mode not in execution["modes"]:
        raise ContractError(f"mode {mode!r} is outside contract {contract['contract_id']}")
    if policy not in {"iid", "identity"} and policy not in contract["policy_bindings"]:
        raise ContractError(
            f"policy {policy!r} has no argument binding in contract {contract['contract_id']}"
        )

    tensor_by_index = {
        item["arg_index"]: item for item in contract["tensor_inputs"]
    }
    dtype = parameters.get("dtype")
    if dtype is not None:
        adapter = contract["input_adapters"].get("dtype")
        if adapter is None:
            raise ContractError("dtype case has no contract dtype adapter")
        for index in adapter["arg_indices"]:
            if dtype not in tensor_by_index[index]["dtypes"]:
                raise ContractError(f"dtype {dtype!r} is outside tensor argument {index} contract")

    if parameters.get("requires_backward") and not execution["backward"]:
        raise ContractError("backward case is outside the execution contract")

    if "batch_size" in parameters:
        adapter = contract["input_adapters"].get("batch")
        if adapter is None or parameters["batch_size"] not in adapter["allowed_values"]:
            raise ContractError("batch-size case is outside the input contract")

    layout = parameters.get("layout")
    if layout is not None:
        adapter = contract["input_adapters"].get("layout")
        if adapter is None or layout not in adapter["allowed_values"]:
            raise ContractError("layout case is outside the input contract")
        for index in adapter["arg_indices"]:
            layouts = tensor_by_index[index]["layouts"]
            if layout not in layouts and "any" not in layouts:
                raise ContractError(f"layout {layout!r} is outside tensor argument {index} contract")

    if mode == "repeated":
        repeat_count = parameters.get("repeat_count")
        if not execution["deterministic"]:
            raise ContractError("determinism is not required by this contract")
        if repeat_count > execution["repeat_count_max"]:
            raise ContractError("repeat_count exceeds the execution contract")


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device or left.layout != right.layout:
        return False
    if left.layout != torch.strided:
        return left is right
    return left.untyped_storage()._cdata == right.untyped_storage()._cdata


def validate_call_inputs(args: Sequence[Any], contract: Mapping[str, Any]) -> None:
    """Validate actual call inputs against the frozen tensor contract."""

    tensor_by_index: Dict[int, torch.Tensor] = {}
    symbolic_dimensions: Dict[str, int] = {}
    for specification in contract["tensor_inputs"]:
        index = specification["arg_index"]
        if index >= len(args) or not isinstance(args[index], torch.Tensor):
            raise ContractError(f"argument {index} is not the contracted tensor input")
        tensor = args[index]
        tensor_by_index[index] = tensor
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        if dtype_name not in specification["dtypes"]:
            raise ContractError(
                f"argument {index} dtype {dtype_name!r} is outside the contract"
            )
        if len(tensor.shape) != len(specification["shape"]):
            raise ContractError(f"argument {index} rank is outside the contract")
        for dimension_index, (actual, expected) in enumerate(
            zip(tensor.shape, specification["shape"])
        ):
            actual_size = int(actual)
            if isinstance(expected, int):
                valid = actual_size == expected
            else:
                valid = expected["min"] <= actual_size <= expected["max"]
            if not valid:
                raise ContractError(
                    f"argument {index} dimension {dimension_index} is outside the contract"
                )
            if isinstance(expected, Mapping):
                name = expected["name"]
                previous = symbolic_dimensions.setdefault(name, actual_size)
                if previous != actual_size:
                    raise ContractError(
                        f"symbolic dimension {name!r} is inconsistent across inputs"
                    )

        layouts = specification["layouts"]
        if "any" not in layouts:
            actual_layouts = {str(tensor.layout).removeprefix("torch.")}
            if tensor.layout == torch.strided:
                actual_layouts.add(
                    "contiguous" if tensor.is_contiguous() else "noncontiguous"
                )
                actual_layouts.add("strided")
            if not actual_layouts.intersection(layouts):
                raise ContractError(f"argument {index} layout is outside the contract")
        if bool(tensor.requires_grad) != specification["requires_grad"]:
            raise ContractError(
                f"argument {index} requires_grad differs from the contract"
            )

        domain = specification["value_domain"]
        kind = domain["kind"]
        if tensor.numel() == 0 or kind == "task_specific":
            continue
        domain_tensor = (
            tensor.to_dense()
            if tensor.layout == torch.sparse_coo
            else tensor
        )
        if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
            if not bool(torch.isfinite(domain_tensor).all().item()):
                raise ContractError(f"argument {index} contains non-finite values")
        if tensor.dtype.is_complex and kind != "unrestricted_finite":
            raise ContractError(
                f"argument {index} complex value domain requires a task-specific checker"
            )
        if kind == "bounded":
            if not bool(
                (
                    (domain_tensor >= domain["min"])
                    & (domain_tensor <= domain["max"])
                ).all().item()
            ):
                raise ContractError(f"argument {index} is outside its bounded domain")
        elif kind == "positive" and not bool((domain_tensor > 0).all().item()):
            raise ContractError(f"argument {index} is not strictly positive")
        elif kind == "nonnegative" and not bool((domain_tensor >= 0).all().item()):
            raise ContractError(f"argument {index} is negative")
        elif kind == "probability" and not bool(
            ((domain_tensor >= 0) & (domain_tensor <= 1)).all().item()
        ):
            raise ContractError(f"argument {index} is outside [0, 1]")
        elif kind == "integer_index" and tensor.dtype.is_floating_point:
            raise ContractError(f"argument {index} is not an integer index tensor")

    for left_index, left in tensor_by_index.items():
        declared_aliases = set(
            next(
                item["aliases"]
                for item in contract["tensor_inputs"]
                if item["arg_index"] == left_index
            )
        )
        for right_index, right in tensor_by_index.items():
            if right_index <= left_index:
                continue
            actual = _shares_storage(left, right)
            declared = right_index in declared_aliases
            if actual != declared:
                raise ContractError(
                    f"arguments {left_index} and {right_index} alias relationship "
                    "differs from the contract"
                )
