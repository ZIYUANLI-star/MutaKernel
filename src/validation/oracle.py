"""Structure-, shape-, dtype-, and non-finite-aware output oracle."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .types import Mismatch, OracleResult, ValidationStatus

try:  # NumPy outputs are supported when NumPy is installed.
    import numpy as np
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    np = None


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float


def _default_dtype_tolerances() -> Dict[torch.dtype, Tolerance]:
    return {
        torch.float16: Tolerance(rtol=1e-3, atol=1e-3),
        torch.bfloat16: Tolerance(rtol=1e-3, atol=1e-3),
        torch.float32: Tolerance(rtol=1e-5, atol=1e-6),
        torch.float64: Tolerance(rtol=1e-7, atol=1e-9),
        torch.complex64: Tolerance(rtol=1e-5, atol=1e-6),
        torch.complex128: Tolerance(rtol=1e-7, atol=1e-9),
    }


@dataclass(frozen=True)
class OracleConfig:
    """Comparison policy for structured outputs."""

    default_tolerance: Tolerance = Tolerance(rtol=1e-5, atol=1e-8)
    dtype_tolerances: Mapping[torch.dtype, Tolerance] = field(
        default_factory=_default_dtype_tolerances,
    )
    equal_nan: bool = True
    require_dtype: bool = True
    require_device: bool = True
    require_layout: bool = True
    require_stride: bool = False
    require_aliasing: bool = False
    max_mismatches: int = 20

    def tolerance_for(self, dtype: torch.dtype) -> Tolerance:
        return self.dtype_tolerances.get(dtype, self.default_tolerance)


def _summary(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device}, layout={value.layout})"
        )
    if np is not None and isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    text = repr(value)
    return text if len(text) <= 160 else text[:157] + "..."


class _Comparison:
    def __init__(self, config: OracleConfig):
        self.config = config
        self.compared_leaves = 0
        self.mismatches: List[Mismatch] = []
        self.definite_failure = False
        self.inconclusive = False
        self.tensor_pairs: List[Tuple[str, torch.Tensor, torch.Tensor]] = []

    @property
    def full(self) -> bool:
        return len(self.mismatches) >= self.config.max_mismatches

    def record(
        self,
        path: str,
        kind: str,
        message: str,
        reference: Any,
        candidate: Any,
        *,
        conclusive: bool = True,
    ) -> None:
        if not self.full:
            self.mismatches.append(
                Mismatch(
                    path=path,
                    kind=kind,
                    message=message,
                    reference=_summary(reference),
                    candidate=_summary(candidate),
                )
            )
        if conclusive:
            self.definite_failure = True
        else:
            self.inconclusive = True


def _nonfinite_and_close(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    tolerance: Tolerance,
    equal_nan: bool,
) -> Tuple[bool, str]:
    """Compare one real floating tensor with explicit NaN/Inf handling."""

    reference_nan = torch.isnan(reference)
    candidate_nan = torch.isnan(candidate)
    if not torch.equal(reference_nan, candidate_nan):
        count = int(torch.count_nonzero(reference_nan != candidate_nan).item())
        return False, f"NaN positions differ at {count} element(s)"
    if torch.any(reference_nan) and not equal_nan:
        return False, "NaN values are present while equal_nan=False"

    reference_inf = torch.isinf(reference)
    candidate_inf = torch.isinf(candidate)
    if not torch.equal(reference_inf, candidate_inf):
        count = int(torch.count_nonzero(reference_inf != candidate_inf).item())
        return False, f"Inf positions differ at {count} element(s)"
    if torch.any(reference_inf):
        if not torch.equal(reference[reference_inf], candidate[candidate_inf]):
            return False, "Inf signs differ"

    finite = ~(reference_nan | reference_inf)
    if not torch.any(finite):
        return True, ""
    reference_finite = reference[finite]
    candidate_finite = candidate[finite]
    close = torch.isclose(
        reference_finite,
        candidate_finite,
        rtol=tolerance.rtol,
        atol=tolerance.atol,
        equal_nan=False,
    )
    if bool(torch.all(close).item()):
        return True, ""
    count = int(torch.count_nonzero(~close).item())
    absolute_error = torch.abs(reference_finite - candidate_finite)
    max_error = float(torch.max(absolute_error).item())
    return (
        False,
        f"{count} value(s) exceed rtol={tolerance.rtol:g}, "
        f"atol={tolerance.atol:g}; max_abs_error={max_error:g}",
    )


def _compare_tensors(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    path: str,
    state: _Comparison,
) -> None:
    state.compared_leaves += 1
    state.tensor_pairs.append((path, reference, candidate))
    config = state.config
    if tuple(reference.shape) != tuple(candidate.shape):
        state.record(path, "shape", "tensor shapes differ", reference, candidate)
        return
    if config.require_dtype and reference.dtype != candidate.dtype:
        state.record(path, "dtype", "tensor dtypes differ", reference, candidate)
        return
    if config.require_device and reference.device != candidate.device:
        state.record(path, "device", "tensor devices differ", reference, candidate)
        return
    if config.require_layout and reference.layout != candidate.layout:
        state.record(path, "layout", "tensor layouts differ", reference, candidate)
        return
    if (
        config.require_stride
        and reference.layout == torch.strided
        and candidate.layout == torch.strided
        and reference.stride() != candidate.stride()
    ):
        state.record(path, "stride", "tensor strides differ", reference, candidate)
        return
    if reference.is_quantized or candidate.is_quantized:
        state.record(
            path,
            "unsupported",
            "quantized tensor comparison requires an explicit quantization contract",
            reference,
            candidate,
            conclusive=False,
        )
        return

    if reference.layout != torch.strided:
        reference = reference.to_dense()
        candidate = candidate.to_dense()

    if reference.dtype == torch.bool or not (
        reference.dtype.is_floating_point or reference.dtype.is_complex
    ):
        if not torch.equal(reference, candidate):
            count = int(torch.count_nonzero(reference != candidate).item())
            state.record(
                path,
                "value",
                f"{count} integer/bool value(s) differ; exact equality is required",
                reference,
                candidate,
            )
        return

    tolerance = config.tolerance_for(reference.dtype)
    if reference.dtype.is_complex:
        real_ok, real_message = _nonfinite_and_close(
            reference.real,
            candidate.real,
            tolerance,
            config.equal_nan,
        )
        imag_ok, imag_message = _nonfinite_and_close(
            reference.imag,
            candidate.imag,
            tolerance,
            config.equal_nan,
        )
        if not real_ok or not imag_ok:
            message = "; ".join(
                part
                for part in (
                    f"real: {real_message}" if real_message else "",
                    f"imag: {imag_message}" if imag_message else "",
                )
                if part
            )
            state.record(path, "value", message, reference, candidate)
        return

    matches, message = _nonfinite_and_close(
        reference,
        candidate,
        tolerance,
        config.equal_nan,
    )
    if not matches:
        state.record(path, "value", message, reference, candidate)


def _compare_float_scalars(
    reference: float,
    candidate: float,
    path: str,
    state: _Comparison,
) -> None:
    state.compared_leaves += 1
    if math.isnan(reference) or math.isnan(candidate):
        if not (
            state.config.equal_nan
            and math.isnan(reference)
            and math.isnan(candidate)
        ):
            state.record(path, "value", "scalar NaN values differ", reference, candidate)
        return
    if math.isinf(reference) or math.isinf(candidate):
        if reference != candidate:
            state.record(path, "value", "scalar Inf values or signs differ", reference, candidate)
        return
    tolerance = state.config.default_tolerance
    if not math.isclose(reference, candidate, rel_tol=tolerance.rtol, abs_tol=tolerance.atol):
        state.record(
            path,
            "value",
            f"scalars exceed rtol={tolerance.rtol:g}, atol={tolerance.atol:g}",
            reference,
            candidate,
        )


def _compare_numpy(
    reference: Any,
    candidate: Any,
    path: str,
    state: _Comparison,
) -> None:
    state.compared_leaves += 1
    if reference.shape != candidate.shape:
        state.record(path, "shape", "array shapes differ", reference, candidate)
        return
    if state.config.require_dtype and reference.dtype != candidate.dtype:
        state.record(path, "dtype", "array dtypes differ", reference, candidate)
        return
    if reference.dtype.kind in "biu":
        if not np.array_equal(reference, candidate):
            state.record(path, "value", "integer/bool array values differ", reference, candidate)
        return
    if reference.dtype.kind in "fc":
        # Reuse the thoroughly tested Torch path without incrementing leaves twice.
        state.compared_leaves -= 1
        _compare_tensors(torch.from_numpy(reference), torch.from_numpy(candidate), path, state)
        return
    if not np.array_equal(reference, candidate):
        state.record(path, "value", "array values differ", reference, candidate)


def _compare(reference: Any, candidate: Any, path: str, state: _Comparison) -> None:
    if isinstance(reference, torch.Tensor) or isinstance(candidate, torch.Tensor):
        if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
            state.record(path, "structure", "only one output is a tensor", reference, candidate)
            return
        _compare_tensors(reference, candidate, path, state)
        return

    if np is not None and (
        isinstance(reference, np.ndarray) or isinstance(candidate, np.ndarray)
    ):
        if not isinstance(reference, np.ndarray) or not isinstance(candidate, np.ndarray):
            state.record(path, "structure", "only one output is an ndarray", reference, candidate)
            return
        _compare_numpy(reference, candidate, path, state)
        return

    if type(reference) is not type(candidate):
        state.record(
            path,
            "structure",
            "output types differ",
            reference,
            candidate,
        )
        return

    if dataclasses.is_dataclass(reference) and not isinstance(reference, type):
        if type(reference) is not type(candidate):
            state.record(path, "structure", "dataclass types differ", reference, candidate)
            return
        for data_field in dataclasses.fields(reference):
            _compare(
                getattr(reference, data_field.name),
                getattr(candidate, data_field.name),
                f"{path}.{data_field.name}",
                state,
            )
        return

    if isinstance(reference, Mapping):
        reference_keys = set(reference.keys())
        candidate_keys = set(candidate.keys())
        if reference_keys != candidate_keys:
            state.record(
                path,
                "structure",
                "mapping keys differ",
                sorted(map(repr, reference_keys)),
                sorted(map(repr, candidate_keys)),
            )
            return
        for key in reference.keys():
            _compare(reference[key], candidate[key], f"{path}[{key!r}]", state)
        return

    if isinstance(reference, (tuple, list)):
        if len(reference) != len(candidate):
            state.record(path, "structure", "sequence lengths differ", reference, candidate)
            return
        if hasattr(reference, "_fields") and getattr(reference, "_fields") != getattr(
            candidate, "_fields", None
        ):
            state.record(path, "structure", "namedtuple fields differ", reference, candidate)
            return
        for index, (reference_item, candidate_item) in enumerate(zip(reference, candidate)):
            _compare(reference_item, candidate_item, f"{path}[{index}]", state)
        return

    if reference is None or isinstance(reference, (str, bytes, bool, int)):
        state.compared_leaves += 1
        if reference != candidate:
            state.record(path, "value", "exact scalar values differ", reference, candidate)
        return

    if isinstance(reference, float):
        _compare_float_scalars(reference, candidate, path, state)
        return

    if isinstance(reference, complex):
        _compare_float_scalars(reference.real, candidate.real, f"{path}.real", state)
        _compare_float_scalars(reference.imag, candidate.imag, f"{path}.imag", state)
        return

    state.record(
        path,
        "unsupported",
        f"no sound oracle is registered for output type {type(reference).__name__}",
        reference,
        candidate,
        conclusive=False,
    )


def _tensors_share_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.device != right.device or left.layout != right.layout:
        return False
    if left.layout != torch.strided:
        return left is right
    try:
        return left.untyped_storage()._cdata == right.untyped_storage()._cdata
    except Exception:  # pragma: no cover - backend-specific storage fallback.
        return left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()


def compare_outputs(
    reference: Any,
    candidate: Any,
    config: Optional[OracleConfig] = None,
) -> OracleResult:
    """Compare arbitrary nested outputs without implicit casts or flattening."""

    resolved_config = config or OracleConfig()
    if resolved_config.max_mismatches <= 0:
        raise ValueError("max_mismatches must be positive")
    state = _Comparison(resolved_config)
    try:
        _compare(reference, candidate, "$", state)
        if resolved_config.require_aliasing:
            for left_index, (left_path, left_ref, left_candidate) in enumerate(
                state.tensor_pairs
            ):
                for right_path, right_ref, right_candidate in state.tensor_pairs[
                    left_index + 1 :
                ]:
                    ref_alias = _tensors_share_storage(left_ref, right_ref)
                    candidate_alias = _tensors_share_storage(
                        left_candidate, right_candidate
                    )
                    if ref_alias != candidate_alias:
                        state.record(
                            f"{left_path}<->{right_path}",
                            "aliasing",
                            "output storage-alias relationship differs",
                            f"aliases={ref_alias}",
                            f"aliases={candidate_alias}",
                        )
    except Exception as exc:
        return OracleResult(
            status=ValidationStatus.INCONCLUSIVE,
            compared_leaves=state.compared_leaves,
            mismatches=state.mismatches,
            reason=f"oracle raised {type(exc).__name__}: {exc}",
        )

    if state.definite_failure:
        status = ValidationStatus.FAIL
        reason = f"observed {len(state.mismatches)} output mismatch(es)"
    elif state.inconclusive:
        status = ValidationStatus.INCONCLUSIVE
        reason = "one or more outputs have no sound registered oracle"
    else:
        status = ValidationStatus.PASS
        reason = f"all {state.compared_leaves} compared output leaf/leaves agree"
    return OracleResult(
        status=status,
        compared_leaves=state.compared_leaves,
        mismatches=state.mismatches,
        reason=reason,
    )
