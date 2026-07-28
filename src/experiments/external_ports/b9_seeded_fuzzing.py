"""B9: seeded differential fuzzing port (gpuemu, Sarkar 2026, @ 696b510b).

Upstream protocol (gpuemu daemon + gpuemu-corpus drivers/_p1lib.py):
  * op-schema-aware sampling — shapes drawn from per-dimension candidate
    lists in the op schema; dtypes from the op's declared list; values
    uniform in [-10, 10] (the Rust fuzzer's range);
  * fp64 CPU reference — the op's reference script computes in float64 and
    rounds back to the kernel output dtype (the correctly-rounded ideal);
  * per-(op,dtype) calibrated absolute tolerances — corpus meta.json values,
    calibrated as p95 of the correct controls' max-abs error x 1.5;
  * comparison semantics (validator.rs): shape mismatch, NaN, Inf, then
    count of |err| > tol; error statistics (max/mean abs, rel, percentiles).

Port mapping onto frozen MutaKernel subjects:
  * op schema      <- the subject contract: fixed shapes stay fixed; the
    contract's batch adapter (when present) provides the sampled dimension
    candidates; dtypes = contract floating dtypes intersected with the
    calibrated set;
  * value sampling -> :func:`b9_uniform10` (uniform [-10, 10], floating
    tensors only, shape/dtype/layout preserved);
  * fp64 reference -> :func:`run_fp64_cpu_reference`;
  * tolerance      -> :func:`calibrate_tolerance` (p95 x 1.5 over control
    errors), with :data:`B9_DEFAULT_TOLERANCES` as the documented fallback
    until per-subject controls are executed;
  * comparison     -> :func:`compare_to_fp64` mirrors validator.rs failure
    kinds and error statistics.

Budget-matched layout: 32 sampled cases, one candidate invocation each
(fp64 reference executions are reference-side and are not charged to the
candidate budget — identical accounting to every other row).

UNSUPPORTED upstream clauses (alignment checklist): the Rust daemon's
bit-exact case generator (we mirror the documented distribution, not the
byte stream), layout-metadata fuzzing beyond the contract's layouts, and the
daemon IPC protocol itself.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import torch

from src.experiments.strategy import StrategySpec, TestCaseSpec
from src.stress.policy_bank import _make_policy

from .budget32 import UNIFIED_CANDIDATE_BUDGET, assert_budget_matched

UPSTREAM_COMMIT = "696b510bc21f8036bcee395749b7fa2c2b4baf2a"
VALUE_RANGE = 10.0  # Rust fuzzer: uniform [-10, 10]
CALIBRATION_QUANTILE = 0.95
CALIBRATION_FACTOR = 1.5

# Corpus meta.json convention (e.g. softmax_triton): documented fallback
# until per-(subject,dtype) control calibration has run.
B9_DEFAULT_TOLERANCES = {"float32": 1e-5, "float16": 2e-2}

B9_PORT_STRATEGY = StrategySpec(
    name="b9-seeded-fuzzing-style-port",
    version="1",
    parameters={
        "comparison_role": "external_protocol_port",
        "derived_from_commit": UPSTREAM_COMMIT,
        "sampling_family": "op_schema_uniform10",
        "oracle": "fp64_cpu_reference_calibrated_abs_tol",
        "budget_matched": True,
        "calibration": "p95_of_controls_x_1.5",
    },
)


def _b9_uniform_values(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    return (
        torch.rand(t.shape, dtype=torch.float32, generator=g) * 2.0 - 1.0
    ) * VALUE_RANGE


b9_uniform10 = _make_policy(_b9_uniform_values)

B9_PORT_POLICIES: Dict[str, Any] = {"b9_uniform10": b9_uniform10}


def plan_b9_port(
    subject_id: str,
    dtypes: Sequence[str] = ("float32", "float16"),
    batch_values: Optional[Sequence[int]] = None,
) -> List[TestCaseSpec]:
    """32 schema-sampled cases, round-robin over (dtype x batch candidate).

    ``dtypes`` must stay within the calibrated tolerance set; ``batch_values``
    comes from the subject contract's batch adapter (None = shapes fixed by
    the contract, exactly like a schema whose dims have a single candidate).
    """

    usable_dtypes = [d for d in dtypes if d in B9_DEFAULT_TOLERANCES]
    if not usable_dtypes:
        raise ValueError(
            f"no calibrated dtype among {list(dtypes)}; "
            f"calibrated set is {sorted(B9_DEFAULT_TOLERANCES)}"
        )
    contexts: List[Dict[str, Any]] = []
    for dtype in usable_dtypes:
        if batch_values:
            for batch in batch_values:
                contexts.append({"dtype": dtype, "batch_size": int(batch)})
        else:
            contexts.append({"dtype": dtype})

    cases = []
    for seed in range(UNIFIED_CANDIDATE_BUDGET):
        context = contexts[seed % len(contexts)]
        cases.append(
            TestCaseSpec(
                subject_id=subject_id,
                strategy=B9_PORT_STRATEGY,
                policy="b9_uniform10",
                seed=seed,
                mode="eval",
                scope="external_port",
                parameters=dict(context),
            )
        )
    return list(assert_budget_matched(cases))


# ---------------------------------------------------------------------------
# fp64 CPU reference + validator.rs comparison semantics
# ---------------------------------------------------------------------------

def run_fp64_cpu_reference(
    reference_module: torch.nn.Module,
    args: Sequence[Any],
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute the correctly-rounded ideal: float64 on CPU, rounded back.

    Mirrors the gpuemu reference scripts (_refkit.py): references compute in
    float64 and return the SAME dtype as the kernel output.
    """

    module = reference_module.to("cpu").double()
    converted = [
        arg.detach().to("cpu", torch.float64)
        if isinstance(arg, torch.Tensor) and arg.is_floating_point()
        else (arg.detach().to("cpu") if isinstance(arg, torch.Tensor) else arg)
        for arg in args
    ]
    with torch.no_grad():
        ideal = module(*converted)
    if not isinstance(ideal, torch.Tensor):
        raise TypeError("B9 port supports single-tensor outputs (upstream protocol)")
    return ideal.to(output_dtype)


def error_stats(
    output: torch.Tensor, reference: torch.Tensor, tol: float
) -> Dict[str, float]:
    """Mirror _p1lib.error_stats (abs/rel error distribution vs the ideal)."""

    out64 = output.detach().to("cpu", torch.float64).reshape(-1)
    ref64 = reference.detach().to("cpu", torch.float64).reshape(-1)
    abs_err = (out64 - ref64).abs()
    abs_err = torch.where(
        torch.isfinite(abs_err), abs_err, torch.full_like(abs_err, math.inf)
    )
    nonzero = ref64 != 0.0
    if bool(nonzero.any()):
        rel = (out64[nonzero] - ref64[nonzero]).abs() / ref64[nonzero].abs()
    else:
        rel = torch.zeros(1, dtype=torch.float64)
    finite_abs = abs_err[torch.isfinite(abs_err)]

    def pct(q: float) -> float:
        if finite_abs.numel() == 0:
            return 0.0
        return float(torch.quantile(finite_abs, q).item())

    return {
        "count": int(abs_err.numel()),
        "num_exceeding": int((abs_err > tol).sum().item()),
        "max_abs": float(abs_err.max().item()) if abs_err.numel() else 0.0,
        "mean_abs": float(finite_abs.mean().item()) if finite_abs.numel() else math.inf,
        "p50_abs": pct(0.50),
        "p90_abs": pct(0.90),
        "p99_abs": pct(0.99),
        "max_rel": float(rel.max().item()) if rel.numel() else 0.0,
        "mean_rel": float(rel.mean().item()) if rel.numel() else 0.0,
    }


def compare_to_fp64(
    output: torch.Tensor, reference: torch.Tensor, tol: float
) -> Dict[str, Any]:
    """Mirror validator.rs: shape, NaN, Inf, absolute-tolerance count."""

    if tuple(output.shape) != tuple(reference.shape):
        return {
            "passed": False,
            "failure_kind": "shape",
            "max_abs_err": math.inf,
            "max_rel_err": math.inf,
            "error_stats": None,
        }
    stats = error_stats(output, reference, tol)
    out64 = output.detach().to("cpu", torch.float64)
    failure_kind = None
    if bool(torch.isnan(out64).any()):
        failure_kind = "nan"
    elif bool(torch.isinf(out64).any()):
        failure_kind = "inf"
    elif stats["num_exceeding"] > 0:
        failure_kind = "tolerance"
    return {
        "passed": failure_kind is None,
        "failure_kind": failure_kind,
        "max_abs_err": stats["max_abs"],
        "max_rel_err": stats["max_rel"],
        "error_stats": stats,
    }


def calibrate_tolerance(
    control_max_abs_errors: Sequence[float],
    quantile: float = CALIBRATION_QUANTILE,
    factor: float = CALIBRATION_FACTOR,
) -> float:
    """Upstream calibration: p95 of correct controls' max-abs error x 1.5."""

    if not control_max_abs_errors:
        raise ValueError("calibration requires at least one control error")
    errors = torch.tensor(sorted(control_max_abs_errors), dtype=torch.float64)
    if not bool(torch.isfinite(errors).all()):
        raise ValueError("control errors must be finite")
    return float(torch.quantile(errors, quantile).item()) * factor
