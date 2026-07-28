"""B8: KernelBenchX protocol port (BonnieW05/KernelBenchX @ fd419229).

Upstream protocol (EVAL/1_exe_acc.py + utils/data_utils.py):
  * input families — ``rand_tensor(mode=...)``:
      standard : N(0,1) draws,
      outlier  : N(0,1) with probability ``outlier_prob=0.001`` elements
                 redrawn as N(0,1) * ``outlier_scale=50``,
      uniform  : U(low, high) (used by a minority of golden tests),
    plus per-task hand-written boundary test cases inside each golden file;
  * seeding — one seed for the whole file run (KERNELBENCHX_SEED);
  * oracle — dtype-aware defaults (fp16/bf16: rtol=atol=5e-3; fp32/fp64:
    1e-5) via ``torch.testing.assert_close``; per-task overrides use
    cosine/L1/RMSE precision thresholds;
  * task frame — the frozen 176-task ``data/kernelbenchx_v1.json`` manifest
    (never the repository directory glob).

Port mapping onto frozen MutaKernel subjects:
  * standard/outlier become value families applied to the subject's template
    inputs (shape/dtype/device preserved; only floating tensors redrawn),
    with per-case seeds — :func:`b8_standard`, :func:`b8_outlier`;
  * budget-matched layout: 16 standard + 16 outlier = exactly 32 candidate
    invocations;
  * hand-written per-task boundary cases are UNSUPPORTED on C2-C5 subjects
    (they exist only for upstream's own 176 tasks; the native mode keeps
    them); recorded in the alignment checklist;
  * oracle — unified pipeline; upstream's dtype-aware defaults are recorded
    here (:data:`B8_NATIVE_DTYPE_TOLERANCES`) for the native anchor row and
    the reproduction delta, never merged into port rows.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from src.experiments.strategy import StrategySpec, TestCaseSpec
from src.stress.policy_bank import _make_policy

from .budget32 import assert_budget_matched

UPSTREAM_COMMIT = "fd4192293bf9a8c645327a9d46aa1e807f1f9cf2"
OUTLIER_PROB = 0.001
OUTLIER_SCALE = 50.0

# Upstream EVAL/1_exe_acc.py::_default_tol — native-mode oracle record.
B8_NATIVE_DTYPE_TOLERANCES = {
    "float16": {"rtol": 5e-3, "atol": 5e-3},
    "bfloat16": {"rtol": 5e-3, "atol": 5e-3},
    "float32": {"rtol": 1e-5, "atol": 1e-5},
    "float64": {"rtol": 1e-5, "atol": 1e-5},
}

B8_PORT_STRATEGY = StrategySpec(
    name="b8-kernelbenchx-style-port",
    version="1",
    parameters={
        "comparison_role": "external_protocol_port",
        "derived_from_commit": UPSTREAM_COMMIT,
        "sampling_family": "kernelbenchx_value_families",
        "oracle": "unified",
        "budget_matched": True,
        "outlier_prob": OUTLIER_PROB,
        "outlier_scale": OUTLIER_SCALE,
    },
)

UNSUPPORTED_CLAUSES = (
    "hand-written per-task boundary test cases (golden files' test_case_*): "
    "defined only for upstream's 176 tasks; native mode only",
    "cosine/L1/RMSE custom precision thresholds (precision_thresholds "
    "overrides): superseded by the unified oracle in port rows; recorded "
    "for the native anchor",
    "kernel-export/AST hygiene gate (impl_must_export_kernel, "
    "check_triton_validity): specific to upstream's Triton file layout",
)


def _b8_standard_values(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    # data_utils.rand_tensor(mode="standard"): plain standard normal.
    return torch.randn(t.shape, dtype=torch.float32, generator=g)


def _b8_outlier_values(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    # data_utils.rand_tensor(mode="outlier"): N(0,1) with outlier_prob
    # elements redrawn as N(0,1) * outlier_scale.
    values = torch.randn(t.shape, dtype=torch.float32, generator=g)
    mask = torch.rand(t.shape, generator=g) < OUTLIER_PROB
    count = int(mask.sum().item())
    if count > 0:
        values[mask] = (
            torch.randn(count, dtype=torch.float32, generator=g) * OUTLIER_SCALE
        )
    return values


def _b8_uniform_values(t: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    # data_utils.rand_tensor(mode="uniform"), upstream defaults low=-1, high=1.
    low, high = -1.0, 1.0
    return (high - low) * torch.rand(
        t.shape, dtype=torch.float32, generator=g
    ) + low


# Same wrapper the M-full policy bank uses: shape/dtype/layout preserved,
# only non-empty floating tensors are redrawn, no silent no-ops.
b8_standard = _make_policy(_b8_standard_values)
b8_outlier = _make_policy(_b8_outlier_values)
b8_uniform = _make_policy(_b8_uniform_values)

B8_PORT_POLICIES: Dict[str, Any] = {
    "b8_standard": b8_standard,
    "b8_outlier": b8_outlier,
    "b8_uniform": b8_uniform,
}


def plan_b8_port(subject_id: str) -> List[TestCaseSpec]:
    """16 standard + 16 outlier cases = exactly 32 candidate invocations."""

    cases = [
        TestCaseSpec(
            subject_id=subject_id,
            strategy=B8_PORT_STRATEGY,
            policy=policy,
            seed=seed,
            mode="eval",
            scope="external_port",
            parameters={},
        )
        for policy in ("b8_standard", "b8_outlier")
        for seed in range(16)
    ]
    return list(assert_budget_matched(cases))
