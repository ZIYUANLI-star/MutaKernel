"""B7: robust-kbench protocol port (SakanaAI/robust-kbench @ 078f5bab).

Upstream protocol (run_kernel.py / robust_kbench/):
  * multi-init      — the task model is re-instantiated under several init
                      configurations/seeds (ConfigTask ``multi_init_configs``);
  * multi-input     — several fresh ``get_inputs()`` draws per init
                      (``multi_input_configs``);
  * repeated trials — every forward correctness trial invokes the candidate
                      twice (configs/external_baselines.json
                      ``candidate_calls_per_forward_trial: 2``; both calls are
                      charged, per the port reporting rule);
  * forward AND backward comparison where the task defines a backward;
  * statistical output filters (run_filter.py) that flag degenerate tasks:
    output range, output std, per-axis variation, input impact, plus an
    LLM-based sanity check.

Port mapping onto frozen MutaKernel subjects:
  * multi-init  -> ``parameters["init_seed"]`` — the runner re-instantiates
    reference and candidate under ``torch.manual_seed(init_seed)``;
  * multi-input -> policy ``iid`` with distinct seeds (the subject task's own
    ``get_inputs`` distribution, exactly what upstream draws from);
  * repeat pair -> mode ``repeated`` with ``repeat_count=2`` (cost 2);
  * backward    -> mode ``train`` with ``requires_backward`` (cost 1), only
    when the subject contract authorizes backward;
  * oracle      -> unified pipeline (src.validation), replacing upstream's
    ``torch.allclose(atol=rtol=1e-5)``;
  * filters     -> :func:`run_output_filters` ports the four statistical
    filters verbatim (same 0.01 thresholds); the LLM sanity filter is
    UNSUPPORTED in the port (external service; see the alignment checklist).

Budget-matched layout (exactly 32 candidate invocations):
  * backward supported:   2 inits x 4 input draws x repeat-pair (16)
                        + 2 inits x 8 input draws x backward     (16)
  * forward only:         4 inits x 4 input draws x repeat-pair  (32)
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from src.experiments.strategy import StrategySpec, TestCaseSpec

from .budget32 import assert_budget_matched

UPSTREAM_COMMIT = "078f5bab29934a822268d59a4e707d449abf9b4e"
FILTER_THRESHOLD = 0.01  # upstream run_filter.py uses 0.01 everywhere

B7_PORT_STRATEGY = StrategySpec(
    name="b7-robust-kbench-style-port",
    version="1",
    parameters={
        "comparison_role": "external_protocol_port",
        "derived_from_commit": UPSTREAM_COMMIT,
        "sampling_family": "kernelbench_default",
        "oracle": "unified",
        "budget_matched": True,
        "counting_rule": "both candidate invocations of a repeated forward trial are charged",
    },
)

UNSUPPORTED_CLAUSES = (
    "LLM sanity filter (run_filter.py filter_llm_sanity): external "
    "frontier-LLM dependency; not executed in the port",
    "NCU/clang-tidy profiling and speedup measurement (prof_cuda_kernel): "
    "performance measurement is outside the correctness-port scope",
    "upstream .cu task interface (forward.cu/backward.cu compilation): the "
    "port drives the subject's own candidate module instead",
)


def plan_b7_port(subject_id: str, backward_supported: bool) -> List[TestCaseSpec]:
    """Exactly 32 candidate invocations per subject (blueprint Table 2)."""

    cases: List[TestCaseSpec] = []
    if backward_supported:
        forward_layout = [(init, draw) for init in range(2) for draw in range(4)]
        backward_layout = [(init, draw) for init in range(2) for draw in range(8)]
    else:
        forward_layout = [(init, draw) for init in range(4) for draw in range(4)]
        backward_layout = []

    for init_seed, input_seed in forward_layout:
        cases.append(
            TestCaseSpec(
                subject_id=subject_id,
                strategy=B7_PORT_STRATEGY,
                policy="iid",
                seed=input_seed,
                mode="repeated",
                scope="external_port",
                parameters={"repeat_count": 2, "init_seed": init_seed},
            )
        )
    for init_seed, input_seed in backward_layout:
        cases.append(
            TestCaseSpec(
                subject_id=subject_id,
                strategy=B7_PORT_STRATEGY,
                policy="iid",
                seed=100 + input_seed,
                mode="train",
                scope="external_port",
                parameters={"requires_backward": True, "init_seed": init_seed},
            )
        )
    return list(assert_budget_matched(cases))


# ---------------------------------------------------------------------------
# Statistical output filters — verbatim port of run_filter.py checks 1-5.
# The filters characterize the *task protocol* (degenerate reference outputs),
# never the candidate verdict, exactly as upstream applies them.
# ---------------------------------------------------------------------------

def _stack(outputs: Sequence[torch.Tensor]) -> torch.Tensor:
    if not outputs:
        raise ValueError("filters require at least one output")
    return torch.stack([out.detach().float().cpu() for out in outputs])


def filter_output_range(outputs: Sequence[torch.Tensor]) -> bool:
    """True when ALL values across seeds sit inside (-0.01, 0.01)."""
    stacked = _stack(outputs)
    return bool(((stacked > -FILTER_THRESHOLD) & (stacked < FILTER_THRESHOLD)).all())


def filter_output_std(outputs: Sequence[torch.Tensor]) -> bool:
    """True when the per-element std across seeds is everywhere < 0.01."""
    stacked = _stack(outputs)
    if stacked.shape[0] < 2:
        raise ValueError("std filter requires >= 2 seed outputs")
    return bool((torch.std(stacked, dim=0) < FILTER_THRESHOLD).all())


def filter_output_axes(outputs: Sequence[torch.Tensor]) -> bool:
    """True when ANY axis of the stacked outputs has uniformly tiny std."""
    stacked = _stack(outputs)
    if stacked.shape[0] < 2:
        raise ValueError("axes filter requires >= 2 seed outputs")
    for axis in range(stacked.ndim):
        if stacked.shape[axis] < 2:
            continue
        if bool((torch.std(stacked, dim=axis) < FILTER_THRESHOLD).all()):
            return True
    return False


def filter_input_impact(outputs_fixed_init: Sequence[torch.Tensor]) -> bool:
    """True when varying inputs (fixed init) leaves the output unchanged."""
    return filter_output_std(outputs_fixed_init)


def run_output_filters(
    outputs_by_seed: Sequence[torch.Tensor],
    outputs_fixed_init: Sequence[torch.Tensor],
) -> Dict[str, bool]:
    return {
        "filter_output_range": filter_output_range(outputs_by_seed),
        "filter_output_std": filter_output_std(outputs_by_seed),
        "filter_output_axes": filter_output_axes(outputs_by_seed),
        "filter_input_impact": filter_input_impact(outputs_fixed_init),
        "filter_llm_sanity": None,  # UNSUPPORTED in the port (see docstring)
    }
