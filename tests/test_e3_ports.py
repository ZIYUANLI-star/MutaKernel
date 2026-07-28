"""CPU-only tests for the E3 external-baseline ports (B1u/B7/B8/B9).

Covers: unified 32-invocation budget accounting, port plan layouts, the B8
value families, the B7 statistical output filters, the B9 fp64-reference
comparison/calibration, and an end-to-end CPU smoke of the run_e3_external
scheduler on synthetic KernelBench-style task modules (never a C2-C5
candidate; data-separation red line).
"""

import sys
import textwrap
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.external_ports import (
    B1U_CANDIDATE_RUNS,
    UNIFIED_CANDIDATE_BUDGET,
    b8_outlier,
    b8_standard,
    b9_uniform10,
    calibrate_tolerance,
    compare_to_fp64,
    filter_input_impact,
    filter_output_axes,
    filter_output_range,
    filter_output_std,
    fresh_budget_state,
    plan_b1u,
    plan_b7_port,
    plan_b8_port,
    plan_b9_port,
    run_fp64_cpu_reference,
    total_candidate_cost,
)
from src.experiments.external_ports.budget32 import (
    BudgetMismatchError,
    assert_budget_matched,
)
from scripts.run_e3_external import PlanRunner, build_plan, _load_module


# ---------------------------------------------------------------------------
# Budget accounting
# ---------------------------------------------------------------------------

def test_all_port_plans_cost_exactly_32():
    assert total_candidate_cost(plan_b7_port("S", backward_supported=False)) == 32
    assert total_candidate_cost(plan_b7_port("S", backward_supported=True)) == 32
    assert total_candidate_cost(plan_b8_port("S")) == 32
    assert total_candidate_cost(plan_b9_port("S")) == 32
    assert UNIFIED_CANDIDATE_BUDGET == 32


def test_b1u_is_a_five_call_anchor():
    cases = plan_b1u("S")
    assert total_candidate_cost(cases) == B1U_CANDIDATE_RUNS == 5
    assert all(case.policy == "iid" and case.mode == "eval" for case in cases)
    assert cases[0].strategy.parameters["budget_matched"] is False


def test_budget_mismatch_raises():
    cases = plan_b8_port("S")
    with pytest.raises(BudgetMismatchError):
        assert_budget_matched(cases[:-1])


def test_budget_state_blocks_overrun():
    state = fresh_budget_state()
    for _ in range(32):
        state = state.charge(candidate_runs=1)
    assert state.exhausted
    assert not state.can_start(candidate_runs=1).allowed
    with pytest.raises(RuntimeError):
        state.charge(candidate_runs=1)


def test_b7_plan_counts_both_calls_of_a_repeated_trial():
    cases = plan_b7_port("S", backward_supported=True)
    forward = [c for c in cases if c.mode == "repeated"]
    backward = [c for c in cases if c.mode == "train"]
    assert len(forward) == 8 and all(c.candidate_run_cost == 2 for c in forward)
    assert len(backward) == 16 and all(c.candidate_run_cost == 1 for c in backward)
    assert {c.parameters["init_seed"] for c in cases} == {0, 1}
    assert len({c.test_id for c in cases}) == len(cases)


def test_b9_plan_round_robins_calibrated_dtypes():
    cases = plan_b9_port("S", dtypes=("float32", "float16"))
    dtypes = {c.parameters["dtype"] for c in cases}
    assert dtypes == {"float32", "float16"}
    with pytest.raises(ValueError):
        plan_b9_port("S", dtypes=("bfloat16",))  # not in the calibrated set


# ---------------------------------------------------------------------------
# B8 value families (data_utils.rand_tensor semantics)
# ---------------------------------------------------------------------------

def _template():
    torch.manual_seed(7)
    return [torch.randn(64, 128)]


def test_b8_standard_preserves_shape_dtype_and_is_seeded():
    template = _template()
    first = b8_standard(template, 3)
    second = b8_standard(template, 3)
    other = b8_standard(template, 4)
    assert first[0].shape == template[0].shape
    assert first[0].dtype == template[0].dtype
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[0], other[0])


def test_b8_outlier_injects_scaled_extremes():
    template = _template()
    extreme_found = False
    for seed in range(16):
        values = b8_outlier(template, seed)[0]
        assert values.shape == template[0].shape
        if values.abs().max().item() > 10.0:
            extreme_found = True
    assert extreme_found, "outlier family never injected an extreme value"


def test_b9_uniform10_range():
    values = b9_uniform10(_template(), 0)[0]
    assert values.abs().max().item() <= 10.0
    assert values.abs().max().item() > 5.0  # actually spreads over the range


# ---------------------------------------------------------------------------
# B7 statistical output filters (run_filter.py port)
# ---------------------------------------------------------------------------

def test_b7_filters_flag_degenerate_outputs():
    tiny = [torch.full((4, 4), 0.001) for _ in range(5)]
    assert filter_output_range(tiny) is True
    assert filter_output_std(tiny) is True
    assert filter_input_impact(tiny) is True


def test_b7_filters_pass_varied_outputs():
    torch.manual_seed(0)
    varied = [torch.randn(4, 4) * 100.0 for _ in range(5)]
    assert filter_output_range(varied) is False
    assert filter_output_std(varied) is False
    assert filter_output_axes(varied) is False


def test_b7_axes_filter_catches_a_constant_axis():
    torch.manual_seed(0)
    outputs = []
    for _ in range(5):
        # varies across seeds and rows, constant along the last axis
        column = torch.randn(4, 1) * 100.0
        outputs.append(column.expand(4, 6).contiguous())
    assert filter_output_axes(outputs) is True
    assert filter_output_std(outputs) is False


# ---------------------------------------------------------------------------
# B9 fp64 reference comparison (validator.rs semantics) + calibration
# ---------------------------------------------------------------------------

def test_b9_compare_failure_kinds():
    ref = torch.zeros(4)
    assert compare_to_fp64(torch.zeros(5), ref, 1e-5)["failure_kind"] == "shape"
    assert compare_to_fp64(torch.tensor([0.0, float("nan"), 0, 0]), ref,
                           1e-5)["failure_kind"] == "nan"
    assert compare_to_fp64(torch.tensor([0.0, float("inf"), 0, 0]), ref,
                           1e-5)["failure_kind"] == "inf"
    off = compare_to_fp64(torch.full((4,), 1e-3), ref, 1e-5)
    assert off["failure_kind"] == "tolerance"
    assert off["error_stats"]["num_exceeding"] == 4
    ok = compare_to_fp64(torch.full((4,), 1e-6), ref, 1e-5)
    assert ok["passed"] and ok["failure_kind"] is None


def test_b9_calibration_is_p95_times_factor():
    assert calibrate_tolerance([1e-6] * 20) == pytest.approx(1.5e-6)
    with pytest.raises(ValueError):
        calibrate_tolerance([])
    with pytest.raises(ValueError):
        calibrate_tolerance([float("nan")])


def test_b9_fp64_reference_rounds_back_to_requested_dtype():
    class Scale(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.5))

        def forward(self, x):
            return x * self.scale

    inputs = [torch.randn(8, dtype=torch.float32)]
    ideal = run_fp64_cpu_reference(Scale(), inputs, torch.float32)
    assert ideal.dtype == torch.float32
    # x * 1.5 is exactly representable through the double path
    assert torch.equal(ideal, inputs[0] * 1.5)


# ---------------------------------------------------------------------------
# End-to-end CPU smoke of the plan/run scheduler (synthetic KernelBench task)
# ---------------------------------------------------------------------------

TASK_SRC = textwrap.dedent(
    """
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.5))

        def forward(self, x):
            return x * self.scale

    def get_inputs():
        return [torch.randn(64, 128)]

    def get_init_inputs():
        return []
    """
)

EXACT_CANDIDATE_SRC = TASK_SRC.replace("class Model", "class ModelNew")

CLAMP_CANDIDATE_SRC = textwrap.dedent(
    """
    import torch
    import torch.nn as nn

    class ModelNew(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.5))

        def forward(self, x):
            return torch.clamp(x, -8.0, 8.0) * self.scale

    def get_inputs():
        return [torch.randn(64, 128)]
    """
)

OFFSET_CANDIDATE_SRC = EXACT_CANDIDATE_SRC.replace(
    "return x * self.scale", "return x * self.scale + 1e-3"
)


def _runner(tmp_path, candidate_src, name):
    task_path = tmp_path / "task.py"
    task_path.write_text(TASK_SRC, encoding="utf-8")
    candidate_path = tmp_path / f"{name}.py"
    candidate_path.write_text(candidate_src, encoding="utf-8")
    task_module = _load_module(task_path, f"e3_test_task_{name}")
    candidate_module = _load_module(candidate_path, f"e3_test_cand_{name}")
    return PlanRunner(task_module, candidate_module, device="cpu")


@pytest.mark.parametrize("baseline,kwargs,expected_runs", [
    ("b1u", {}, 5),
    ("b7", {"backward": True}, 32),
    ("b8", {}, 32),
    ("b9", {}, 32),
])
def test_exact_candidate_passes_every_baseline(tmp_path, baseline, kwargs,
                                               expected_runs):
    plan = build_plan(baseline, "SMOKE", **kwargs)
    runner = _runner(tmp_path, EXACT_CANDIDATE_SRC, f"exact_{baseline}")
    observations, budget = runner.run_plan(plan, fresh_budget_state(
        plan["candidate_run_budget"]))
    assert budget.candidate_runs == expected_runs
    statuses = {obs["status"] for obs in observations}
    assert statuses == {"PASS"}, observations


def test_b8_outlier_family_catches_a_clamp_bug(tmp_path):
    plan = build_plan("b8", "SMOKE")
    runner = _runner(tmp_path, CLAMP_CANDIDATE_SRC, "clamp")
    observations, _ = runner.run_plan(plan, fresh_budget_state())
    by_policy = {}
    for obs in observations:
        by_policy.setdefault(obs["policy"], []).append(obs["status"])
    assert set(by_policy["b8_standard"]) == {"PASS"}
    assert "FAIL" in by_policy["b8_outlier"]


def test_b9_fp64_reference_catches_a_constant_offset(tmp_path):
    plan = build_plan("b9", "SMOKE")
    runner = _runner(tmp_path, OFFSET_CANDIDATE_SRC, "offset")
    observations, _ = runner.run_plan(plan, fresh_budget_state())
    assert all(obs["status"] == "FAIL" for obs in observations)
    assert all(obs["b9"]["failure_kind"] == "tolerance" for obs in observations)


def test_runner_charges_before_running_and_stops_on_overrun(tmp_path):
    plan = build_plan("b8", "SMOKE")
    plan["cases"] = plan["cases"] + [plan["cases"][0]]  # 33 invocations
    runner = _runner(tmp_path, EXACT_CANDIDATE_SRC, "overrun")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        runner.run_plan(plan, fresh_budget_state())
