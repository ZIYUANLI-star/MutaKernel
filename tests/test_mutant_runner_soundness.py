"""CPU regression tests for sound Phase-I mutant classification."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from src.models import KernelInfo, Mutant, MutantStatus, MutationSite
from src.mutengine.mutant_runner import MutantRunner


class _ReferenceLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class _ReferenceCrash(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("reference contract rejected this input")


def _kernel() -> KernelInfo:
    return KernelInfo(
        problem_id=1,
        level=2,
        problem_name="soundness_test",
        source_path="",
        kernel_code="",
        reference_module_path="",
        language="python",
    )


def _mutant(source: str) -> Mutant:
    return Mutant(
        id="soundness-1",
        operator_name="test",
        operator_category="A",
        site=MutationSite(
            line_start=1,
            line_end=1,
            original_code="",
            node_type="test",
        ),
        original_code="",
        mutated_code=source,
    )


def _run(reference_class: type[nn.Module], candidate_source: str) -> Mutant:
    runner = MutantRunner(
        atol=0.0,
        rtol=0.0,
        num_test_inputs=2,
        device="cpu",
        seed=17,
    )
    try:
        return runner.run_mutant(
            _kernel(),
            _mutant(candidate_source),
            SimpleNamespace(Model=reference_class),
            get_inputs_fn=lambda: [torch.arange(6, dtype=torch.float32).reshape(2, 3)],
            get_init_inputs_fn=lambda: [],
        )
    finally:
        runner.cleanup()


def test_separately_initialized_stateful_models_are_synchronized() -> None:
    candidate = """
import torch
from torch import nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(value)
"""

    result = _run(_ReferenceLinear, candidate)

    assert result.status is MutantStatus.SURVIVED
    assert all(
        trial["status"] == "pass"
        for trial in result.equiv_detail["phase1_validation"]["trials"]
    )


def test_real_output_divergence_is_killed() -> None:
    candidate = """
import torch
from torch import nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(value) + 1
"""

    result = _run(_ReferenceLinear, candidate)

    assert result.status is MutantStatus.KILLED
    assert result.kill_input_seed == 17


def test_reference_failure_is_inconclusive_not_a_false_kill() -> None:
    candidate = """
import torch
from torch import nn

class ModelNew(nn.Module):
    def forward(self, value):
        return value
"""

    result = _run(_ReferenceCrash, candidate)

    assert result.status is MutantStatus.UNKNOWN
    assert result.kill_input_seed is None
    assert all(
        trial["status"] == "inconclusive"
        for trial in result.equiv_detail["phase1_validation"]["trials"]
    )


def test_candidate_failure_when_reference_succeeds_is_killed() -> None:
    candidate = """
import torch
from torch import nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        raise RuntimeError("candidate crash")
"""

    result = _run(_ReferenceLinear, candidate)

    assert result.status is MutantStatus.KILLED


def test_renamed_state_schema_is_aligned_and_survives() -> None:
    # An unambiguous rename (``linear`` -> ``renamed``) is aligned by the
    # name-normalizing state sync and validated normally.
    candidate = """
import torch
from torch import nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.renamed = nn.Linear(3, 2)

    def forward(self, value):
        return self.renamed(value)
"""

    result = _run(_ReferenceLinear, candidate)

    assert result.status is MutantStatus.SURVIVED


def test_ambiguous_state_schema_is_inconclusive() -> None:
    # Two same-shape buffers with unrelated names cannot be aligned without
    # guessing; the sync must refuse and the mutant stays UNKNOWN.
    candidate = """
import torch
from torch import nn

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("mystery_one", torch.zeros(4))
        self.register_buffer("mystery_two", torch.ones(4))

    def forward(self, value):
        return self.linear(value)
"""

    result = _run(_ReferenceLinear, candidate)

    assert result.status is MutantStatus.UNKNOWN
    assert "inconclusive" in result.error_message.lower()
